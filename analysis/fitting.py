from dataclasses import dataclass, field

import numpy as np
from expression import Error, Ok
from thunor.curve_fit import HillCurveLL4, HillCurveNull, fit_drc


@dataclass(frozen=True)
class FitResult:
    status: str
    reason: str | None = None
    ec50: float | None = None
    hill_slope: float | None = None
    lower_asymptote: float | None = None
    upper_asymptote: float | None = None
    r2: float | None = None
    rmse: float | None = None
    predictions: np.ndarray | None = None
    ec20: float | None = None
    ec80: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def predict(self, doses):
        if self.status != 'ok':
            return Error(self.reason)
        x = np.asarray(doses)
        if x.ndim != 1 or x.dtype.kind not in 'iuf' or not np.all(np.isfinite(x)) or np.any(x <= 0):
            return Error('Prediction concentrations must be finite and positive')
        y = HillCurveLL4.fit_fn(x, self.hill_slope, self.lower_asymptote, self.upper_asymptote, self.ec50)
        return Ok(y) if np.all(np.isfinite(y)) else Error('Nonfinite model predictions')


def validate_observations(doses, responses):
    x, y = np.asarray(doses), np.asarray(responses)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        return Error('Nonempty one-dimensional arrays of equal length required')
    if x.dtype.kind not in 'iuf' or y.dtype.kind not in 'iuf':
        return Error('Numeric concentrations and responses required')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return Error('Nonfinite observations')
    if np.any(x <= 0):
        return Error('Concentrations must be strictly positive')
    return Ok((x.astype(float), y.astype(float)))


def fit_hill(doses, responses):
    validated = validate_observations(doses, responses)
    if validated.is_error():
        return FitResult('failed', validated.error)
    x, y = validated.ok
    diagnostics = {'n_observations': len(x), 'n_concentrations': len(np.unique(x)),
                   'min_dose_um': float(x.min()), 'max_dose_um': float(x.max()), 'flags': []}
    if len(np.unique(x)) < 4 or len(x) < 5:
        return FitResult('failed', 'At least four distinct concentrations and five observations required', diagnostics=diagnostics)
    if np.ptp(y) == 0:
        return FitResult('failed', 'Constant response; R² undefined', diagnostics=diagnostics)
    fit = fit_drc(x, y, null_rejection_threshold=None)
    if fit is None or isinstance(fit, HillCurveNull):
        return FitResult('failed', 'LL4 optimization failed', diagnostics=diagnostics)
    params = np.asarray(fit.popt)
    if params.shape != (4,) or not np.all(np.isfinite(params)) or params[3] <= 0:
        return FitResult('failed', 'Invalid fitted parameters', diagnostics=diagnostics)
    slope, lower, upper, ec50 = map(float, params)
    if slope == 0 or np.isclose(lower, upper, rtol=1e-8, atol=1e-10):
        return FitResult('failed', 'Degenerate asymptotes or slope', diagnostics=diagnostics)
    if slope < 0:
        slope, lower, upper = -slope, upper, lower
    pred = HillCurveLL4.fit_fn(x, slope, lower, upper, ec50)
    if not np.all(np.isfinite(pred)):
        return FitResult('failed', 'Nonfinite fitted predictions', diagnostics=diagnostics)
    sse, sst = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
    flags = diagnostics['flags']
    if upper <= lower:
        flags.append('Increasing response; inhibition range undefined')
    with np.errstate(over='ignore'):
        ec20, ec80 = ec50 * np.exp(np.log([0.25, 4.0]) / slope)
    if upper <= lower or not np.all(np.isfinite([ec20, ec80])) or ec20 <= 0:
        ec20, ec80 = None, None
    else:
        ec20, ec80 = float(ec20), float(ec80)
        if ec20 < x.min() or ec80 > x.max():
            flags.append('EC20–EC80 extends beyond observed doses')
        if np.count_nonzero((np.unique(x) >= ec20) & (np.unique(x) <= ec80)) < 2:
            flags.append('Transition sampled by fewer than two concentrations')
    if not x.min() <= ec50 <= x.max():
        flags.append('EC50 outside observed doses')
    if slope >= 20:
        flags.append('Extreme slope; inspect identifiability')
    diagnostics['parameter_bounds'] = 'LL4 unbounded; EC50 log parameterization; no imposed slope bound'
    jac = HillCurveLL4.jac_fn_log(x, slope, lower, upper, np.log(ec50))
    condition = float(np.linalg.cond(jac))
    diagnostics['jacobian_condition'] = condition if np.isfinite(condition) else None
    if not np.isfinite(condition) or condition > 1e8:
        flags.append('Ill-conditioned parameter Jacobian')
    diagnostics['perturbation_stability'] = 'Not evaluated'
    return FitResult('ok', ec50=ec50, hill_slope=slope, lower_asymptote=lower,
                     upper_asymptote=upper, r2=float(1 - sse / sst), rmse=float(np.sqrt(sse / len(x))),
                     predictions=pred, ec20=ec20, ec80=ec80, diagnostics=diagnostics)
