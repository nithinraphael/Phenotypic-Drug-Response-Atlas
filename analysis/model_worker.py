import json
import sys
from pathlib import Path

import numpy as np

from analysis.models import tabpfn_config


def main():
    model, directory = sys.argv[1:]
    root = Path(directory)
    config = json.loads((root / 'config.json').read_text())
    np.random.seed(config['random_seed'])
    with np.load(root / 'input.npz', allow_pickle=False) as data:
        x, y, grid = data['x'], data['y'], data['grid']
    if model == 'TabPFN':
        verified = tabpfn_config(root / 'config.json')
        if verified.is_error():
            (root / 'error.txt').write_text(verified.error)
            return
        from tabpfn import TabPFNRegressor
        import torch
        torch.manual_seed(config['random_seed'])
        def features(dose):
            return np.log10(dose).reshape(-1, 1) if config['feature_definition'] == 'log10_dose' else np.column_stack([np.log10(dose), dose])
        reg = TabPFNRegressor(model_path=config['checkpoint_path'], device=config['device'],
                              inference_precision=config['precision'], fit_mode='low_memory', random_state=config['random_seed'])
        reg.fit(features(x), y)
        pred = reg.predict(features(grid))
    else:
        import piecewise_regression
        fit = piecewise_regression.Fit(np.log10(x), y, n_breakpoints=1, n_boot=20, verbose=False)
        if not fit.get_results()['converged']:
            (root / 'error.txt').write_text('Segmented optimization failed')
            return
        pred = fit.predict(np.log10(grid))
    np.save(root / 'predictions.npy', pred, allow_pickle=False)


if __name__ == '__main__':
    main()
