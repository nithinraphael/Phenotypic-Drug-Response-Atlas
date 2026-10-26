import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from expression import Error, Ok

from analysis.fitting import validate_observations
from analysis.provenance import sha256


def tabpfn_config(path='config/tabpfn.json'):
    if not Path(path).is_file():
        return Error('TabPFN unavailable: no explicit local checkpoint configuration')
    config = json.loads(Path(path).read_text())
    required = {'model_family', 'model_version', 'package_version', 'checkpoint_path', 'checkpoint_sha256', 'device', 'precision', 'feature_definition', 'random_seed'}
    if not required.issubset(config):
        return Error('Incomplete TabPFN configuration')
    if config['model_family'] != 'TabPFN' or config['package_version'] != importlib.metadata.version('tabpfn'):
        return Error('Unsupported TabPFN family or package version')
    if config['feature_definition'] not in ('log10_dose', 'log10_and_raw_dose') or config['device'] not in ('cpu', 'cuda') or config['precision'] not in ('auto', 'autocast'):
        return Error('Unsupported TabPFN features, device or precision')
    checkpoint = Path(config['checkpoint_path'])
    if not checkpoint.is_file():
        return Error(f'TabPFN checkpoint missing: {checkpoint}')
    if sha256(checkpoint) != config['checkpoint_sha256']:
        return Error('TabPFN checkpoint hash mismatch')
    if config['device'] == 'cuda':
        import torch
        if not torch.cuda.is_available():
            return Error('Configured CUDA device unavailable; no device substitution')
    return Ok(config)


def isolated_predict(model, x, y, grid, config):
    validated = validate_observations(x, y)
    if validated.is_error():
        return validated
    x, y = validated.ok
    grid_check = validate_observations(grid, np.zeros(len(grid)))
    if grid_check.is_error():
        return grid_check
    if len(np.unique(x)) < 4:
        return Error('Insufficient training concentrations')
    with TemporaryDirectory() as directory:
        root = Path(directory)
        np.savez(root / 'input.npz', x=x, y=y, grid=grid)
        (root / 'config.json').write_text(json.dumps(config))
        completed = subprocess.run([sys.executable, '-m', 'analysis.model_worker', model, str(root)], capture_output=True, text=True)
        if completed.returncode:
            return Error(f'{model} worker failed (exit {completed.returncode}); predictions unavailable')
        if (root / 'error.txt').exists():
            return Error((root / 'error.txt').read_text())
        if not (root / 'predictions.npy').exists():
            return Error(f'{model} produced no predictions')
        pred = np.load(root / 'predictions.npy', allow_pickle=False)
        if pred.shape != np.asarray(grid).shape or not np.all(np.isfinite(pred)):
            return Error('Invalid prediction shape or nonfinite predictions')
        return Ok(pred)


def predict_tabpfn(x, y, grid, config):
    if config is None:
        return Error('TabPFN not configured')
    return isolated_predict('TabPFN', x, y, grid, config)
