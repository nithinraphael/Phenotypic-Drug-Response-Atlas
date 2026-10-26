import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from expression import Error, Ok

PACKAGES = ('marimo', 'polars', 'numpy', 'scipy', 'thunor', 'tabpfn', 'torch', 'torchvision', 'umap-learn', 'plotly', 'expression', 'tifffile', 'scikit-learn', 'piecewise-regression', 'pillow', 'kaleido', 'pyreadr')
SCHEMA_VERSION = 1


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def versions():
    return {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in PACKAGES}}


def code_hashes():
    return {str(p): sha256(p) for p in sorted(Path('analysis').glob('*.py'))}


def manifest(analysis, inputs, config, status='completed', **counts):
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=False)
    return dict(schema_version=SCHEMA_VERSION, analysis=analysis, status=status,
                inputs={str(p): sha256(p) for p in inputs}, config=config,
                software_versions=versions(), created_at=datetime.now(timezone.utc).isoformat(),
                code_revision=revision.stdout.strip() or None, working_tree_dirty=bool(dirty.stdout), analysis_code_hashes=code_hashes(), **counts)


def verify_manifest(metadata, inputs, config):
    if metadata.get('schema_version') != SCHEMA_VERSION:
        return Error('Unsupported provenance schema')
    if metadata.get('status') != 'completed':
        return Error('Analysis did not complete')
    if metadata.get('config') != config:
        return Error('Configuration mismatch')
    for path in inputs:
        if not Path(path).is_file():
            return Error(f'Missing input: {path}')
        if metadata.get('inputs', {}).get(str(path)) != sha256(path):
            return Error(f'Input hash mismatch: {path}')
    if metadata.get('analysis_code_hashes') != code_hashes():
        return Error('Analysis implementation mismatch')
    if metadata.get('software_versions') != versions():
        return Error('Software version mismatch')
    return Ok(metadata)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
