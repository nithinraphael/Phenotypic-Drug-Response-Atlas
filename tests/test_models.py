import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from analysis.models import isolated_predict,tabpfn_config
from analysis.provenance import sha256


class ModelTests(unittest.TestCase):
    def test_checkpoint_identity_package_and_device_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'test.ckpt';checkpoint.write_bytes(b'synthetic test checkpoint; not a pretrained model')
            config=dict(model_family='TabPFN',model_version='test-only',package_version=importlib.metadata.version('tabpfn'),
                        checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),device='cpu',precision='auto',
                        feature_definition='log10_dose',random_seed=42)
            path=Path(directory)/'config.json';path.write_text(json.dumps(config))
            self.assertTrue(tabpfn_config(path).is_ok())
            for key,value in [('package_version','unsupported'),('checkpoint_sha256','wrong'),('feature_definition','unknown')]:
                path.write_text(json.dumps(dict(config,**{key:value})))
                self.assertTrue(tabpfn_config(path).is_error())
            path.write_text(json.dumps(dict(config,device='cuda')))
            with patch('torch.cuda.is_available',return_value=False):
                self.assertTrue(tabpfn_config(path).is_error())

    def test_worker_failure_never_returns_substitute_predictions(self):
        with patch('analysis.models.subprocess.run',return_value=SimpleNamespace(returncode=1)):
            result=isolated_predict('TabPFN',[.1,1,10,100],[1,.9,.2,.1],[2.],{})
        self.assertTrue(result.is_error())
        self.assertIn('worker failed',result.error)
