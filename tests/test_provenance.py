import tempfile
import unittest
from pathlib import Path

from analysis.provenance import manifest,sha256,verify_manifest


class ProvenanceTests(unittest.TestCase):
    def test_hash_config_version_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'source';p.write_text('abc')
            self.assertEqual(sha256(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            m=manifest('test',[p],{'seed':42})
            self.assertTrue(verify_manifest(m,[p],{'seed':42}).is_ok())
            self.assertTrue(verify_manifest(m,[p],{'seed':1}).is_error())
            self.assertTrue(verify_manifest(dict(m,software_versions={}),[p],{'seed':42}).is_error())
            self.assertTrue(verify_manifest(dict(m,status='failed'),[p],{'seed':42}).is_error())
            p.write_text('changed')
            self.assertTrue(verify_manifest(m,[p],{'seed':42}).is_error())
