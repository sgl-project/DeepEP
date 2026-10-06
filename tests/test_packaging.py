"""Exercise wheel staging without requiring CUDA compilation or a GPU."""

import email
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


class PackagingTest(unittest.TestCase):

    def test_build_from_single_checkout(self):
        # The upstream setup is a CPU-only stand-in; staging, metadata rewriting,
        # and wheel creation all use the real packaging implementation.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            shutil.copytree(Path(__file__).resolve().parents[1] / 'sgl_deep_ep', source / 'sgl_deep_ep')
            package = source / 'deep_ep'
            package.mkdir()
            original_init = "__version__ = '2.1.0'\n"
            (package / '__init__.py').write_text(original_init)
            (source / 'setup.py').write_text("from setuptools import setup\n"
                                             "setup(name='deep_ep', version='2.1.0', packages=['deep_ep'], "
                                             "install_requires=['torch>=2', 'pynvml', 'numpy'])\n")
            (source / 'build/lib/deep_ep').mkdir(parents=True)
            (source / 'build/lib/deep_ep/stale.py').write_text('stale = True\n')
            metadata = root / 'torch-2.13.0.dist-info'
            metadata.mkdir()
            (metadata / 'METADATA').write_text('Name: torch\nVersion: 2.13.0+cu130\n')
            binaries = root / 'bin'
            binaries.mkdir()
            for name, body in [('nvcc', 'exit 0'), ('uname', 'echo x86_64')]:
                executable = binaries / name
                executable.write_text('#!/bin/sh\n' + body + '\n')
                executable.chmod(0o755)
            cuda = root / 'cuda'
            (cuda / 'include/cccl').mkdir(parents=True)
            env = dict(os.environ,
                       PATH=f'{binaries}:{os.environ["PATH"]}',
                       PYTHON_BIN=sys.executable,
                       PYTHONPATH=str(root),
                       CUDA_HOME=str(cuda),
                       SGL_DEEP_EP_VERSION='0.1.3')
            result = subprocess.run(
                ['bash', str(source / 'sgl_deep_ep/build_sgl_deep_ep.sh'),
                 str(source / 'dist'), '13.0', 'x86_64'],
                cwd=root,
                env=env,
                capture_output=True,
                text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            wheels = list((source / 'dist').glob('*.whl'))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as wheel:
                names = wheel.namelist()
                info = email.message_from_bytes(wheel.read(next(name for name in names if name.endswith('/METADATA'))))
                self.assertEqual(info['Name'], 'sgl-deep-ep')
                self.assertEqual(info['Version'], '0.1.3+cu130')
                self.assertCountEqual(info.get_all('Requires-Dist'), ['numpy', 'nvidia-ml-py', 'torch==2.13.0'])
                self.assertIn(b'EXPECTED_CUDA_MAJOR = 13', wheel.read('deep_ep/_build_info.py'))
                self.assertIn(b'check_prerequisites', wheel.read('deep_ep/__init__.py'))
                self.assertIn(original_init.encode(), wheel.read('deep_ep/__init__.py'))
                self.assertNotIn('deep_ep/stale.py', names)
                self.assertFalse(any(name.startswith('sgl_deep_ep/') for name in names))
            self.assertEqual((package / '__init__.py').read_text(), original_init)
            self.assertFalse((source / '_deepep_setup.py').exists())


if __name__ == '__main__':
    unittest.main()
