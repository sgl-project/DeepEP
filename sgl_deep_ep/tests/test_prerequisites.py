import importlib
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


class PrerequisitesTest(unittest.TestCase):
    def test_deep_ep_import_does_not_require_gdrcopy(self):
        torch = types.SimpleNamespace(
            version=types.SimpleNamespace(cuda="13.0"),
            cuda=types.SimpleNamespace(is_available=lambda: True),
        )

        overlay_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "deep_ep"
            package_dir.mkdir()
            shutil.copy2(overlay_dir / "__init__.py", package_dir / "__init__.py")
            shutil.copy2(
                overlay_dir / "prerequisites.py", package_dir / "prerequisites.py"
            )
            (package_dir / "_build_info.py").write_text(
                "EXPECTED_CUDA_MAJOR = 13\nCUDA_TAG = 'cu130'\n"
            )

            try:
                with (
                    mock.patch.dict(sys.modules, {"torch": torch}),
                    mock.patch.object(sys, "path", [temp_dir, *sys.path]),
                    mock.patch("platform.system", return_value="Linux"),
                    mock.patch("platform.machine", return_value="x86_64"),
                    mock.patch(
                        "ctypes.CDLL", side_effect=OSError("libgdrapi.so is missing")
                    ),
                    mock.patch("pathlib.Path.exists", return_value=False),
                ):
                    module = importlib.import_module("deep_ep")
                    self.assertEqual(module.__name__, "deep_ep")
            finally:
                for module_name in tuple(sys.modules):
                    if module_name == "deep_ep" or module_name.startswith("deep_ep."):
                        del sys.modules[module_name]


if __name__ == "__main__":
    unittest.main()
