import sys
import types
import unittest
from unittest import mock

from sgl_deep_ep.prerequisites import check_prerequisites


class PrerequisitesTest(unittest.TestCase):
    def test_gdrcopy_is_not_required_at_import_time(self):
        torch = types.SimpleNamespace(
            version=types.SimpleNamespace(cuda="13.0"),
            cuda=types.SimpleNamespace(is_available=lambda: True),
        )

        with (
            mock.patch.dict(sys.modules, {"torch": torch}),
            mock.patch(
                "sgl_deep_ep.prerequisites.platform.system", return_value="Linux"
            ),
            mock.patch(
                "sgl_deep_ep.prerequisites.platform.machine", return_value="x86_64"
            ),
            mock.patch(
                "ctypes.CDLL", side_effect=OSError("libgdrapi.so is missing")
            ),
            mock.patch("pathlib.Path.exists", return_value=False),
        ):
            check_prerequisites(expected_cuda_major=13)


if __name__ == "__main__":
    unittest.main()
