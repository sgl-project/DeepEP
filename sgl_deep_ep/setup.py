import os
import re
import runpy
from pathlib import Path

import setuptools


def _run_deepep_setup() -> None:
    version = os.environ.get("SGL_DEEP_EP_VERSION")
    if not version:
        raise RuntimeError("SGL_DEEP_EP_VERSION must be set")

    original_setup = setuptools.setup

    def sgl_setup(**kwargs):
        requirements = [
            requirement
            for requirement in kwargs.get("install_requires") or []
            if not re.match(r"\s*torch(?:\s|[<>=!~;@\[]|$)", requirement, re.IGNORECASE)
        ]
        requirements.append("torch==2.13.0")

        kwargs.update(
            name="sgl-deep-ep",
            version=version,
            python_requires=">=3.10",
            long_description=Path("README.md").read_text(),
            long_description_content_type="text/markdown",
            install_requires=requirements,
            license_files=("LICENSE",),
        )
        return original_setup(**kwargs)

    setuptools.setup = sgl_setup
    try:
        runpy.run_path("_deepep_setup.py", run_name="__main__")
    finally:
        setuptools.setup = original_setup


if __name__ == "__main__":
    _run_deepep_setup()
