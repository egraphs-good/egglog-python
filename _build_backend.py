"""PEP 517 builds export the native builtin catalog before packaging."""

import importlib.machinery
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import maturin

_ROOT = Path(__file__).resolve().parent


# All three operations compile a bootstrap wheel and use the same build inputs.
get_requires_for_build_wheel = maturin.get_requires_for_build_wheel
get_requires_for_build_sdist = get_requires_for_build_wheel
get_requires_for_build_editable = get_requires_for_build_wheel
prepare_metadata_for_build_wheel = maturin.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = maturin.prepare_metadata_for_build_editable


def _bootstrap(config_settings):
    # The first wheel supplies only the native transport. Never import egglog:
    # its packaged catalog does not exist yet in a fresh checkout.
    with TemporaryDirectory() as temporary:
        filename = maturin.build_wheel(temporary, config_settings)
        with ZipFile(Path(temporary) / filename) as wheel:
            extensions = [
                name
                for name in wheel.namelist()
                if name.startswith("egglog/bindings") and name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
            ]
            if len(extensions) != 1:
                raise RuntimeError(f"Expected one native bindings extension, got {extensions}")
            extension = Path(wheel.extract(extensions[0], temporary))
        subprocess.run(
            [sys.executable, str(_ROOT / "scripts/export_builtin_catalog.py"), "--extension", str(extension)],
            cwd=_ROOT,
            check=True,
        )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _bootstrap(config_settings)
    # Cargo reuses the bootstrap compilation; maturin packages the fresh catalog.
    return maturin.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _bootstrap(config_settings)
    return maturin.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _bootstrap(config_settings)
    return maturin.build_editable(wheel_directory, config_settings, metadata_directory)
