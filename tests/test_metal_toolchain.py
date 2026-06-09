import shutil
import subprocess

import pytest

from metalenvpool.native.build import SHADER_SOURCES, compile_packaged_shaders


def metal_toolchain_available() -> bool:
    if shutil.which("xcrun") is None:
        return False
    return subprocess.run(("xcrun", "metal", "-v"), capture_output=True, check=False).returncode == 0


@pytest.mark.skipif(not metal_toolchain_available(), reason="Metal Toolchain is not available")
def test_metal_toolchain_compiles_packaged_shaders(tmp_path):
    compiled = compile_packaged_shaders(tmp_path)

    assert {item.source for item in compiled} == set(SHADER_SOURCES)
    for item in compiled:
        assert item.air_path.exists()
        assert item.metallib_path.exists()
        assert item.metallib_path.stat().st_size > 0
