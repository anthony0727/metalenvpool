"""Compile packaged Metal shader sources with Apple's Metal Toolchain."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

SHADER_SOURCES = ("atari_preprocess.metal", "mpe_simple_step.metal", "point_step.metal")


@dataclass(frozen=True)
class CompiledShader:
    source: str
    air_path: Path
    metallib_path: Path


def compile_packaged_shaders(output_dir: str | Path) -> list[CompiledShader]:
    """Compile all packaged MSL sources to AIR and metallib files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compiled = []
    for source in SHADER_SOURCES:
        compiled.append(_compile_one(source, out))
    return compiled


def _compile_one(source: str, output_dir: Path) -> CompiledShader:
    source_ref = resources.files("metalenvpool.native").joinpath(source)
    stem = Path(source).stem
    air_path = output_dir / f"{stem}.air"
    metallib_path = output_dir / f"{stem}.metallib"

    with resources.as_file(source_ref) as source_path:
        _run(("xcrun", "metal", "-c", str(source_path), "-o", str(air_path)))
    _run(("xcrun", "metallib", str(air_path), "-o", str(metallib_path)))
    return CompiledShader(source=source, air_path=air_path, metallib_path=metallib_path)


def _run(argv: tuple[str, ...]) -> None:
    proc = subprocess.run(argv, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        details = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(argv)}\n{details}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/metalenvpool-native")
    args = parser.parse_args()

    for item in compile_packaged_shaders(args.output_dir):
        print(f"{item.source}: {item.metallib_path}")


if __name__ == "__main__":
    main()
