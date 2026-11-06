#!/usr/bin/env python3
"""Print libclang paths for bindgen.

GitHub runners usually provide libclang under /usr/lib/llvm-*/lib. Some local
machines may have LLVM installed by mise under /opt/mise/installs/clang.
This helper keeps mise.toml portable without pinning a flaky clang plugin.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

PATTERNS = [
    "/usr/lib/llvm-*/lib/libclang.so*",
    "/usr/local/lib/libclang.so*",
    "/usr/lib/x86_64-linux-gnu/libclang.so*",
    "/opt/mise/installs/clang/*/lib/libclang.so*",
    "/opt/mise/installs/vfox-clang/*/envs/clang/lib/libclang.so*",
    # Installed by `mise run bootstrap` when none of the above exist.
    str(Path(__file__).resolve().parent.parent / ".mise/libclang/clang/native/libclang.so*"),
]

# The libclang wheel ships no clang resource headers (stdbool.h etc.); fall back to GCC's.
GCC_INCLUDE_PATTERN = "/usr/lib/gcc/*/*/include/stdbool.h"


def version_key(path: Path) -> tuple[int, ...]:
    parts: list[int] = []
    for token in path.as_posix().replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
    return tuple(parts)


def find_libclang_dir() -> Path | None:
    candidates = sorted(
        {Path(match).resolve() for pattern in PATTERNS for match in glob.glob(pattern)},
        key=version_key,
    )
    if candidates:
        return candidates[-1].parent
    return None


def find_clang_resource_include(libclang_dir: Path) -> Path | None:
    candidates = sorted(
        libclang_dir.glob("clang/*/include/stdbool.h"),
        key=version_key,
    )
    if candidates:
        return candidates[-1].parent
    gcc = sorted((Path(m) for m in glob.glob(GCC_INCLUDE_PATTERN)), key=version_key)
    if gcc:
        return gcc[-1].parent
    return None


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bindgen-extra-clang-args",
    action="store_true",
    help="Print BINDGEN_EXTRA_CLANG_ARGS for Clang's bundled C headers.",
)
args = parser.parse_args()

libclang_dir = find_libclang_dir()
if not libclang_dir:
    raise SystemExit(0)

if args.bindgen_extra_clang_args:
    include_dir = find_clang_resource_include(libclang_dir)
    if include_dir:
        print(f"-isystem {include_dir}")
else:
    print(libclang_dir)
