#!/usr/bin/env python3
"""Print a directory containing libclang for bindgen.

GitHub runners usually provide libclang under /usr/lib/llvm-*/lib. Some local
machines may have LLVM installed by mise under /opt/mise/installs/clang.
This helper keeps mise.toml portable without pinning a flaky clang plugin.
"""

from __future__ import annotations

import glob
from pathlib import Path

PATTERNS = [
    "/usr/lib/llvm-*/lib/libclang.so*",
    "/usr/local/lib/libclang.so*",
    "/usr/lib/x86_64-linux-gnu/libclang.so*",
    "/opt/mise/installs/clang/*/lib/libclang.so*",
    "/opt/mise/installs/vfox-clang/*/envs/clang/lib/libclang.so*",
]


def version_key(path: Path) -> tuple[int, ...]:
    parts: list[int] = []
    for token in path.as_posix().replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
    return tuple(parts)


candidates = sorted(
    {Path(match).resolve() for pattern in PATTERNS for match in glob.glob(pattern)},
    key=version_key,
)

if candidates:
    print(candidates[-1].parent)
