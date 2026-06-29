from ._native import *  # type: ignore[reportMissingImports] # noqa: F403

try:
    from ._native import __doc__ as __doc__  # type: ignore[reportMissingImports]
except ImportError:
    __doc__ = None
