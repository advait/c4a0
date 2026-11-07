from ._native import *  # noqa: F403  # Re-export the extension's public API.

try:
    from ._native import __doc__ as __doc__
except ImportError:
    __doc__ = None
