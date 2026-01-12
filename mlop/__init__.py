"""
Backward compatibility shim for mlop -> pluto migration.
This module is deprecated. Use `import pluto` instead.
"""
import warnings

warnings.warn(
    "The 'mlop' package is deprecated and will be removed in a future release. "
    "Please use 'import pluto' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from pluto
from pluto import *  # noqa: F401, F403
from pluto import __version__, __all__  # noqa: F401
