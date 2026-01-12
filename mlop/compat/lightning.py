"""
Backward compatibility shim for mlop.compat.lightning -> pluto.compat.lightning migration.
This module is deprecated. Use `import pluto.compat.lightning` instead.
"""
import warnings

warnings.warn(
    "The 'mlop.compat.lightning' module is deprecated. "
    "Please use 'import pluto.compat.lightning' instead.",
    DeprecationWarning,
    stacklevel=2
)

from pluto.compat.lightning import *  # noqa: F401, F403
