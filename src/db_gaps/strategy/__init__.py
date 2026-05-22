from .base import Strategy  # noqa: F401
from .registry import register, get, available  # noqa: F401
from . import builtin  # noqa: F401 - side-effect: register defaults
from . import dummy_6040  # noqa: F401 - register dummy_6040
