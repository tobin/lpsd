from ._version import get_versions  # noqa: D104

__version__ = get_versions()["version"]
del get_versions

from .lpsd import lpsd  # noqa: F401
