"""兼容入口：实现位于 :mod:`greenhouse.batch_replay`。"""

from ..batch_replay import *  # noqa: F401,F403
from .. import batch_replay as _core


def __getattr__(name):
    return getattr(_core, name)
