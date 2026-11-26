"""面向 Web/外部调用的“重排配水”便捷入口。

历史回放逻辑集中在 :mod:`greenhouse.batch_replay`，流水线集成在
:func:`greenhouse.pipeline.run`。本模块提供稳定的函数式入口，方便不同调用
习惯（直接给额度、给 run 结果、Flask 响应字典）复用同一实现。
"""

from __future__ import annotations

from typing import Any


def improve_snapshot(capacity_or_snapshot: Any = None, **kwargs: Any) -> dict:
    """返回带历史回放重排结果的完整 snapshot。

    * 传入整数额度：等价于 ``run(capacity)``；
    * 传入已经计算好的 run/snapshot 字典：若已含 ``batch_replay`` 直接返回，
      否则按其中可推断的额度重新计算一次。
    """
    from .pipeline import run

    if isinstance(capacity_or_snapshot, dict):
        snapshot = capacity_or_snapshot
        if "batch_replay" in snapshot:
            return snapshot
        capacity = kwargs.get("capacity")
        if capacity is None:
            metrics = snapshot.get("metrics", {})
            capacity = metrics.get("班次额度")
        return run(capacity)
    capacity = capacity_or_snapshot
    if capacity is None:
        capacity = kwargs.get("capacity")
    return run(capacity)


# 常见命名别名
rerun_with_replay = improve_snapshot
replay_snapshot = improve_snapshot
redispatch = improve_snapshot
