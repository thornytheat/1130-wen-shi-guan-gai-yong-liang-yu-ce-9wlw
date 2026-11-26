"""分区配水和泵组约束。"""


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    # 若名字对应真实子模块（scheduling / batch_replay / replay / ...），优先导入它
    import importlib

    full_name = f"{__name__}.{name}"
    try:
        return importlib.import_module(full_name)
    except ModuleNotFoundError as error:
        if error.name != full_name:
            raise
    # 其余名字委托给回放重排核心的宽松命名空间
    from .. import batch_replay as _core
    return getattr(_core, name)
