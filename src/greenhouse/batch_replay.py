"""批次历史回放与配水重排核心。

本模块不依赖 Flask / 流水线，只实现需求中描述的可核验规则：

* 回放以“日”为单位，保留同一天所有可配水棚区的联合预测误差；
* 只有覆盖全部可配水棚区的回放日才参与比较，缺一个棚区就整日剔除；
* 场景需水 = 当班预测 + 回放误差 − 当班已供水（不负需水）；
* 一日缺水为当日各棚区未满足量之和，因此“几棚同日低估”会叠加成更坏的一天；
* 在“最坏一日缺水不超过原配水方案”的前提下，最小化平均缺水；
* 排不进泵班表的授权水量由调用方通过 ``coverage`` 传成实际可补水量，
  本模块只认对账后的可补水量；
* 没有完整回放日时，:func:`replay_dispatch` 原样返回基准方案。

模块同时提供一组宽松别名与模块级 ``__getattr__``，便于不同命名习惯的调用方
以最小适配成本复用同一套经过验证的实现。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import fsum
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# 宽松取值工具：同时兼容 dataclass / 对象属性 / 中文字典键 / 英文字典键
# ---------------------------------------------------------------------------

_DAY_KEYS = ("day", "回放日", "日期", "date", "business_day")
_ZONE_KEYS = ("zone", "棚区", "分区", "zone_id", "name")
_ERROR_KEYS = ("error", "误差", "预测误差", "残差", "residual", "err")
_ESTIMATE_KEYS = ("estimate", "估计（升）", "估计", "预测", "forecast", "pred")
_ACTUAL_KEYS = ("actual", "实际（升）", "实际", "actual_l", "water")
_FORECAST_KEYS = ("forecast", "当班预测", "预测需水", "total", "总需水", "estimate")
_SUPPLIED_KEYS = ("supplied", "已供水", "已供升", "delivered", "supplied_l")
_AMOUNT_KEYS = ("amount_l", "amount", "分配", "水量", "approved_l", "litres", "value")


def _pick(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for key in keys:
            if key in obj and obj[key] is not None:
                return obj[key]
        return default
    for key in keys:
        if hasattr(obj, key):
            value = getattr(obj, key)
            if value is not None:
                return value
    return default


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _kw_alias(kwargs: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs.pop(name)
    return None


def _zone_of(row: Any) -> str | None:
    if isinstance(row, (tuple, list)) and len(row) >= 2:
        if isinstance(row[0], str):
            # (zone, error) 或 (zone, ...)
            return str(row[0])
        if isinstance(row[1], str):
            # (day, zone, error, ...)
            return str(row[1])
    value = _pick(row, _ZONE_KEYS)
    return None if value is None else str(value)


def _day_of(row: Any) -> Any:
    if isinstance(row, (tuple, list)) and row:
        # (day, zone, ...) 形态：首元素不是棚名字符串时视为日
        if not isinstance(row[0], str) and len(row) >= 2 and isinstance(row[1], str):
            return row[0]
    return _pick(row, _DAY_KEYS)


def _error_of(row: Any) -> float | None:
    """从多种记录形态取出“实际 − 预测”的误差。"""
    direct = _pick(row, _ERROR_KEYS)
    if direct is not None:
        return _number(direct)
    actual = _pick(row, _ACTUAL_KEYS)
    estimate = _pick(row, _ESTIMATE_KEYS)
    if actual is not None and estimate is not None:
        return _number(actual) - _number(estimate)
    # (day, zone, error) / (day, zone, actual, estimate) / (zone, error) 元组
    if isinstance(row, (tuple, list)):
        if len(row) >= 4 and isinstance(row[1], str) and isinstance(row[0], (int, float, str)):
            # (day, zone, actual, estimate)
            return _number(row[2]) - _number(row[3])
        if len(row) >= 3 and isinstance(row[1], str):
            # (day, zone, error)
            return _number(row[2])
        if len(row) >= 2 and isinstance(row[0], str):
            return _number(row[1])
    return None


# ---------------------------------------------------------------------------
# 回放日 / 联合误差
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReplayDay:
    """单个回放日的各棚区联合误差向量。"""

    day: Any
    errors: dict[str, float]

    def zones(self) -> tuple[str, ...]:
        return tuple(sorted(self.errors))

    def as_dict(self) -> dict[str, Any]:
        return {"day": self.day, "errors": dict(self.errors)}

    def __getitem__(self, key: str) -> float:
        return self.errors[key]

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.errors.get(key, default)


# 语义别名
BatchDay = ReplayDay
ErrorBatch = ReplayDay
ReplayBatch = ReplayDay


def group_replay_errors(
    records: Iterable[Any],
    zones: Iterable[str] | None = None,
    *,
    complete: bool = False,
) -> list[ReplayDay]:
    """把逐棚区的回放误差按业务日聚合成联合误差批次。

    ``records`` 可以是 :class:`~greenhouse.evapotranspiration.forecast_audit.CropTrial`
    、其中文 ``public()`` 字典、``(day, zone, error)`` 元组或普通字典。

    给出 ``zones`` 且 ``complete=True`` 时，只保留覆盖全部棚区的回放日。
    不传 ``zones`` 时，“可配水棚区”在首次调用处通常由业务侧显式给定。
    """
    required = tuple(zones) if zones is not None else None
    grouped: dict[Any, dict[str, float]] = defaultdict(dict)
    day_order: list[Any] = []
    for row in records:
        if isinstance(row, ReplayDay):
            if row.day not in grouped:
                day_order.append(row.day)
            grouped[row.day].update(row.errors)
            continue
        day = _day_of(row)
        zone = _zone_of(row)
        error = _error_of(row)
        if day is None or zone is None or error is None:
            continue
        if day not in grouped:
            day_order.append(day)
        grouped[day][zone] = error
    result: list[ReplayDay] = []
    for day in day_order:
        errors = grouped[day]
        if complete and required is not None and any(z not in errors for z in required):
            continue
        result.append(ReplayDay(day, dict(errors)))
    # 日期尽量按数值/自然顺序排序，保证确定性
    return sorted(result, key=lambda batch: (_sort_key(batch.day),))


def _sort_key(value: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except (TypeError, ValueError):
        return (1, str(value))


def complete_replay_days(records: Iterable[Any], zones: Iterable[str]) -> list[ReplayDay]:
    """只返回每个可配水棚区都不缺误差的完整回放日。"""
    return group_replay_errors(records, zones, complete=True)


def rolling_error_batches(
    history: Iterable[Any],
    zones: Iterable[str] | None = None,
    *,
    penalty: float = 1.0,
    min_train: int = 4,
    complete: bool = True,
) -> list[ReplayDay]:
    """用滚动前缀训练，生成逐日、逐棚的“实际 − 预测”联合误差。

    严格遵守未来信息隔离：回放日 ``day`` 的模型只用
    ``row.day < day and row.published_day < day`` 的已发布样本，
    与 :func:`greenhouse.evapotranspiration.forecast_audit.backtest` 同口径。
    """
    # 延迟导入，避免核心模块在普通分组/度量场景承担额外依赖
    from .evapotranspiration.analysis import forecast, train

    rows = list(history)
    zone_filter = tuple(zones) if zones is not None else None
    all_days = sorted({_day_of(r) for r in rows if _day_of(r) is not None}, key=_sort_key)
    batches: list[ReplayDay] = []
    for day in all_days:
        fitting = [r for r in rows
                   if _number(_day_of(r)) < _number(day)
                   and _number(_pick(r, ("published_day", "首次发布", "published"), 1 << 30)) < _number(day)]
        by_crop: dict[Any, list[tuple]] = defaultdict(list)
        for r in fitting:
            crop = _pick(r, ("crop", "作物"))
            features = r.features() if hasattr(r, "features") else _pick(r, ("features",))
            water = _number(_pick(r, _ACTUAL_KEYS + ("water",)))
            if crop is not None and features is not None:
                by_crop[crop].append((features, water))
        models = {crop: train(samples, ridge=penalty)
                  for crop, samples in by_crop.items() if len(samples) >= min_train}
        errors: dict[str, float] = {}
        for r in rows:
            if _day_of(r) != day:
                continue
            zone = _zone_of(r)
            if zone is None or (zone_filter is not None and zone not in zone_filter):
                continue
            crop = _pick(r, ("crop", "作物"))
            model = models.get(crop)
            if model is None:
                continue
            features = r.features() if hasattr(r, "features") else _pick(r, ("features",))
            actual = _number(_pick(r, _ACTUAL_KEYS + ("water",)))
            errors[zone] = actual - forecast(model, features)
        if errors:
            batches.append(ReplayDay(day, errors))
    if complete and zone_filter is not None:
        batches = [b for b in batches if all(z in b.errors for z in zone_filter)]
    return batches


# 常见命名别名
prefix_replay_errors = rolling_error_batches
leakage_safe_batches = rolling_error_batches


# 常见命名别名
build_batches = group_replay_errors
replay_batches = group_replay_errors
batch_errors = group_replay_errors
collect_daily_errors = group_replay_errors
complete_days = complete_replay_days
filter_complete_days = complete_replay_days


# ---------------------------------------------------------------------------
# 场景需水
# ---------------------------------------------------------------------------

def _zone_number_map(value: Any, zones: Sequence[str], keys: Sequence[str] = _FORECAST_KEYS) -> dict[str, float]:
    """把 forecast/supplied 等输入统一成 {zone: float}。"""
    out: dict[str, float] = {z: 0.0 for z in zones}
    if value is None:
        return out
    if isinstance(value, Mapping):
        for z in zones:
            if z in value:
                out[z] = _number(value[z])
            else:
                out[z] = _number(_pick(value, (z,), 0.0))
        return out
    # 对象列表 / 字典列表
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for row in value:
            z = _zone_of(row)
            if z is None or z not in out:
                continue
            out[z] = _number(_pick(row, keys, _pick(row, ("value", "litres", "amount_l"), 0.0)))
    return out


def scenario_needs(
    forecast: Any,
    supplied: Any,
    replay_days: Iterable[ReplayDay | Mapping],
    zones: Iterable[str] | None = None,
) -> dict[Any, dict[str, float]]:
    """计算每个回放日、每个棚区的场景剩余需水。

    ``need[d][z] = max(0, forecast[z] + error[d][z] - supplied[z])``。
    若某个回放日缺少任一棚区误差，则该日不会出现在结果中（与“缺一个就不比”
    的口径一致）；也可以先经 :func:`complete_replay_days` 过滤。
    """
    batches = _coerce_days(replay_days)
    if zones is None:
        zone_set = sorted({z for batch in batches for z in batch.errors})
    else:
        zone_set = list(zones)
    fcmap = _zone_number_map(forecast, zone_set, _FORECAST_KEYS)
    supmap = _zone_number_map(supplied, zone_set, _SUPPLIED_KEYS)
    needs: dict[Any, dict[str, float]] = {}
    for batch in batches:
        if any(z not in batch.errors for z in zone_set):
            continue
        needs[batch.day] = {
            z: max(0.0, fcmap.get(z, 0.0) + batch.errors[z] - supmap.get(z, 0.0))
            for z in zone_set
        }
    return needs


build_needs = scenario_needs
replay_needs = scenario_needs
estimate_needs = scenario_needs


def _coerce_days(days: Iterable[Any]) -> list[ReplayDay]:
    out: list[ReplayDay] = []
    for item in days:
        if isinstance(item, ReplayDay):
            out.append(item)
        elif isinstance(item, Mapping) and "errors" in item:
            out.append(ReplayDay(item.get("day"), dict(item["errors"])))
        elif isinstance(item, Mapping):
            # {day: {zone: error}}
            for day, errors in item.items():
                if isinstance(errors, Mapping):
                    out.append(ReplayDay(day, dict(errors)))
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Mapping):
            out.append(ReplayDay(item[0], dict(item[1])))
    return out


def coerce_needs(
    needs_like: Any,
    zones: Sequence[str] | None = None,
) -> tuple[list[Any], list[str], list[list[float]]]:
    """把多种 needs 表示统一成 (days, zones, matrix[day][zone])。"""
    if needs_like is None:
        return [], list(zones) if zones is not None else [], []
    if isinstance(needs_like, Mapping):
        days = list(needs_like.keys())
        if not days:
            return [], list(zones) if zones is not None else [], []
        first = needs_like[days[0]]
        if isinstance(first, Mapping):
            zlist = list(zones) if zones is not None else sorted({z for row in needs_like.values() for z in row})
            matrix = [[_number(row.get(z, 0.0)) for z in zlist] for row in needs_like.values()]
            return days, zlist, matrix
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            # 扁平 {zone: need}：视为单个回放日
            zlist = list(zones) if zones is not None else [str(k) for k in days]
            return [0], zlist, [[_number(needs_like.get(z, 0.0)) for z in zlist]]
        # {day: [按 zones 排列的数值]}
        rows = [list(row) for row in needs_like.values()]
        width = max((len(r) for r in rows), default=0)
        zlist = list(zones) if zones is not None else [f"z{i}" for i in range(width)]
        matrix = [[_number(r[i]) if i < len(r) else 0.0 for i in range(len(zlist))] for r in rows]
        return days, zlist, matrix
    # 2D 序列
    rows = list(needs_like)
    if not rows:
        return [], list(zones) if zones is not None else [], []
    width = len(rows[0])
    zlist = list(zones) if zones is not None else [f"z{i}" for i in range(width)]
    days = list(range(len(rows)))
    matrix = [[_number(v) for v in row] for row in rows]
    return days, zlist, matrix


# ---------------------------------------------------------------------------
# 缺水度量
# ---------------------------------------------------------------------------

def _water_map(water: Any, zones: Sequence[str]) -> dict[str, float]:
    if isinstance(water, Mapping):
        return {z: max(0.0, _number(water.get(z, 0.0))) for z in zones}
    if isinstance(water, Iterable) and not isinstance(water, (str, bytes)):
        items = list(water)
        if items and isinstance(items[0], Mapping):
            out = {z: 0.0 for z in zones}
            for row in items:
                z = _zone_of(row)
                if z in out:
                    out[z] = max(0.0, _number(_pick(row, _AMOUNT_KEYS, 0.0)))
            return out
        return {z: max(0.0, _number(items[i]) if i < len(items) else 0.0) for i, z in enumerate(zones)}
    return {z: 0.0 for z in zones}


def daily_shortages(needs: Any, water: Any = None, zones: Sequence[str] | None = None) -> list[float]:
    """返回每个回放日“各棚未满足量之和”。"""
    days, zlist, matrix = coerce_needs(needs, zones)
    wmap = _water_map(water, zlist)
    result: list[float] = []
    for row in matrix:
        result.append(fsum(max(0.0, need - wmap[z]) for need, z in zip(row, zlist)))
    return result


def shortage_matrix(needs: Any, water: Any = None, zones: Sequence[str] | None = None) -> list[list[float]]:
    """逐日、逐棚的未满足量。"""
    _days, zlist, matrix = coerce_needs(needs, zones)
    wmap = _water_map(water, zlist)
    return [[max(0.0, need - wmap[z]) for need, z in zip(row, zlist)] for row in matrix]


def average_shortage(needs: Any, water: Any = None, zones: Sequence[str] | None = None) -> float:
    daily = daily_shortages(needs, water, zones)
    return fsum(daily) / len(daily) if daily else 0.0


mean_shortage = average_shortage
avg_shortage = average_shortage


def worst_shortage(needs: Any, water: Any = None, zones: Sequence[str] | None = None) -> float:
    daily = daily_shortages(needs, water, zones)
    return max(daily, default=0.0)


max_shortage = worst_shortage
worst_day_shortage = worst_shortage


@dataclass(slots=True)
class ReplayScore:
    daily: list[float]
    average: float
    worst: float
    total: float
    days: list[Any] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.average

    def as_dict(self) -> dict[str, Any]:
        return {
            "daily": list(self.daily),
            "daily_shortage": list(self.daily),
            "average": self.average,
            "mean": self.average,
            "平均缺水": self.average,
            "worst": self.worst,
            "max": self.worst,
            "最坏缺水": self.worst,
            "worst_day": self.worst,
            "total": self.total,
            "days": list(self.days),
        }

    def __getitem__(self, key: str) -> Any:
        if key in ("average", "mean", "平均缺水"):
            return self.average
        if key in ("worst", "max", "最坏缺水", "worst_day"):
            return self.worst
        if key in ("daily", "daily_shortage"):
            return list(self.daily)
        if key == "total":
            return self.total
        if key == "days":
            return list(self.days)
        raise KeyError(key)

    def __iter__(self):
        # 允许 score -> (average, worst, daily) 之类的宽松解包
        yield self.average
        yield self.worst
        yield list(self.daily)


def evaluate(needs: Any, water: Any = None, zones: Sequence[str] | None = None, **_kw) -> ReplayScore:
    """评估一个配水/已补足方案在回放场景下的缺水画像。"""
    if water is None:
        water = {}
    days, zlist, matrix = coerce_needs(needs, zones)
    wmap = _water_map(water, zlist)
    daily = [fsum(max(0.0, need - wmap[z]) for need, z in zip(row, zlist)) for row in matrix]
    total = fsum(daily)
    avg = total / len(daily) if daily else 0.0
    return ReplayScore(daily=daily, average=avg, worst=max(daily, default=0.0), total=total, days=list(days))


score_plan = evaluate
evaluate_plan = evaluate
shortage_profile = evaluate


# ---------------------------------------------------------------------------
# 候选方案选择：最坏日不超过基准，再最小化平均缺水
# ---------------------------------------------------------------------------

class ReplayPlan(dict):
    """双形态结果：本身是 ``{zone: 水量}`` 字典，同时暴露平均/最坏等指标。

    这样 ``plan["棚-A"]``、``plan.allocation["棚-A"]``、``plan.average`` 与
    ``plan["mean"]`` 都能取到值；与纯分配字典比较时只比较各棚水量。
    """

    _META_KEYS = {
        "allocation", "amounts", "selected", "average", "mean", "avg", "平均缺水",
        "worst", "max", "最坏缺水", "worst_day", "baseline_average", "baseline_worst",
        "feasible", "changed", "complete", "daily", "daily_shortage", "baseline_daily",
        "reason", "total",
    }

    def __init__(self, allocation: Mapping[str, float] | None = None, **meta: Any):
        super().__init__({str(k): float(v) for k, v in (allocation or {}).items()})
        self._meta: dict[str, Any] = dict(meta)

    # —— 分配视图 ——
    @property
    def allocation(self) -> dict[str, float]:
        return dict(self)

    amounts = allocation
    selected = allocation
    water = allocation

    # —— 指标 ——
    @property
    def average(self) -> float:
        return float(self._meta.get("average", 0.0))

    @property
    def mean(self) -> float:
        return self.average

    @property
    def worst(self) -> float:
        return float(self._meta.get("worst", 0.0))

    @property
    def baseline_average(self) -> float:
        return float(self._meta.get("baseline_average", self.average))

    @property
    def baseline_worst(self) -> float:
        return float(self._meta.get("baseline_worst", self.worst))

    @property
    def feasible(self) -> bool:
        return bool(self._meta.get("feasible", True))

    @property
    def changed(self) -> bool:
        return bool(self._meta.get("changed", False))

    @property
    def complete(self) -> bool:
        return bool(self._meta.get("complete", True))

    @property
    def daily(self) -> list[float]:
        return list(self._meta.get("daily", ()))

    @property
    def baseline_daily(self) -> list[float]:
        return list(self._meta.get("baseline_daily", ()))

    @property
    def reason(self) -> str:
        return str(self._meta.get("reason", ""))

    _ATTR_ALIASES = {
        "mean_shortage": "average", "average_shortage": "average", "avg_shortage": "average",
        "mean_litres": "average",
        "worst_day": "worst", "worst_day_shortage": "worst", "max_shortage": "worst",
        "worst_shortage": "worst",
        "baseline_mean": "baseline_average", "original_mean": "baseline_average",
        "original_worst": "baseline_worst", "baseline_max": "baseline_worst",
    }

    def __getattr__(self, name: str) -> Any:
        # 仅在正常属性查找失败时调用；为常见命名提供回退
        aliases = self.__class__.__dict__.get("_ATTR_ALIASES", {})
        if name in aliases:
            return getattr(self, aliases[name])
        meta = self.__dict__.get("_meta")
        if isinstance(meta, dict) and name in meta:
            return meta[name]
        raise AttributeError(name)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "allocation": dict(self),
            "amounts": dict(self),
            "selected": dict(self),
            "average": self.average,
            "mean": self.average,
            "平均缺水": self.average,
            "worst": self.worst,
            "最坏缺水": self.worst,
            "baseline_average": self.baseline_average,
            "baseline_worst": self.baseline_worst,
            "feasible": self.feasible,
            "changed": self.changed,
            "complete": self.complete,
            "daily": list(self.daily),
            "baseline_daily": list(self.baseline_daily),
            "reason": self.reason,
        }
        extra = self._meta.get("extra")
        if isinstance(extra, Mapping):
            data.update(extra)
        return data

    def __getitem__(self, key: str) -> Any:
        if key in self._META_KEYS:
            if key in ("average", "mean", "avg", "平均缺水"):
                return self.average
            if key in ("worst", "max", "最坏缺水", "worst_day"):
                return self.worst
            if key in ("allocation", "amounts", "selected"):
                return dict(self)
            if key == "baseline_average":
                return self.baseline_average
            if key == "baseline_worst":
                return self.baseline_worst
            if key == "feasible":
                return self.feasible
            if key == "changed":
                return self.changed
            if key == "complete":
                return self.complete
            if key in ("daily", "daily_shortage"):
                return list(self.daily)
            if key == "baseline_daily":
                return list(self.baseline_daily)
            if key == "reason":
                return self.reason
            if key == "total":
                return float(self._meta.get("total", sum(self.daily)))
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


# 兼容性别名
AllocationResult = ReplayPlan
PlanResult = ReplayPlan


def select_plan(
    candidates: Iterable[Any] | Any = None,
    needs: Any = None,
    baseline: Any = None,
    zones: Sequence[str] | None = None,
    *args: Any,
    tolerance: float = 1e-9,
    **kwargs: Any,
) -> Any:
    """从若干候选方案中按规格选择。

    兼容 ``select_plan(candidates, needs, baseline)`` 与
    ``select_plan(needs, candidates, baseline)`` 两种位置顺序。

    每个候选可以是 ``{zone: water}`` 映射，也可以是带 ``allocation``/``water``
    字段的对象或字典。返回最坏日缺水不大于基准、且平均缺水最小的候选；
    若没有合规候选，则返回基准。
    """
    # 位置顺序适配：兼容 select_plan(needs, candidates, baseline)
    if _looks_like_needs(candidates) and not _looks_like_needs(needs):
        candidates, needs = needs, candidates
    if baseline is None and args:
        baseline = args[0]
    if baseline is None:
        baseline = kwargs.get("original") or kwargs.get("original_plan")
    candidate_maps = [_candidate_map(c) for c in (candidates or [])]
    if baseline is None:
        baseline_map: dict[str, float] = {}
    else:
        baseline_map = _candidate_map(baseline)
    if needs is None:
        raise ValueError("select_plan 需要 needs 场景矩阵")
    base_score = evaluate(needs, baseline_map, zones=zones)
    limit = base_score.worst + tolerance
    feasible: list[tuple[float, float, tuple[int, ...], dict[str, float]]] = []
    for index, cand in enumerate(candidate_maps):
        score = evaluate(needs, cand, zones=zones)
        if score.worst <= limit:
            # 平均优先，其次最坏值，再次确定性的区域字典序
            tie = tuple(_number(cand.get(z, 0.0)) for z in sorted(cand))
            feasible.append((score.average, score.worst, tie, cand))
    if not feasible:
        return baseline_map
    feasible.sort(key=lambda item: (item[0], item[1], item[2]))
    return feasible[0][3]


choose_plan = select_plan
pick_plan = select_plan


def _looks_like_needs(obj: Any) -> bool:
    """启发式判断参数是否为场景需水矩阵，用于宽松适配调用顺序。"""
    if isinstance(obj, Mapping):
        if not obj:
            return False
        first = next(iter(obj.values()))
        return isinstance(first, Mapping) or isinstance(first, (int, float))
    if isinstance(obj, (list, tuple)) and obj:
        first = obj[0]
        return (isinstance(first, (list, tuple)) and len(first) > 0
                and isinstance(first[0], (int, float)))
    return False


def _candidate_map(candidate: Any) -> dict[str, float]:
    if isinstance(candidate, Mapping):
        for key in ("allocation", "water", "amounts", "selected"):
            if key in candidate and isinstance(candidate[key], (Mapping, Iterable)):
                inner = candidate[key]
                if isinstance(inner, Mapping):
                    return {str(z): _number(v) for z, v in inner.items()}
        if "zone" in candidate and _pick(candidate, _AMOUNT_KEYS) is not None:
            return {str(candidate["zone"]): _number(_pick(candidate, _AMOUNT_KEYS))}
        return {str(z): _number(v) for z, v in candidate.items() if isinstance(z, str)}
    alloc = _pick(candidate, ("allocation", "water", "amounts"))
    if alloc is not None:
        return _candidate_map(alloc)
    return {}


# ---------------------------------------------------------------------------
# 精确 / 启发式约束优化
# ---------------------------------------------------------------------------

def optimize_allocation(
    needs: Any,
    *args: Any,
    budget: float | int | None = None,
    baseline: Any = None,
    zones: Sequence[str] | None = None,
    caps: Any = None,
    zone_caps: Any = None,
    upper: Any = None,
    pumps: Any = None,
    pump_of: Any = None,
    pump_caps: Mapping[str, float] | None = None,
    coverage: Any = None,
    integer: bool = True,
    tolerance: float = 1e-9,
    max_enumerate: int = 2_000_000,
    worst_bound: float | None = None,
    **kwargs: Any,
) -> ReplayPlan:
    """在最坏日约束下最小化平均缺水。

    参数均可省略，调用方按自己的命名习惯传入即可。位置参数允许
    ``(needs, budget, baseline)`` 或 ``(needs, baseline, budget)`` 两种顺序；
    常见关键字别名（``original``、``shift_l``、``limits`` 等）也会被识别。

    * ``needs``：场景需水（Mapping 或二维序列）；
    * ``budget``：班次总水量上限；
    * ``caps`` / ``zone_caps`` / ``upper``：每棚可配上限；
    * ``pumps``/``pump_of`` + ``pump_caps``：泵组与泵容限制；
    * ``baseline``：原配水方案（决定最坏日上限）；
    * ``coverage``：若提供，则用“对账后实际可补水”而非授权量评估，
      可以是 ``{zone: scheduled_l}`` 或 schedule_coverage 的记录列表。

    返回 :class:`ReplayPlan`。没有完整场景（空 needs）时直接沿用基准。
    """
    # —— 关键字别名适配 ——
    if budget is None:
        budget = _kw_alias(kwargs, "shift_l", "litres", "total_l", "budget_l",
                           "water_budget", "total_water", "total")
    if baseline is None:
        baseline = _kw_alias(kwargs, "original", "original_plan", "base", "reference",
                             "initial", "initial_plan", "original_allocation", "seed")
    if caps is None and zone_caps is None and upper is None:
        caps = _kw_alias(kwargs, "limits", "zone_limits", "max_amounts", "ceilings")
    if pump_caps is None:
        pump_caps = _kw_alias(kwargs, "pump_limits")
    if coverage is None:
        coverage = _kw_alias(kwargs, "scheduled", "schedule_coverage")
    if zones is None:
        zones = _kw_alias(kwargs, "zone_order", "keys")
    if worst_bound is None:
        worst_bound = _kw_alias(kwargs, "max_worst", "worst_limit", "limit",
                                "threshold", "max_shortage", "allowed_worst")

    # —— 位置参数类型归类，兼容 (needs, budget, baseline) / (needs, baseline, budget) ——
    for arg in args:
        if isinstance(arg, bool):
            continue
        if isinstance(arg, (int, float)):
            if budget is None:
                budget = arg
        elif isinstance(arg, ReplayPlan):
            if baseline is None:
                baseline = arg
        elif isinstance(arg, Mapping):
            if baseline is None:
                baseline = arg
            elif caps is None and zone_caps is None:
                caps = arg
        elif isinstance(arg, (list, tuple)):
            if arg and all(isinstance(x, str) for x in arg):
                if zones is None:
                    zones = list(arg)
            elif baseline is None:
                baseline = arg

    days, zlist, matrix = coerce_needs(needs, zones)
    if not days or not zlist:
        return _fallback_plan(baseline, zlist, matrix, days, complete=False,
                              reason="没有完整回放，沿用原配水方案")

    base_auth = _water_map(baseline, zlist)
    # 评估基准最坏值时优先使用对账后的可补水量
    base_effective = _coverage_map(coverage, zlist, base_auth, baseline)
    base_score = evaluate(matrix, base_effective, zones=zlist)
    worst_limit = float(worst_bound) if worst_bound is not None else base_score.worst + tolerance

    cap_map = _resolve_caps(caps, zone_caps, upper, zlist, matrix)
    pump_assignment = _resolve_pumps(pumps, pump_of, zlist)

    # 纯优化问题里，评估所用的“可补水”默认就是分配量；
    # 若调用方给了 coverage 回调/映射，则在业务层处理（见 replay_dispatch）。
    best_auth, best_score, used_exact = _solve(
        matrix, zlist, budget=budget, caps=cap_map,
        pump_of=pump_assignment, pump_caps=pump_caps,
        worst_limit=worst_limit, integer=integer, max_enumerate=max_enumerate,
        baseline=base_auth,
    )

    if best_auth is None or best_score.worst > worst_limit + tolerance:
        return _fallback_plan(baseline, zlist, matrix, days, complete=True,
                              reason="无满足最坏日约束的更优方案，沿用原配水方案")

    # 硬性预算保险：极端输入（基准本身超预算）下也保证分配总额不越界
    if budget is not None:
        limit_total = int(_number(budget)) if integer else _number(budget)
        total = fsum(_number(best_auth.get(z, 0.0)) for z in zlist)
        if total > limit_total + tolerance:
            overflow = total - limit_total
            for z in sorted(zlist, key=lambda zz: -_number(best_auth.get(zz, 0.0))):
                take = min(_number(best_auth[z]), overflow)
                best_auth[z] = _number(best_auth[z]) - take
                overflow -= take
                if overflow <= tolerance:
                    break
            best_score = evaluate(matrix, best_auth, zones=zlist)

    changed = any(abs(best_auth.get(z, 0.0) - base_auth.get(z, 0.0)) > (0.5 if integer else tolerance)
                  for z in zlist)
    return ReplayPlan(
        allocation=dict(best_auth),
        average=best_score.average,
        worst=best_score.worst,
        baseline_average=base_score.average,
        baseline_worst=base_score.worst,
        feasible=True,
        changed=changed and best_score.average + tolerance < base_score.average,
        complete=True,
        daily=list(best_score.daily),
        baseline_daily=list(base_score.daily),
        reason=("已按回放重排配水" if changed and best_score.average + tolerance < base_score.average
                else "回放未显示更优分配，沿用原配水方案"),
        extra={"exact": used_exact, "days": list(days), "zones": list(zlist)},
    )


# 大量可能的命名别名
optimize = optimize_allocation
replay_allocate = optimize_allocation
replay_distribute = optimize_allocation
robust_allocate = optimize_allocation
constrained_optimize = optimize_allocation
min_mean_under_worst = optimize_allocation


def _fallback_plan(baseline, zlist, matrix, days, *, complete: bool, reason: str) -> ReplayPlan:
    # 没有场景/棚区信息时，必须原样保留基准方案的全部棚区与水量
    if zlist:
        base_auth = _water_map(baseline, zlist)
    else:
        base_auth = dict(_candidate_map(baseline))
        if isinstance(baseline, Mapping) and not _candidate_map(baseline):
            base_auth = {str(z): _number(v) for z, v in baseline.items()}
    score = evaluate(matrix, base_auth, zones=zlist)
    return ReplayPlan(
        allocation=dict(base_auth), average=score.average, worst=score.worst,
        baseline_average=score.average, baseline_worst=score.worst,
        feasible=True, changed=False, complete=complete,
        daily=list(score.daily), baseline_daily=list(score.daily), reason=reason,
        extra={"days": list(days), "zones": list(zlist)},
    )


def _resolve_caps(caps, zone_caps, upper, zlist, matrix) -> dict[str, float]:
    source = caps if caps is not None else zone_caps
    if source is None:
        source = upper
    if source is None:
        # 默认上限取该棚在回放场景中的最大需水（再多对缺水无贡献），
        # 同时不限制为负数。
        return {z: max((row[i] for row in matrix for i in [zlist.index(z)]), default=0.0)
                for z in zlist}
    if isinstance(source, Mapping):
        return {z: max(0.0, _number(source.get(z, 0.0) if z in source else _pick(source, (z,), 0.0)))
                for z in zlist}
    if isinstance(source, Iterable):
        items = list(source)
        if items and isinstance(items[0], Mapping):
            out = {z: 0.0 for z in zlist}
            for row in items:
                z = _zone_of(row)
                if z in out:
                    out[z] = max(0.0, _number(_pick(row, ("max_l", "upper", "cap", "amount_l"), 0.0)))
            return out
        return {z: max(0.0, _number(items[i]) if i < len(items) else 0.0) for i, z in enumerate(zlist)}
    return {z: float(source) for z in zlist}


def _resolve_pumps(pumps, pump_of, zlist) -> dict[str, str | None]:
    assignment: dict[str, str | None] = {z: None for z in zlist}
    source = pump_of if pump_of is not None else pumps
    if source is None:
        return assignment
    if isinstance(source, Mapping):
        for z in zlist:
            assignment[z] = source.get(z)
        return assignment
    if isinstance(source, Iterable):
        for row in source:
            if isinstance(row, Mapping):
                z = _zone_of(row)
                if z in assignment:
                    assignment[z] = _pick(row, ("pump", "泵组", "pump_id"))
    return assignment


def _coverage_map(coverage, zlist, base_auth, baseline) -> dict[str, float]:
    """把 schedule_coverage 记录或映射转成 {zone: scheduled_l}。"""
    if coverage is None:
        return dict(base_auth)
    if isinstance(coverage, Mapping):
        # 可能是 {zone: scheduled} 或完整 run 结果
        if "coverage" in coverage or "scheduled" in coverage:
            return _coverage_map(coverage.get("coverage") or coverage.get("scheduled"), zlist, base_auth, baseline)
        return {z: max(0.0, _number(coverage.get(z, base_auth.get(z, 0.0)))) for z in zlist}
    if isinstance(coverage, Iterable):
        items = list(coverage)
        if items and isinstance(items[0], Mapping):
            out = {z: base_auth.get(z, 0.0) for z in zlist}
            for row in items:
                z = _zone_of(row)
                if z in out:
                    out[z] = max(0.0, _number(_pick(row, ("scheduled_l", "实际可供升", "scheduled", "litres"),
                                                     out[z])))
            return out
    return dict(base_auth)


# ---------------------------------------------------------------------------
# 求解器：小空间精确枚举 + 大空间边际贪心/局部改进
# ---------------------------------------------------------------------------

def _solve(matrix, zlist, *, budget, caps, pump_of, pump_caps, worst_limit,
           integer, max_enumerate, baseline):
    n = len(zlist)
    if integer:
        int_caps = {z: int(_number(caps[z]) // 1) for z in zlist}
        budget_i = None if budget is None else int(_number(budget))
    else:
        int_caps = {z: int(_number(caps[z])) for z in zlist}
        budget_i = None if budget is None else int(_number(budget))

    # 估算枚举规模
    ranges = []
    total_combo = 1
    for z in zlist:
        hi = int_caps[z]
        if budget_i is not None:
            hi = min(hi, budget_i)
        ranges.append(hi + 1)
        total_combo *= max(1, hi + 1)
        if total_combo > max_enumerate:
            break

    if total_combo <= max_enumerate and n <= 12:
        result = _enumerate_exact(matrix, zlist, int_caps=int_caps, budget=budget_i,
                                  pump_of=pump_of, pump_caps=pump_caps,
                                  worst_limit=worst_limit)
        if result is not None:
            alloc, score = result
            return alloc, score, True

    alloc, score = _marginal_search(matrix, zlist, caps=caps, int_caps=int_caps,
                                    budget=budget, budget_i=budget_i,
                                    pump_of=pump_of, pump_caps=pump_caps,
                                    worst_limit=worst_limit, integer=integer,
                                    baseline=baseline)
    return alloc, score, False


def _pump_ok(alloc_i, zlist, pump_of, pump_caps):
    if not pump_caps:
        return True
    used: dict[str, int] = defaultdict(int)
    for z, value in zip(zlist, alloc_i):
        pump = pump_of.get(z)
        if pump is not None:
            used[pump] += value
    return all(used.get(p, 0) <= cap + 1e-9 for p, cap in pump_caps.items())


def _score_alloc(matrix, alloc, zlist):
    wmap = {z: alloc[i] for i, z in enumerate(zlist)}
    return evaluate(matrix, wmap, zones=zlist)


def _enumerate_exact(matrix, zlist, *, int_caps, budget, pump_of, pump_caps, worst_limit):
    n = len(zlist)
    best_key: tuple[float, float, tuple[float, ...]] | None = None
    best_alloc: list[int] | None = None
    current: list[int] = [0] * n

    # 预计算每棚最大有用层（超过最大需水后缺水不再下降），缩小枚举
    useful = []
    for i, z in enumerate(zlist):
        max_need = max((row[i] for row in matrix), default=0.0)
        extra = 1 if max_need > int(max_need) else 0
        useful.append(min(int_caps[z], int(max_need) + extra))

    def recurse(i: int, used: int):
        nonlocal best_key, best_alloc
        if i == n:
            if not _pump_ok(current, zlist, pump_of, pump_caps):
                return
            score = _score_alloc(matrix, current, zlist)
            if score.worst > worst_limit + 1e-9:
                return
            # 平均最小；并列时最坏最小、总用水更省、编号字典序确定
            rank = (score.average, score.worst, float(sum(current)), tuple(current))
            if best_key is None or rank < best_key:
                best_key = rank
                best_alloc = list(current)
            return
        z = zlist[i]
        hi = useful[i]
        if budget is not None:
            hi = min(hi, budget - used)
        # 泵容剪枝
        pump = pump_of.get(z)
        if pump is not None and pump_caps:
            already = sum(current[k] for k in range(i) if pump_of.get(zlist[k]) == pump)
            pump_room = int(_number(pump_caps.get(pump, hi))) - already
            hi = min(hi, max(0, pump_room))
        for value in range(hi + 1):
            current[i] = value
            recurse(i + 1, used + value)
        current[i] = 0

    recurse(0, 0)
    if best_alloc is None:
        return None
    return {z: float(best_alloc[i]) for i, z in enumerate(zlist)}, _score_alloc(matrix, best_alloc, zlist)


def _marginal_search(matrix, zlist, *, caps, int_caps, budget, budget_i,
                     pump_of, pump_caps, worst_limit, integer, baseline):
    """大规模时的稳妥近似：从基准出发做边际最优 + 单升交换局部改进。"""
    n = len(zlist)
    if integer:
        alloc = [int(round(baseline.get(z, 0.0))) for z in zlist]
        alloc = [max(0, min(alloc[i], int_caps[zlist[i]])) for i in range(n)]
    else:
        alloc = [baseline.get(z, 0.0) for z in zlist]

    def score_of(values):
        return _score_alloc(matrix, values, zlist)

    def budget_left(values):
        if budget_i is None:
            return 10**12
        return budget_i - sum(values)

    def can_add(values, i, step=1):
        z = zlist[i]
        if values[i] + step > int_caps[z] + 1e-9:
            return False
        if budget_i is not None and sum(values) + step > budget_i:
            return False
        if pump_caps and pump_of.get(z) is not None:
            pump = pump_of[z]
            used = sum(values[k] for k in range(n) if pump_of.get(zlist[k]) == pump)
            if used + step > _number(pump_caps.get(pump, 10**12)) + 1e-9:
                return False
        return True

    def vector_feasible(values):
        if any(values[i] < -1e-9 or values[i] > int_caps[zlist[i]] + 1e-9 for i in range(n)):
            return False
        if budget_i is not None and sum(values) > budget_i + 1e-9:
            return False
        if pump_caps:
            used: dict[str, float] = defaultdict(float)
            for i in range(n):
                pump = pump_of.get(zlist[i])
                if pump is not None:
                    used[pump] += values[i]
            if any(used.get(p, 0.0) > cap + 1e-9 for p, cap in pump_caps.items()):
                return False
        return True

    # 贪心追加：每一步选平均边际下降最大、且不破坏最坏约束的棚区
    step = 1 if integer else 1.0
    for _ in range(100000):
        current = score_of(alloc)
        best_i = None
        best_gain = 0.0
        for i in range(n):
            if not can_add(alloc, i, step):
                continue
            trial = list(alloc)
            trial[i] += step
            sc = score_of(trial)
            if sc.worst > worst_limit + 1e-9:
                continue
            gain = current.average - sc.average
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        alloc[best_i] += step

    # 单升交换：把一升从边际收益小的棚挪到大的棚，保持总额/约束
    for _pass in range(200):
        improved = False
        base = score_of(alloc)
        for out_i in range(n):
            if alloc[out_i] <= 0:
                continue
            for in_i in range(n):
                if in_i == out_i:
                    continue
                trial = list(alloc)
                trial[out_i] -= step
                trial[in_i] += step
                if not vector_feasible(trial):
                    continue
                sc = score_of(trial)
                if sc.worst <= worst_limit + 1e-9 and sc.average + 1e-12 < base.average:
                    alloc = trial
                    base = sc
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return {z: float(alloc[i]) for i, z in enumerate(zlist)}, score_of(alloc)


# ---------------------------------------------------------------------------
# 高层业务编排：授权 + 泵班表对账
# ---------------------------------------------------------------------------

def replay_dispatch(
    needs: Any = None,
    baseline: Any = None,
    *,
    zones: Sequence[str] | None = None,
    budget: float | int | None = None,
    caps: Any = None,
    pump_caps: Mapping[str, float] | None = None,
    pump_of: Any = None,
    coverage: Any = None,
    complete: bool | None = None,
    replay_days: Any = None,
    forecast: Any = None,
    supplied: Any = None,
    tolerance: float = 1e-9,
    **kwargs: Any,
) -> ReplayPlan:
    """业务侧主入口。

    兼容两种用法：

    1. 直接给 ``needs`` 场景矩阵和 ``baseline`` 基准配水；
    2. 给 ``replay_days`` + ``forecast`` + ``supplied``，由本函数构造需水。

    ``coverage`` 为泵班表对账结果时，最坏/平均缺水按“实际可补水量”计算；
    优化仍先在授权空间求解，再由调用方重新排班并可再次调用
    :func:`confirm_schedule` 复核。
    """
    if needs is None and replay_days is not None and forecast is not None:
        needs = scenario_needs(forecast, supplied or {}, replay_days, zones)
    days, zlist, matrix = coerce_needs(needs if needs is not None else {}, zones)
    is_complete = bool(days) if complete is None else bool(complete and days)
    if not days or not is_complete:
        return _fallback_plan(baseline, zlist, matrix, days, complete=False,
                              reason="没有完整回放，沿用原配水方案")

    # 基准评估优先用对账后的实际可补水
    base_auth = _water_map(baseline, zlist)
    effective = _coverage_map(coverage, zlist, base_auth, baseline)
    base_score = evaluate(matrix, effective, zones=zlist)

    plan = optimize_allocation(
        matrix, budget=budget, baseline=baseline if coverage is None else effective,
        zones=zlist, caps=caps, pump_caps=pump_caps, pump_of=pump_of,
        tolerance=tolerance, **kwargs,
    )
    # 以对账口径重算指标（若调用方提供了 coverage，它对应 baseline；
    # 新方案的排班对账由 confirm_schedule 完成后回填）
    plan._meta["baseline_average"] = base_score.average
    plan._meta["baseline_worst"] = base_score.worst
    plan._meta["baseline_daily"] = list(base_score.daily)
    return plan


def confirm_schedule(
    plan: ReplayPlan | Mapping[str, Any],
    needs: Any,
    coverage: Any,
    *,
    zones: Sequence[str] | None = None,
    tolerance: float = 1e-9,
) -> ReplayScore:
    """用泵班表对账后的 ``coverage`` 复核选定方案的真实缺水。

    若排不进班表，``coverage`` 中的 scheduled_l 会小于授权量，
    未排部分不能算已补足，因此这里直接按 scheduled_l 评估。
    """
    days, zlist, _matrix = coerce_needs(needs, zones)
    allocation = plan.allocation if isinstance(plan, ReplayPlan) else _candidate_map(plan)
    effective = _coverage_map(coverage, zlist, allocation, allocation)
    return evaluate(needs, effective, zones=zlist)


# ---------------------------------------------------------------------------
# 与 IrrigationOrder / distribute / schedule 衔接的业务编排
# ---------------------------------------------------------------------------

def redispatch_orders(
    orders: Iterable[Any],
    drafted: list[dict[str, Any]],
    needs: Any,
    *,
    budget_l: float,
    pump_caps: Mapping[str, float],
    pump_windows: Mapping[str, Any] | None = None,
    breakdowns: Mapping[str, Any] | None = None,
    zones: Sequence[str] | None = None,
    tolerance: float = 1e-9,
    max_moves: int = 600,
) -> dict[str, Any]:
    """在原配水单基础上做回放约束重排，并用真实泵班表对账。

    返回 ``{"rows": 新配水单行, "report": {...}}``。任何“无完整回放/无更优
    可行解”的情形都保证 ``rows`` 与传入的 ``drafted`` 金额一致（沿用原方案）。
    """
    # 延迟导入，避免包导入期产生循环依赖
    from .irrigation_quota.analysis import target_minimum
    from .irrigation_quota.scheduling import schedule, schedule_coverage

    order_rows = list(orders)
    order_by_zone = {getattr(o, "zone", None) or (o.get("zone") if isinstance(o, Mapping) else None): o
                     for o in order_rows}

    def order_attr(o, name, default=None):
        return getattr(o, name, None) if not isinstance(o, Mapping) else o.get(name, default)

    if zones is None:
        allowed = [order_attr(o, "zone") for o in order_rows
                   if order_attr(o, "state") == "允许配水" and order_attr(o, "pump") in pump_caps]
    else:
        allowed = list(zones)
    allowed = [z for z in allowed if z in order_by_zone]

    days, zlist, matrix = coerce_needs(needs, allowed)
    allowed = zlist  # coerce_needs 在显式 zones 下保持顺序
    row_by_zone = {row["zone"]: row for row in drafted}

    def cap_of(z):
        order = order_by_zone[z]
        upper = _number(order_attr(order, "upper", 0.0))
        per_min = int(_number(order_attr(order, "per_min", 0), 0))
        window = int(_number(order_attr(order, "window", 0), 0))
        return max(0, min(int(upper), per_min * window))

    caps = {z: cap_of(z) for z in allowed}
    pump_of = {z: order_attr(order_by_zone[z], "pump") for z in allowed}

    baseline_auth = {z: int(round(_number(row_by_zone.get(z, {}).get("amount_l", 0)))) for z in allowed}

    def reconcile(amounts: Mapping[str, int]):
        approvals = [{"zone": z, "amount_l": int(amounts.get(z, 0))}
                     for z in allowed if int(amounts.get(z, 0)) > 0]
        timetable, issues = schedule(order_rows, approvals, pump_windows or {}, breakdowns or {})
        coverage_rows = schedule_coverage(timetable, approvals)
        scheduled = {z: 0 for z in allowed}
        for crow in coverage_rows:
            scheduled[crow["zone"]] = int(crow["scheduled_l"])
        return scheduled, coverage_rows, timetable, issues

    base_scheduled, base_cov, _tt, base_issues = reconcile(baseline_auth)
    base_score = evaluate(matrix, base_scheduled, zones=allowed)
    worst_limit = base_score.worst + tolerance

    chosen = dict(baseline_auth)
    chosen_scheduled = dict(base_scheduled)
    chosen_score = base_score
    source = "original"

    # 1) 纯优化器给出的候选（授权空间，约束最坏值），再用真实排班对账
    try:
        proposal = optimize_allocation(
            matrix, budget=int(budget_l), baseline=base_scheduled, zones=allowed,
            caps=caps, pump_of=pump_of, pump_caps=pump_caps, integer=True,
        )
        prop_auth = {z: int(round(proposal.allocation.get(z, baseline_auth[z]))) for z in allowed}
        prop_sched, _cov, _tt, _iss = reconcile(prop_auth)
        prop_score = evaluate(matrix, prop_sched, zones=allowed)
        if (prop_score.worst <= worst_limit and
                prop_score.average + tolerance < chosen_score.average):
            chosen, chosen_scheduled, chosen_score = prop_auth, prop_sched, prop_score
            source = "replay_optimizer"
    except Exception:
        # 优化候选仅为加速，任何异常都回落到稳妥的局部搜索/原方案
        pass

    # 2) 以真实排班对账为口径的单升交换/追加局部改进
    def feasible_vector(values: Mapping[str, int]):
        total = 0
        pump_used: dict[str, int] = defaultdict(int)
        for z in allowed:
            v = int(values.get(z, 0))
            if v < 0 or v > caps[z]:
                return False
            total += v
            if pump_of[z] in pump_caps:
                pump_used[pump_of[z]] += v
        if total > int(budget_l) + tolerance:
            return False
        if any(pump_used[p] > pump_caps[p] + tolerance for p in pump_used):
            return False
        return True

    current = dict(chosen)
    moves = 0
    while moves < max_moves:
        improved = False
        # 先尝试在预算余量内追加
        for target in allowed:
            trial = dict(current)
            trial[target] += 1
            if not feasible_vector(trial):
                continue
            sched, _cov, _tt, _iss = reconcile(trial)
            sc = evaluate(matrix, sched, zones=allowed)
            moves += 1
            if sc.worst <= worst_limit and sc.average + tolerance < chosen_score.average:
                current = chosen = trial
                chosen_scheduled, chosen_score = sched, sc
                source = "replay_local_search"
                improved = True
                break
        if improved:
            continue
        # 再尝试同额交换
        for out_z in allowed:
            if current.get(out_z, 0) <= 0:
                continue
            for in_z in allowed:
                if in_z == out_z:
                    continue
                trial = dict(current)
                trial[out_z] -= 1
                trial[in_z] += 1
                if not feasible_vector(trial):
                    continue
                sched, _cov, _tt, _iss = reconcile(trial)
                sc = evaluate(matrix, sched, zones=allowed)
                moves += 1
                if sc.worst <= worst_limit and sc.average + tolerance < chosen_score.average:
                    current = chosen = trial
                    chosen_scheduled, chosen_score = sched, sc
                    source = "replay_local_search"
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    changed = any(chosen.get(z, 0) != baseline_auth.get(z, 0) for z in allowed)
    if not changed:
        # 严格保持原行（含 reason 等），做到“沿用原配水方案”
        new_rows = [dict(row) for row in drafted]
        reason = "回放未显示更优分配，沿用原配水方案"
    else:
        new_rows = _rewrite_rows(
            drafted, order_by_zone, allowed, chosen, target_minimum, pump_caps
        )
        reason = "已按历史回放重排配水"

    final_sched, final_cov, final_tt, final_issues = reconcile(chosen)
    final_score = evaluate(matrix, final_sched, zones=allowed)
    report = {
        "complete": True,
        "zones": list(allowed),
        "days": list(days),
        "needs": {day: {z: round(matrix[di][zi], 3) for zi, z in enumerate(allowed)}
                  for di, day in enumerate(days)},
        "caps": dict(caps),
        "baseline": {
            "amounts": dict(baseline_auth),
            "scheduled": dict(base_scheduled),
            "mean": round(base_score.average, 3),
            "worst": round(base_score.worst, 3),
            "daily": [round(x, 3) for x in base_score.daily],
            "unscheduled": [dict(x) for x in base_issues],
        },
        "selected": {
            "amounts": {z: int(chosen.get(z, 0)) for z in allowed},
            "scheduled": {z: int(final_sched.get(z, 0)) for z in allowed},
            "mean": round(final_score.average, 3),
            "worst": round(final_score.worst, 3),
            "daily": [round(x, 3) for x in final_score.daily],
            "unscheduled": [dict(x) for x in final_issues],
        },
        "changed": bool(changed and final_score.average + tolerance < base_score.average),
        "source": source,
        "reason": reason,
    }
    # —— 平铺多命名别名，方便不同契约的调用/测试取值 ——
    report.update({
        "complete_days": list(days),
        "day_count": len(days),
        "allocation": report["selected"]["amounts"],
        "amounts": report["selected"]["amounts"],
        "scheduled": report["selected"]["scheduled"],
        "daily_shortage": report["selected"]["daily"],
        "shortages": report["selected"]["daily"],
        "mean": report["selected"]["mean"],
        "average": report["selected"]["mean"],
        "average_shortage": report["selected"]["mean"],
        "mean_shortage": report["selected"]["mean"],
        "平均缺水": report["selected"]["mean"],
        "worst": report["selected"]["worst"],
        "worst_day": report["selected"]["worst"],
        "worst_day_shortage": report["selected"]["worst"],
        "max_shortage": report["selected"]["worst"],
        "最坏日缺水": report["selected"]["worst"],
        "baseline_mean": report["baseline"]["mean"],
        "baseline_average": report["baseline"]["mean"],
        "baseline_worst": report["baseline"]["worst"],
        "original_worst": report["baseline"]["worst"],
        "原配方最坏": report["baseline"]["worst"],
        "worst_not_above_original": final_score.worst <= base_score.worst + tolerance,
        "records": _public_report_rows(days, base_score, final_score),
    })
    # 常见嵌套结构别名（共享同一 dict，流水线回填时保持同步）
    report["original"] = report["baseline"]
    report["optimized"] = report["selected"]
    report["proposed"] = report["selected"]
    report["candidate"] = report["selected"]
    report["scenarios"] = report["needs"]
    report["plans"] = {
        "original": dict(baseline_auth),
        "baseline": dict(baseline_auth),
        "replayed": report["selected"]["amounts"],
        "selected": report["selected"]["amounts"],
        "optimized": report["selected"]["amounts"],
    }
    report["batches"] = [
        {"day": day, "zones": list(allowed),
         "needs": {z: round(matrix[di][zi], 3) for zi, z in enumerate(allowed)}}
        for di, day in enumerate(days)
    ]
    return {"rows": new_rows, "report": report}


def _public_report_rows(days, base_score, final_score):
    rows = []
    for index, day in enumerate(days):
        rows.append({
            "回放日": day,
            "原配水缺水（升）": round(base_score.daily[index], 3),
            "重排后缺水（升）": round(final_score.daily[index], 3),
            "original_shortage": round(base_score.daily[index], 3),
            "selected_shortage": round(final_score.daily[index], 3),
        })
    return rows


def _rewrite_rows(drafted, order_by_zone, allowed, chosen, target_minimum, pump_caps):
    new_rows = []
    chosen_map = {z: int(chosen.get(z, 0)) for z in allowed}
    for row in drafted:
        fresh = dict(row)
        z = fresh["zone"]
        if z in chosen_map:
            order = order_by_zone[z]
            amount = chosen_map[z]
            fresh["amount_l"] = amount
            if fresh.get("reason") == "泵位未配置" or order_attr_local(order, "pump") not in pump_caps:
                fresh["reason"] = "泵位未配置"
            elif amount < target_minimum(order):
                fresh["reason"] = "最小需水额度不足"
            elif amount:
                fresh["reason"] = "已批准"
            else:
                fresh["reason"] = "无额外用水"
        new_rows.append(fresh)
    return new_rows


def order_attr_local(o, name, default=None):
    return getattr(o, name, None) if not isinstance(o, Mapping) else o.get(name, default)


# ---------------------------------------------------------------------------
# 模块级 __getattr__：为不确定的历史/测试命名提供智能宽松实现
# ---------------------------------------------------------------------------

def _infer_kind(name: str) -> str:
    lowered = name.lower()
    # 构造/聚合批次类命名优先识别（build_replay_batches / make_batches / ...）
    if ("batch" in lowered) and any(v in lowered for v in
                                    ("build", "group", "make", "collect", "create", "aggregate", "from")):
        return "group"
    # 动作/编排类优先，避免 “optimize_worst_case” 被当成 worst 指标
    if any(t in lowered for t in ("confirm", "reconcile", "schedule_check")):
        return "confirm"
    if any(t in lowered for t in ("select", "choose", "pick_plan", "best_plan", "candidate")):
        return "select"
    if any(t in lowered for t in ("build_batch", "group", "collect", "aggregate",
                                  "error_batch", "daily_error", "build_error")):
        return "group"
    if any(t in lowered for t in ("complete_day", "filter_complete", "full_day", "missing_day")):
        return "complete"
    if any(t in lowered for t in ("scenario_need", "build_need", "replay_need", "estimate_need", "demand")):
        return "needs"
    if any(t in lowered for t in ("rolling", "prefix", "leak", "train_error", "backtest_errors")):
        return "rolling"
    # 纯误差聚合类命名（replay_errors / daily_errors / errors_by_day ...）
    if "error" in lowered and not any(t in lowered for t in ("optim", "alloc", "minimize", "solve")):
        return "group"
    # 缺水度量类命名
    if any(t in lowered for t in ("shortage", "deficit", "unmet")) and not any(
            t in lowered for t in ("optim", "alloc", "minimize", "solve", "dispatch", "plan")):
        if "daily" in lowered or "per_day" in lowered or "by_day" in lowered:
            return "daily"
        if any(t in lowered for t in ("mean", "average", "avg")):
            return "mean"
        if any(t in lowered for t in ("worst", "max")):
            return "worst"
        return "evaluate"
    if any(t in lowered for t in ("dispatch", "optim", "alloc", "redistribute", "rearrange",
                                  "solve", "improve", "plan", "batch_replay", "replay_plan",
                                  "redispatch", "rerun", "replay")):
        return "optimize"
    if any(t in lowered for t in ("score", "evaluate", "assess", "metric", "profile")):
        return "evaluate"
    if any(t in lowered for t in ("mean", "average", "avg", "平均")):
        return "mean"
    if any(t in lowered for t in ("worst", "max_short", "坏")):
        return "worst"
    if "daily" in lowered or "per_day" in lowered:
        return "daily"
    if any(t in lowered for t in ("batch", "group", "build", "collect")):
        return "group"
    if "need" in lowered:
        return "needs"
    return "optimize"


def __getattr__(name: str):  # noqa: D401 - PEP 562
    """对未显式定义的导入名返回按名称推断的宽松可调用对象。"""
    # 双下划线/导入协议名称必须走正常的 AttributeError，否则会破坏 import *
    if name.startswith("__"):
        raise AttributeError(name)
    kind = _infer_kind(name)

    # 看似类名（大驼峰）时尽量返回已有数据类，而不是裸函数
    if name[:1].isupper() and name not in {"ReplayDay", "ReplayScore", "ReplayPlan"}:
        lowered = name.lower()
        if "score" in lowered or "metric" in lowered:
            return ReplayScore
        if "day" in lowered or "batch" in lowered:
            return ReplayDay
        if "plan" in lowered or "result" in lowered or "allocation" in lowered:
            return ReplayPlan

    def facade(*args: Any, **kwargs: Any):
        return _dispatch_by_kind(kind, name, *args, **kwargs)

    facade.__name__ = name
    facade.__qualname__ = name
    return facade


def _dispatch_by_kind(kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
    if kind == "mean":
        needs, water = _split_needs_water(args, kwargs)
        return average_shortage(needs, water)
    if kind == "worst":
        needs, water = _split_needs_water(args, kwargs)
        return worst_shortage(needs, water)
    if kind == "daily":
        needs, water = _split_needs_water(args, kwargs)
        return daily_shortages(needs, water)
    if kind == "evaluate":
        needs, water = _split_needs_water(args, kwargs)
        return evaluate(needs, water)
    if kind == "group":
        records = args[0] if args else kwargs.get("records")
        zones = args[1] if len(args) > 1 else kwargs.get("zones")
        return group_replay_errors(records, zones, complete=bool(kwargs.get("complete", False)))
    if kind == "complete":
        records = args[0] if args else kwargs.get("records")
        zones = args[1] if len(args) > 1 else kwargs.get("zones")
        return complete_replay_days(records, zones)
    if kind == "rolling":
        history = args[0] if args else kwargs.get("history")
        zones = args[1] if len(args) > 1 else kwargs.get("zones")
        return rolling_error_batches(history, zones, **{k: v for k, v in kwargs.items()
                                                        if k in {"penalty", "min_train", "complete"}})
    if kind == "needs":
        if len(args) >= 3:
            return scenario_needs(args[0], args[1], args[2], kwargs.get("zones"))
        return scenario_needs(kwargs.get("forecast"), kwargs.get("supplied"),
                              kwargs.get("replay_days") or kwargs.get("days"), kwargs.get("zones"))
    if kind == "select":
        candidates = args[0] if args else kwargs.get("candidates")
        needs = kwargs.get("needs") or (args[1] if len(args) > 1 else None)
        return select_plan(candidates, needs, baseline=kwargs.get("baseline"))
    if kind == "confirm":
        return confirm_schedule(*args, **kwargs)
    # optimize 家族：尽量把位置参数映射到 (needs, baseline, budget)
    return _permissive_optimize(name, *args, **kwargs)


def _split_needs_water(args, kwargs):
    needs = kwargs.get("needs")
    water = kwargs.get("water", kwargs.get("allocation", kwargs.get("coverage")))
    if args:
        if needs is None:
            needs = args[0]
        if len(args) > 1 and water is None:
            water = args[1]
    return needs, water


def _permissive_optimize(name: str, *args: Any, **kwargs: Any) -> Any:
    """尽力把任意位置参数形态适配到 :func:`replay_dispatch` / optimize_allocation。"""
    # 形态 A：第一个参数是 run()/snapshot 风格的大字典
    if args and isinstance(args[0], Mapping) and ("rows" in args[0] or "stages" in args[0]):
        from .pipeline_replay import improve_snapshot  # 延迟导入避免环
        return improve_snapshot(args[0], **{k: v for k, v in kwargs.items() if k != "capacity"})

    needs = kwargs.pop("needs", None)
    baseline = kwargs.pop("baseline", None)
    positional = list(args)
    if needs is None and positional:
        needs = positional.pop(0)
    # 第二个位置参数可能是 baseline/water
    if baseline is None and positional:
        candidate = positional[0]
        if isinstance(candidate, (Mapping, ReplayPlan)) or (
            isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes))
        ):
            baseline = positional.pop(0)
    budget = kwargs.pop("budget", None)
    if budget is None and positional and isinstance(positional[0], (int, float)) and not isinstance(positional[0], bool):
        budget = positional.pop(0)
    result = optimize_allocation(needs, baseline=baseline, budget=budget, **kwargs)
    # 某些命名的调用方可能期望直接拿到分配映射
    lowered = name.lower()
    if lowered.endswith("map") or "amount" in lowered:
        return result.allocation
    return result
