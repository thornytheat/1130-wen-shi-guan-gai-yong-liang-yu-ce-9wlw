"""历史回放重排的规格守卫测试。

覆盖需求中的关键不变量：同日联合误差、缺棚整日剔除、前缀发布隔离、
场景需水计算、最坏日约束下最小平均缺水、泵班表对账以及无完整回放时回退。
"""

from __future__ import annotations

import itertools

import pytest

from greenhouse import batch_replay as br
from greenhouse.irrigation_quota.contracts import IrrigationOrder


ZONES = ["棚-A", "棚-B", "棚-C"]


def _needs():
    # 第 4 天三棚同日被低估，纯平均最优与最坏约束会给出不同选择
    return {
        1: {"棚-A": 10, "棚-B": 0, "棚-C": 0},
        2: {"棚-A": 9, "棚-B": 1, "棚-C": 0},
        3: {"棚-A": 8, "棚-B": 0, "棚-C": 1},
        4: {"棚-A": 2, "棚-B": 6, "棚-C": 6},
    }


def test_joint_day_errors_keep_same_day_vector():
    records = [
        {"回放日": 1, "棚区": "棚-A", "实际（升）": 12, "估计（升）": 10},
        {"回放日": 1, "棚区": "棚-B", "实际（升）": 5, "估计（升）": 2},
        (2, "棚-A", 2.0),
    ]
    days = br.complete_replay_days(records, ["棚-A", "棚-B"])
    assert [day.day for day in days] == [1]
    assert days[0].errors == {"棚-A": 2.0, "棚-B": 3.0}


def test_missing_single_zone_drops_whole_day_but_keeps_others():
    records = [(1, z, 1.0) for z in ZONES] + [(2, "棚-A", 1.0), (2, "棚-B", 2.0)]
    days = br.complete_replay_days(records, ZONES)
    assert [day.day for day in days] == [1]


def test_scenario_need_is_forecast_plus_error_minus_supplied():
    batches = [br.ReplayDay(7, {"棚-A": 3.0, "棚-B": -2.0})]
    needs = br.scenario_needs({"棚-A": 100, "棚-B": 50},
                              {"棚-A": 98, "棚-B": 60}, batches, ["棚-A", "棚-B"])
    assert needs[7] == {"棚-A": 5.0, "棚-B": 0.0}


def test_worst_day_sums_zones_so_joint_underestimate_counts():
    needs = _needs()
    # 完全不配水时，第 4 天合计 14 为最坏日
    assert br.worst_shortage(needs, {z: 0 for z in ZONES}) == pytest.approx(14)
    assert br.average_shortage(needs, {z: 0 for z in ZONES}) == pytest.approx(
        (10 + 10 + 9 + 14) / 4
    )


def _brute_optimum(needs, budget, baseline, caps):
    days, zones, matrix = br.coerce_needs(needs, ZONES)
    limit = br.worst_shortage(needs, baseline, zones=zones)
    best = None
    for values in itertools.product(*[range(min(budget, caps[z]) + 1) for z in zones]):
        if sum(values) > budget:
            continue
        water = dict(zip(zones, values))
        score = br.evaluate(needs, water, zones=zones)
        if score.worst > limit + 1e-9:
            continue
        key = (score.average, score.worst, values)
        if best is None or key < best[0]:
            best = (key, water, score)
    return best


def test_constrained_optimum_matches_brute_force():
    baseline = {"棚-A": 6, "棚-B": 2, "棚-C": 2}
    caps = {z: 10 for z in ZONES}
    plan = br.optimize_allocation(_needs(), budget=10, baseline=baseline,
                                  zones=ZONES, caps=caps)
    brute = _brute_optimum(_needs(), 10, baseline, caps)
    assert plan.worst <= br.worst_shortage(_needs(), baseline) + 1e-9
    assert plan.average == pytest.approx(brute[2].average)
    assert dict(plan) == {z: float(v) for z, v in brute[1].items()}


@pytest.mark.parametrize("seed", range(60))
def test_optimizer_always_respects_worst_bound_and_matches_bruteforce(seed):
    import random

    rng = random.Random(seed)
    n = rng.randint(2, 4)
    zones = [f"z{i}" for i in range(n)]
    ndays = rng.randint(1, 4)
    needs = {d: {z: rng.randint(0, 8) for z in zones} for d in range(ndays)}
    budget = rng.randint(0, 12)
    caps = {z: rng.randint(0, 8) for z in zones}
    # 原配水方案在规格中必然满足同一预算：随机生成后裁剪到预算内
    baseline = {z: rng.randint(0, min(4, caps[z])) for z in zones}
    overflow = sum(baseline.values()) - budget
    for z in zones:
        if overflow <= 0:
            break
        take = min(baseline[z], overflow)
        baseline[z] -= take
        overflow -= take
    plan = br.optimize_allocation(needs, budget=budget, baseline=baseline,
                                  zones=zones, caps=caps)
    limit = br.worst_shortage(needs, baseline, zones=zones)
    assert plan.worst <= limit + 1e-9
    brute = _brute_optimum_zones(needs, zones, budget, baseline, caps)
    if brute is not None:
        assert plan.average == pytest.approx(brute[2].average)


def _brute_optimum_zones(needs, zones, budget, baseline, caps):
    limit = br.worst_shortage(needs, baseline, zones=zones)
    best = None
    for values in itertools.product(*[range(min(budget, caps[z]) + 1) for z in zones]):
        if sum(values) > budget:
            continue
        water = dict(zip(zones, values))
        score = br.evaluate(needs, water, zones=zones)
        if score.worst > limit + 1e-9:
            continue
        key = (score.average, score.worst, values)
        if best is None or key < best[0]:
            best = (key, water, score)
    return best


def test_no_complete_replay_returns_original_plan():
    original = {"棚-A": 7, "棚-B": 3}
    plan = br.replay_dispatch(needs={}, baseline=original, complete=False)
    assert plan.complete is False
    assert plan.changed is False
    assert dict(plan) == {"棚-A": 7.0, "棚-B": 3.0}


def _order(zone, *, pump="P1", flow=10, window=10, upper=1000):
    return IrrigationOrder.parse({
        "zone": zone, "pump": pump, "crop": "瓜", "state": "允许配水",
        "upper": upper, "lower": 0, "urgency": 0.5,
        "per_min": flow, "window": window,
    })


def test_unschedulable_litres_do_not_count_as_covered():
    orders = [_order("棚-A", flow=10, window=2), _order("棚-B", flow=10, window=10)]
    drafted = [
        {"zone": "棚-A", "pump": "P1", "crop": "瓜", "amount_l": 200,
         "min_l": 0, "max_l": 1000, "reason": "已批准"},
        {"zone": "棚-B", "pump": "P1", "crop": "瓜", "amount_l": 0,
         "min_l": 0, "max_l": 1000, "reason": "无额外用水"},
    ]
    needs = {"d1": {"棚-A": 100, "棚-B": 100}, "d2": {"棚-A": 100, "棚-B": 0}}
    result = br.redispatch_orders(
        orders, drafted, needs, budget_l=200, pump_caps={"P1": 10000},
        pump_windows={"P1": [(0, 3)]}, breakdowns={},
    )
    report = result["report"]
    # 3 分钟泵窗口 * 10 升/分 = 最多 30 升可排班
    assert sum(report["selected"]["scheduled"].values()) <= 30
    assert report["worst_not_above_original"] is True
    assert report["selected"]["worst"] <= report["baseline"]["worst"] + 1e-6


def test_pipeline_replay_report_conserves_and_respects_worst_bound():
    from greenhouse.pipeline import run

    for capacity in (1, 2, 3, 17, 31, 65, 100):
        result = run(capacity)
        report = result["batch_replay"]
        assert report is not None
        total = sum(row["amount_l"] for row in result["rows"])
        assert total <= capacity * 100
        summary = result["shift_summary"]
        assert summary["已排升"] + summary["未排升"] == total
        if report.get("complete"):
            assert report["worst_not_above_original"] is True
            assert report["selected"]["worst"] <= report["baseline"]["worst"] + 1e-6


def test_tight_capacity_replay_concentrates_on_replay_need_zone():
    from greenhouse.pipeline import run

    original = run(65)
    tight = run(2)
    report = tight["batch_replay"]
    assert report["complete"] is True
    assert report["changed"] is True
    amounts = report["selected"]["amounts"]
    # 回放场景里只有 棚-01 仍有正剩余需水，200 升应全部排给它
    assert amounts["棚-01"] == 200
    assert report["selected"]["mean"] < report["baseline"]["mean"]
    assert report["selected"]["worst"] <= report["baseline"]["worst"]


def test_web_scenario_exposes_batch_replay():
    from greenhouse.delivery import application

    client = application.test_client()
    response = client.post("/api/scenario", json={"capacity": 2})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["batch_replay"]["complete"] is True
    replay = client.post("/api/replay", json={"capacity": 2})
    assert replay.status_code == 200
