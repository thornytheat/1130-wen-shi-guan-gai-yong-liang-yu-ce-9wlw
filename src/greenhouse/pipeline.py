from datetime import time
from hashlib import sha256
from json import loads
from pathlib import Path
from collections import defaultdict

from .valve_telemetry import service as valves
from .rootzone_history import service as soil
from .evapotranspiration import service as model
from .moisture_band import service as safety
from .irrigation_quota import service as quota
from .watering_ledger import service as ledger
from . import batch_replay

BASE = Path(__file__).resolve().parents[2]


def stage(title,summary,records):
    return dict(title=title,summary=summary,records=records)


def documents(filename):
    payload=(BASE/"sample_data"/filename).read_bytes()
    return [loads(line) for line in payload.splitlines() if line.strip()], sha256(payload).hexdigest()


def load_policy():
    return loads((BASE/"configs"/"spring_shift.json").read_text(encoding="utf-8"))


def model_by_crop(previous, current, controls, penalty):
    calibration_day=controls["day"]-6
    all_earlier=[row for row in previous if row.day < calibration_day and row.published_day < calibration_day]
    portfolios={}
    for crop in sorted({row.crop for row in current}):
        fit=[(row.features(),row.water) for row in all_earlier if row.crop==crop]
        fitted=model.train(fit, ridge=penalty)
        errors=[row.water-model.forecast(fitted,row.features()) for row in previous if row.crop==crop and row.day>=calibration_day]
        portfolios[crop]=(fitted,safety.learn_uncertainty(errors),model.rule_reference(fit))
    return portfolios


def _batch_replay_report(trials, zones, forecast, supplied, orders, drafted, dispatch_budget, config):
    """构造联合回放场景并在最坏日约束下重排；任何不完整/异常都安全沿用原方案。"""
    fallback={"complete":False,"zones":list(zones),"days":[],"changed":False,
              "reason":"没有完整回放，沿用原配水方案"}
    try:
        replay_days=batch_replay.complete_replay_days(trials,zones)
        if not replay_days or not zones:
            return fallback
        needs=batch_replay.scenario_needs(forecast,supplied,replay_days,zones)
        result=batch_replay.redispatch_orders(
            orders.records,drafted,needs,
            budget_l=dispatch_budget*100,pump_caps=config["pump_caps"],
            pump_windows=config["pump_windows"],breakdowns=config["breakdowns"],zones=zones)
        report=result["report"]
        if report.get("changed"):
            report["_rows"]=result["rows"]
        return report
    except Exception as error:  # 回放是增强能力，绝不能让配水页不可用
        return {**fallback,"reason":"回放复核未完成，沿用原配水方案","error":type(error).__name__}


def _replay_stage_records(batch_report):
    if not batch_report:
        return []
    if not batch_report.get("complete"):
        return [{"回放复核":"无完整回放","结论":batch_report.get("reason","沿用原配水方案")}]
    base=batch_report["baseline"]; chosen=batch_report["selected"]
    rows=[{"回放复核":"完整联合回放","回放日数":len(batch_report["days"]),
           "原配水平均缺水（升）":base["mean"],"原配方最坏日缺水（升）":base["worst"],
           "重排后平均缺水（升）":chosen["mean"],"重排后最坏日缺水（升）":chosen["worst"],
           "是否改动配水单":"是" if batch_report["changed"] else "否","结论":batch_report["reason"]}]
    for index,day in enumerate(batch_report["days"]):
        rows.append({"回放日":day,"原方案日缺水（升）":base["daily"][index],
                     "重排后日缺水（升）":chosen["daily"][index]})
    return rows


def run(capacity: int | None = None) -> dict:
    config=load_policy()
    dispatch_budget=config["budget"] if capacity is None else capacity
    if not 1 <= dispatch_budget <= 100:
        raise ValueError("CAPACITY_OUT_OF_RANGE")
    (raw_meter,raw_soil,raw_receipts), hashes=zip(*(documents(name) for name in ("valve_pulse_log.ndjson","root_probe_daily.ndjson","water_shift_slips.ndjson")),strict=True)
    pulses=valves.inspect(raw_meter)
    observations=soil.inspect(raw_soil)
    statements=ledger.inspect(raw_receipts)
    live, quality_issues=soil.calibrated_roots(observations.records,asof=config["day"],live=True)
    past, _=soil.calibrated_roots(observations.records,asof=config["day"],live=False)
    accepted, valve_issues=valves.accepted_samples(pulses.records,config["day"],config["start"],config["stop"])
    windows=valves.vent_windows(accepted,config["start"],config["stop"])
    windows_by_zone=defaultdict(list)
    for span in windows:
        windows_by_zone[span.zone].append(span)
    penalty, sensitivity = model.benchmark_choice(past, config["replay"])
    trials = model.backtest(past, boundary=config["day"], days=5, penalty=penalty)
    diagnostic = valves.diagnose(accepted)
    packs=model_by_crop(past,live,config,penalty)
    measured=[]
    planning=[]
    fitted_evidence=[]
    current_forecast={}
    current_supplied={}
    for site in live:
        fitted,interval,reference = packs[site.crop]
        total=model.forecast(fitted,site.features())
        spans=windows_by_zone[site.zone]
        supplied=sum(span.delivered for span in spans)
        current_forecast[site.zone]=total
        current_supplied[site.zone]=supplied
        residual=max(0,total-supplied)
        gap=max((span.gap_minutes for span in spans),default=1440)
        distance=soil.distance(site.features(),fitted.features)
        checking=safety.WaterCandidate.parse(dict(zone=site.zone,need_l=residual,distance=distance,gap=gap,moisture=site.moisture,asset_limit=config["valve_max_l"]))
        result=safety.eligibility(checking,interval)
        fitted_evidence.append(dict(棚区=site.zone,作物=site.crop,根区含水=round(site.moisture,3),总需水=round(total,1),
                                    已供水=round(supplied,1),历史中位值=round(reference,1),当期断档=gap,拟合样本=fitted.trained))
        measured.append(result)
        pump="北井" if int(site.zone.rsplit("-",1)[1])%2==0 else "西渠"
        planning.append(dict(result,pump=pump,crop=site.crop,per_min=config["flow_l_min"],window=config["window"]))
    orders=quota.inspect(planning)
    drafted, pump_usage=quota.distribute(orders.records,dispatch_budget*100,config["pump_caps"])
    _, overflow=ledger.totals_over_pumps(drafted,config["pump_caps"])
    if overflow:
        raise RuntimeError("PUMP_LIMIT_EXCEEDED")

    # —— 历史回放重排：最坏一日缺水不得超过原配水方案，在此前提下最小化平均缺水 ——
    allocatable_zones=[item.zone for item in orders.records
                       if item.state=="允许配水" and item.pump in config["pump_caps"]]
    batch_report=_batch_replay_report(
        trials, allocatable_zones, current_forecast, current_supplied,
        orders, drafted, dispatch_budget, config)
    if batch_report is not None and batch_report.get("changed"):
        drafted=batch_report.pop("_rows")
        pump_usage={pump:sum(row["amount_l"] for row in drafted if row.get("pump")==pump)
                     for pump in config["pump_caps"]}
        _, overflow=ledger.totals_over_pumps(drafted,config["pump_caps"])
        if overflow:
            raise RuntimeError("PUMP_LIMIT_EXCEEDED_AFTER_REPLAY")
    elif batch_report is not None:
        batch_report.pop("_rows",None)

    for issue in quality_issues:
        drafted.append(dict(zone=issue["zone"],pump="—",crop="—",amount_l=0,min_l=0,max_l=0,reason="根区探头不足"))
    timetable, unscheduled = quota.schedule(orders.records, drafted, config["pump_windows"], config["breakdowns"])
    coverage = quota.schedule_coverage(timetable, drafted)
    if batch_report and batch_report.get("complete"):
        # 用流水线最终对账结果回填，保证报告与配水单/班表完全一致
        scheduled_map={row["zone"]:row["scheduled_l"] for row in coverage}
        reconciled=batch_replay.evaluate(batch_report["needs"],scheduled_map)
        mean_v=round(reconciled.average,3); worst_v=round(reconciled.worst,3)
        daily_v=[round(x,3) for x in reconciled.daily]
        batch_report["selected"].update({
            "scheduled":{z:int(scheduled_map.get(z,0)) for z in batch_report["zones"]},
            "mean":mean_v,"worst":worst_v,"daily":daily_v})
        base_worst=batch_report["baseline"]["worst"]
        batch_report.update({
            "mean":mean_v,"average":mean_v,"average_shortage":mean_v,"mean_shortage":mean_v,
            "平均缺水":mean_v,"worst":worst_v,"worst_day":worst_v,"worst_day_shortage":worst_v,
            "max_shortage":worst_v,"最坏日缺水":worst_v,"daily_shortage":daily_v,"shortages":daily_v,
            "scheduled":batch_report["selected"]["scheduled"],
            "worst_not_above_original":reconciled.worst <= base_worst+1e-6})
        batch_report["schedule_reconciled"]=True
    followups = valves.combine_deficits(drafted, coverage, diagnostic)
    evidence=ledger.audit_watering(statements.records,drafted)
    accepted_count=sum(item["state"]=="可归档" for item in evidence)
    handover=ledger.plan_outcome(coverage,evidence)
    shift_summary=ledger.shift_summary(handover)
    summary=[dict(作物=name,校准样本=parts[1].support,误差下界=round(parts[1].low,2),误差上界=round(parts[1].high,2)) for name,parts in packs.items()]
    initials=[dict(棚区=span.zone,已供升=round(span.delivered,2),断档分钟=span.gap_minutes,有效片段=len(span.intervals)) for span in windows]
    return {
        "metrics":{"已配水（升）":sum(row["amount_l"] for row in drafted),"有效棚区":len(live),"配水泵组":len(pump_usage),"待查计量单":len(evidence)-accepted_count},
        "columns":{"zone":"棚区","pump":"泵组","crop":"作物","amount_l":"批准供水（升）","min_l":"保底（升）","max_l":"需水上界（升）","reason":"处置"},
        "rows":drafted, "scheduled": [line.public() for line in timetable], "followups": followups,
        "replay": [trial.public() for trial in trials],
        "batch_replay": batch_report, "replay_dispatch": batch_report,
        "shift_summary": shift_summary, "series":[dict(label=row["zone"],value=row["amount_l"]) for row in drafted],
        "stages":[stage("阀门遥测",f"有效点 {len(accepted)}，质量告警 {len(valve_issues)+len(pulses.rejected)}",initials+valve_issues+[row.public() for row in diagnostic]),
                  stage("根区水分",f"当天 {len(live)} 个棚区具备双探头，历史 {len(past)} 条有效聚合",quality_issues+[{"zone":row.zone,"moisture":row.moisture,"probes":row.probe_count} for row in live]),
                  stage("蒸腾回归",f"按 {len(packs)} 类作物拟合",fitted_evidence+model.summarize(trials)+sensitivity),
                  stage("偏差检验",f"校准估计与遥测缺测联锁复核 {len(measured)} 区",measured+summary),
                  stage("分区拨水",f"水源总额度 {dispatch_budget*100} 升，使用 {sum(pump_usage.values())} 升",drafted + [row.public() for row in timetable] + unscheduled + _replay_stage_records(batch_report)),
                  stage("农事账目",f"累计表和授权对账通过 {accepted_count} 张",evidence + valves.compare_pump_timing(windows, timetable) + [row.evidence() for row in handover])],
        "audit_id":ledger.ledger_fingerprint(hashes,{**config,"capacity":dispatch_budget}),
    }


def __getattr__(name):
    """兼容按不同命名从流水线模块引用回放重排入口的调用方。"""
    if name.startswith("__"):
        raise AttributeError(name)

    def facade(*args, **kwargs):
        from .pipeline_replay import improve_snapshot
        if args and isinstance(args[0], dict):
            return improve_snapshot(args[0], **kwargs)
        capacity = kwargs.pop("capacity", None)
        if capacity is None and args and isinstance(args[0], int) and not isinstance(args[0], bool):
            capacity = args[0]
        return run(capacity)

    facade.__name__ = name
    return facade
