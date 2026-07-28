# -*- coding: utf-8 -*-
"""
本地导出：趋势策略优化版 → results/
供 Streamlit Cloud 只读展示（不在云端跑全量回测）。

用法：
  python D:\\ALL\\trend_opt_streamlit\\export_cloud_results.py
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
RESULTS = PKG / "results"
ALL_ROOT = Path(r"D:\ALL")
TREND_FILE = ALL_ROOT / "趋势策略优化版.py"
ASSET_FOLDER = r"D:\数据集文件\期货全品种日线数据"
INDEX_FOLDER = r"D:\数据集文件\日线"
BENCH_NAME = "文华商品"
START, END = "2020-01-01", "2026-12-31"
TREND_UI_MARKER = "# ================= 7. UI 主控制渲染 ================="

for p in (ALL_ROOT, ALL_ROOT / "轮动策略", str(PKG)):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _st_noop(*a, **k):
    return None


class _ProgStub:
    def progress(self, *a, **k):
        return None

    def empty(self, *a, **k):
        return None


class _StStub:
    def __getattr__(self, name):
        return _st_noop


def load_trend_engine():
    src = TREND_FILE.read_text(encoding="utf-8-sig")
    if TREND_UI_MARKER in src:
        src = src.split(TREND_UI_MARKER, 1)[0]
    src = "\n".join(ln for ln in src.splitlines() if "st.set_page_config" not in ln)
    fake_st = types.ModuleType("streamlit")
    fake_st.__dict__.update({
        "cache_data": lambda *a, **k: (lambda fn: fn),
        "cache_resource": lambda *a, **k: (lambda fn: fn),
        "progress": lambda *a, **k: _ProgStub(),
        "spinner": lambda *a, **k: _st_noop(),
        "warning": _st_noop, "caption": _st_noop, "write": _st_noop,
        "markdown": _st_noop, "error": _st_noop, "info": _st_noop,
        "session_state": {}, "columns": lambda *a, **k: (_st_noop(), _st_noop()),
        "slider": lambda *a, **k: 0.25, "number_input": lambda *a, **k: 0,
        "checkbox": lambda *a, **k: False, "text_input": lambda *a, **k: "",
        "date_input": lambda *a, **k: pd.Timestamp("2020-01-01"),
        "expander": lambda *a, **k: _st_noop(), "sidebar": _StStub(),
        "dataframe": _st_noop, "metric": _st_noop, "divider": _st_noop,
        "header": _st_noop, "subheader": _st_noop, "set_option": _st_noop,
        "pyplot": _st_noop, "code": _st_noop, "radio": lambda *a, **k: None,
        "tabs": lambda *a, **k: [_st_noop()] * 9, "button": lambda *a, **k: False,
        "selectbox": lambda *a, **k: None,
    })
    mod = types.ModuleType("trend_opt_cloud_export")
    g = mod.__dict__
    g["st"] = fake_st
    g["__name__"] = "trend_opt_cloud_export"
    g["__file__"] = str(TREND_FILE)
    real_st = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake_st
    try:
        exec(compile(src, str(TREND_FILE), "exec"), g)
    finally:
        if real_st is not None:
            sys.modules["streamlit"] = real_st
        else:
            sys.modules["streamlit"] = importlib.import_module("streamlit")
    return mod


def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna().astype(float)
    tot = float(nav.iloc[-1] - 1)
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    ann = (1 + tot) ** (365.25 / days) - 1
    dr = nav.pct_change().dropna()
    sh = float((dr.mean() / dr.std()) * np.sqrt(252)) if len(dr) and dr.std() else 0.0
    dd = float((nav / nav.cummax() - 1).min())
    cal = abs(ann / dd) if dd else 0.0
    win = float((dr > 0).mean()) if len(dr) else 0.0
    return {
        "tot_pct": tot * 100, "ann_pct": ann * 100, "sharpe": sh,
        "dd_pct": dd * 100, "calmar": cal, "win_rate": win,
        "final": float(nav.iloc[-1]), "days": int(len(nav)),
    }


def hold_pipe(d: dict, key: str) -> str:
    items = d.get(key) or {}
    return "|".join(f"{k}:{float(v):.4f}" for k, v in items.items() if v and float(v) > 0)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=== load trend engine ===", flush=True)
    trend = load_trend_engine()
    print(f"=== panels {ASSET_FOLDER} ===", flush=True)
    panels = trend.load_data_and_calc_metrics(ASSET_FOLDER, 20, _panel_dedupe_version=3)
    (
        df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs,
        df_vol, df_vol_ma, df_amount, df_liquidity, err,
    ) = panels
    if err:
        raise RuntimeError(err)

    data_end = str(df_p.index.max().date())
    end = data_end

    params = {
        "periods": [5, 10, 20, 60],
        "hold_num": 4,
        "max_per_sector": 2,
        "ma": 60,
        "liquidity_mode": "original",
        "use_legacy_pct_stop": True,
        "adv_unconditional_high_dd": True,
        "swing_peak_stop": False,
        "swing_peak_stop_pct": 0.04,
        "stop_loss_trail": 0.04,
        "stop_loss_hard": 0.04,
        "stop_atr_trail": 4.0,
        "stop_atr_hard": 5.5,
        "stop_atr_chand": 4.5,
        "stop_atr_adv": 3.0,
        "start_date": START,
        "end_date": end,
        "commission": 1.5 / 10000.0,
        "target_volatility": 0.25,
        "use_vol_scaling": True,
        "use_steady_filter": True,
        "er_win": 20,
        "er_thresh": 0.30,
        "net_exposure_target": 1.0,
        "max_gross_exposure": 1.0,
        "max_margin_usage": 0.35,
        "max_single_weight": 0.30,
        "mom_tolerance": 1,
        "rank_protect_enabled": True,
        "rank_protect_threshold": 8,
        "use_regime_gate": True,
        "regime_use_er": True,
        "use_short_symmetric_filter": True,
        "use_short_ois_spec_gate": False,
        "use_limit_board": True,
        "limit_data_folder": ASSET_FOLDER,
        "_lb_events": [],
        "_lb_stats": {"buy_blocked": 0, "sell_blocked": 0},
    }

    print(f"=== backtest full {START} ~ {end} ===", flush=True)
    res_nav, res_logs, _dbg, res_contrib, details = trend.run_long_short_strategy(
        df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs,
        df_vol, df_vol_ma, df_amount, df_liquidity, params,
    )
    if res_nav is None or res_nav.empty:
        raise RuntimeError("空结果")

    # 基准
    bench_s = None
    idx_p, err_idx = trend.load_index_close_panel(INDEX_FOLDER)
    if err_idx:
        print("bench warn:", err_idx, flush=True)
    if idx_p is not None:
        col = next((c for c in idx_p.columns if BENCH_NAME in str(c)), None)
        if col:
            bs = idx_p.loc[res_nav.index[0]:res_nav.index[-1], col].dropna()
            if len(bs):
                bench_s = (bs / bs.iloc[0]).reindex(res_nav.index).ffill()

    def dump_window(tag: str, nav_df: pd.DataFrame, det: list, logs: list, contrib: dict, lb_stats: dict):
        """写出独立回测窗口，供云端与本地「按开始日独立回测」对齐。"""
        out_dir = RESULTS / "independent" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        nav_out = nav_df.reset_index()
        if nav_out.columns[0] != "date":
            nav_out = nav_out.rename(columns={nav_out.columns[0]: "date"})
        nav_out.to_csv(out_dir / "nav.csv", index=False, encoding="utf-8-sig")

        rows = []
        for d in det or []:
            stops = d.get("stops") or []
            stop_txt = "；".join(
                f"{s.get('asset', '')}({s.get('dir', '')}){s.get('reason', '')}" for s in stops
            )
            rows.append({
                "date": pd.Timestamp(d["date"]),
                "ret": float(d.get("ret", 0.0)),
                "nav": float(d["nav"]) if d.get("nav") is not None else np.nan,
                "long_exposure": float(d.get("long_exposure", 0.0) or 0.0),
                "short_exposure": float(d.get("short_exposure", 0.0) or 0.0),
                "gross_exposure": float(d.get("gross_exposure", 0.0) or 0.0),
                "net_exposure": float(d.get("net_exposure", 0.0) or 0.0),
                "margin_usage": float(d.get("margin_usage", 0.0) or 0.0),
                "gap_long_pnl": float(d.get("gap_long_pnl", 0.0) or 0.0),
                "gap_short_pnl": float(d.get("gap_short_pnl", 0.0) or 0.0),
                "stop_count": len(stops),
                "long_hold": hold_pipe(d, "start_hold_long"),
                "short_hold": hold_pipe(d, "start_hold_short"),
                "next_long": hold_pipe(d, "next_day_hold_long"),
                "next_short": hold_pipe(d, "next_day_hold_short"),
                "stops": stop_txt,
                "banned": "|".join(str(x) for x in (d.get("banned_list") or [])),
                "n_long": len([k for k, v in (d.get("start_hold_long") or {}).items() if v and v > 0]),
                "n_short": len([k for k, v in (d.get("start_hold_short") or {}).items() if v and v > 0]),
            })
        daily_w = pd.DataFrame(rows).sort_values("date")
        daily_w.to_csv(out_dir / "daily.csv", index=False, encoding="utf-8-sig")
        (out_dir / "log.txt").write_text("\n".join(logs or []), encoding="utf-8")
        cdf = pd.DataFrame(list((contrib or {}).items()), columns=["Asset", "Contribution"])
        cdf.sort_values("Contribution", ascending=False, inplace=True)
        cdf.to_csv(out_dir / "contrib.csv", index=False, encoding="utf-8-sig")
        _, ws = trend.compute_holding_wave_stats(det, nav_df.index)
        if ws is not None and not ws.empty:
            wsc = ws.copy()
            for c in ("Start_Date", "End_Date"):
                if c in wsc.columns:
                    wsc[c] = pd.to_datetime(wsc[c]).dt.strftime("%Y-%m-%d")
            wsc.to_csv(out_dir / "waves.csv", index=False, encoding="utf-8-sig")
        nav_s = nav_df["nav"].astype(float)
        m = metrics(nav_s)
        meta_w = {
            "tag": tag,
            "start": str(nav_s.index[0].date()),
            "end": str(nav_s.index[-1].date()),
            "mode": "independent",
            "metrics": m,
            "avg_long_exposure": float(daily_w["long_exposure"].mean()) if len(daily_w) else 0.0,
            "avg_short_exposure": float(daily_w["short_exposure"].mean()) if len(daily_w) else 0.0,
            "avg_daily_holdings": float((daily_w["n_long"] + daily_w["n_short"]).mean()) if len(daily_w) else 0.0,
            "avg_hold_days": float(ws["Hold_Days"].mean()) if ws is not None and not ws.empty else 0.0,
            "lb_stats": lb_stats or {},
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta_w, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return meta_w

    # —— 独立分年（对齐本地侧栏改开始日期）——
    ind_index = []
    for y in range(2020, pd.Timestamp(end).year + 1):
        ys, ye = f"{y}-01-01", min(f"{y}-12-31", end)
        if pd.Timestamp(ys) > pd.Timestamp(end):
            continue
        print(f"=== independent {y}: {ys} ~ {ye} ===", flush=True)
        p = {**params, "start_date": ys, "end_date": ye,
             "_lb_stats": {"buy_blocked": 0, "sell_blocked": 0}, "_lb_events": []}
        nav_y, logs_y, _, contrib_y, det_y = trend.run_long_short_strategy(
            df_p, df_atr_norm, df_l, df_o, df_h, df_atr_abs,
            df_vol, df_vol_ma, df_amount, df_liquidity, p,
        )
        if nav_y is None or nav_y.empty:
            continue
        mw = dump_window(str(y), nav_y, det_y, logs_y, contrib_y, dict(p.get("_lb_stats") or {}))
        ind_index.append({
            "tag": str(y),
            "start": mw["start"],
            "end": mw["end"],
            "tot_pct": mw["metrics"]["tot_pct"],
            "sharpe": mw["metrics"]["sharpe"],
            "final": mw["metrics"]["final"],
        })
        print(f"  {y}: nav={mw['metrics']['final']:.4f} sh={mw['metrics']['sharpe']:.3f}", flush=True)

    (RESULTS / "independent" / "index.json").write_text(
        json.dumps(ind_index, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    nav_out = res_nav.copy()
    nav_out = nav_out.reset_index().rename(columns={"index": "date"})
    if "date" not in nav_out.columns:
        nav_out = res_nav.reset_index()
        if nav_out.columns[0] != "date":
            nav_out = nav_out.rename(columns={nav_out.columns[0]: "date"})
    if bench_s is not None:
        nav_out["bench"] = bench_s.values
    nav_out.to_csv(RESULTS / "nav.csv", index=False, encoding="utf-8-sig")

    # 日明细
    rows = []
    n_global = 0
    for d in details or []:
        stops = d.get("stops") or []
        for s in stops:
            if "全局预警" in str(s.get("reason", "")):
                n_global += 1
        stop_txt = "；".join(
            f"{s.get('asset', '')}({s.get('dir', '')}){s.get('reason', '')}" for s in stops
        )
        rows.append({
            "date": pd.Timestamp(d["date"]),
            "ret": float(d.get("ret", 0.0)),
            "nav": float(d["nav"]) if d.get("nav") is not None else np.nan,
            "long_exposure": float(d.get("long_exposure", 0.0) or 0.0),
            "short_exposure": float(d.get("short_exposure", 0.0) or 0.0),
            "gross_exposure": float(d.get("gross_exposure", 0.0) or 0.0),
            "net_exposure": float(d.get("net_exposure", 0.0) or 0.0),
            "margin_usage": float(d.get("margin_usage", 0.0) or 0.0),
            "gap_long_pnl": float(d.get("gap_long_pnl", 0.0) or 0.0),
            "gap_short_pnl": float(d.get("gap_short_pnl", 0.0) or 0.0),
            "stop_count": len(stops),
            "long_hold": hold_pipe(d, "start_hold_long"),
            "short_hold": hold_pipe(d, "start_hold_short"),
            "next_long": hold_pipe(d, "next_day_hold_long"),
            "next_short": hold_pipe(d, "next_day_hold_short"),
            "stops": stop_txt,
            "banned": "|".join(str(x) for x in (d.get("banned_list") or [])),
            "n_long": len([k for k, v in (d.get("start_hold_long") or {}).items() if v and v > 0]),
            "n_short": len([k for k, v in (d.get("start_hold_short") or {}).items() if v and v > 0]),
        })
    daily = pd.DataFrame(rows).sort_values("date")
    daily.to_csv(RESULTS / "daily.csv", index=False, encoding="utf-8-sig")

    # 周
    nav = res_nav["nav"].astype(float)
    wrows = []
    iso = nav.index.isocalendar()
    for (y, w), g in nav.groupby([iso.year, iso.week]):
        ret = nav.pct_change()
        week_ret = float(np.prod(1.0 + ret.loc[g.index].fillna(0.0)) - 1.0)
        wrows.append({
            "周次": f"{int(y)}-W{int(w):02d}",
            "起始": str(g.index[0].date()),
            "结束": str(g.index[-1].date()),
            "周收益": week_ret,
            "周末净值": float(g.iloc[-1]),
        })
    pd.DataFrame(wrows).to_csv(RESULTS / "weekly.csv", index=False, encoding="utf-8-sig")

    # 年（切片收益，非独立回测）
    yrows = []
    for y, g in nav.groupby(nav.index.year):
        if len(g) >= 2:
            yrows.append({"年份": int(y), "收益%": float(g.iloc[-1] / g.iloc[0] - 1) * 100})
    pd.DataFrame(yrows).to_csv(RESULTS / "yearly.csv", index=False, encoding="utf-8-sig")

    # 贡献
    cdf = pd.DataFrame(list((res_contrib or {}).items()), columns=["Asset", "Contribution"])
    cdf.sort_values("Contribution", ascending=False, inplace=True)
    cdf.to_csv(RESULTS / "contrib.csv", index=False, encoding="utf-8-sig")

    # 波段
    holding_df, wave_stats = trend.compute_holding_wave_stats(details, res_nav.index)
    if wave_stats is not None and not wave_stats.empty:
        ws = wave_stats.copy()
        for c in ("Start_Date", "End_Date"):
            if c in ws.columns:
                ws[c] = pd.to_datetime(ws[c]).dt.strftime("%Y-%m-%d")
        ws.to_csv(RESULTS / "waves.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(RESULTS / "waves.csv", index=False, encoding="utf-8-sig")

    (RESULTS / "log.txt").write_text("\n".join(res_logs or []), encoding="utf-8")

    m = metrics(nav)
    avg_long = float(daily["long_exposure"].mean()) if len(daily) else 0.0
    avg_short = float(daily["short_exposure"].mean()) if len(daily) else 0.0
    avg_n = float((daily["n_long"] + daily["n_short"]).mean()) if len(daily) else 0.0
    avg_hold = float(wave_stats["Hold_Days"].mean()) if wave_stats is not None and not wave_stats.empty else 0.0
    lb = dict(params.get("_lb_stats") or {})

    meta = {
        "product": "趋势策略优化版",
        "source": str(TREND_FILE),
        "start": START,
        "end": end,
        "asset_folder": ASSET_FOLDER,
        "use_limit_board": True,
        "commission_bp": 1.5,
        "global_atr_crash": True,
        "global_atr_crash_k": 3.0,
        "use_regime_gate": True,
        "use_short_symmetric_filter": True,
        "use_legacy_pct_stop": True,
        "adv_unconditional_high_dd": True,
        "hold_num": 4,
        "max_per_sector": 2,
        "periods": [5, 10, 20, 60],
        "ma": 60,
        "target_volatility": 0.25,
        "metrics": m,
        "avg_long_exposure": avg_long,
        "avg_short_exposure": avg_short,
        "avg_daily_holdings": avg_n,
        "avg_hold_days": avg_hold,
        "global_crash_events": n_global,
        "lb_stats": lb,
        "bench_name": BENCH_NAME if bench_s is not None else None,
        "metric_note": "全样本为连续回测；results/independent/{年}/ 为各年独立回测（对齐本地改开始日期）",
        "independent_years": [x["tag"] for x in ind_index],
        "exported_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    (RESULTS / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(
        f"nav={m['final']:.4f} sharpe={m['sharpe']:.3f} dd={m['dd_pct']:.2f}% "
        f"global={n_global} lb={lb}",
        flush=True,
    )
    print(f"saved → {RESULTS}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
