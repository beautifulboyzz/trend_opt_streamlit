# -*- coding: utf-8 -*-
"""
CSRE 日频看板 · 商品期货截面残差动量（Push final）

只读本地预计算 CSV/JSON，不重跑引擎。路径相对本文件，不依赖云端目录。

启动（与趋势页 8506 并列）::

    streamlit run csre_dashboard/streamlit_app.py --server.port 8507
"""
from __future__ import annotations

import json
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
ANN = 242
PUSH_K = 2

DATE_COLS = ["date", "dt", "trade_date", "datetime", "日期", "Date"]
NAV_COLS = ["nav", "equity", "net_value", "wealth", "cum_nav", "净值", "nav_net", "port_nav"]
RET_COLS = ["ret", "daily_ret", "return", "pnl", "port_ret", "strategy_ret", "net_ret", "日收益"]
TURN_COLS = ["turnover", "turn", "换手", "turnover_rate"]
COST_COLS = ["cost", "fee", "daily_cost", "commission", "cost_ret", "fee_ret"]
CONE_COLS = ["cone", "cone_on", "gate", "in_cone", "cone_pass"]
SYMBOL_COLS = ["symbol", "asset", "product", "code", "variety", "品种", "ticker", "underlying"]
SIDE_COLS = ["side", "direction", "dir", "pos_side", "方向"]
Z_COLS = ["z", "resid_z", "zscore", "score_z"]
HL_COLS = ["hl", "hl_days", "holding_days", "hold_days", "HL"]
REASON_COLS = ["reason", "filter_reason", "drop_reason", "why", "过滤原因", "filter"]
CONTRIB_COLS = ["contribution", "contrib", "pnl_gross", "gross_pnl", "毛贡献", "pnl"]
WEIGHT_COLS = ["weight", "w", "target_w", "pos", "weight_after"]

COLOR_NAV = "#2ca02c"
COLOR_DD = "#d62728"
COLOR_TURN = "#1f77b4"

DEFAULT_PARAMS = {
    "nf": 1,
    "resid_z": 8,
    "factor_lb": 90,
    "cone_q": 0.35,
    "z": 1.0,
    "top_k": 2,
    "hl_max": 12,
    "vol": "8%",
    "pos_ewma": 1.0,
    "fee": "1.5bps",
    "ANN": ANN,
    "signal_mode": "momentum",
}


def setup_cn_font():
    candidates = [
        REPO_DIR / "fonts" / "SimHei.ttf",
        APP_DIR / "fonts" / "SimHei.ttf",
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            fm.fontManager.addfont(str(path))
        except Exception:
            pass
        prop = fm.FontProperties(fname=str(path))
        name = prop.get_name()
        plt.rcParams["font.family"] = name
        plt.rcParams["font.sans-serif"] = [name, "SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return prop
    plt.rcParams["axes.unicode_minus"] = False
    return fm.FontProperties(family=["DejaVu Sans"])


def artifacts_root() -> Path:
    env = os.environ.get("CSRE_ARTIFACTS", "").strip()
    if env:
        return Path(env)
    return APP_DIR / "artifacts_push"


def pick_col(df: pd.DataFrame | None, names: list[str]) -> str | None:
    if df is None or len(df.columns) == 0:
        return None
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_json(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_extract_archives() -> str | None:
    """If dashboard CSV is missing, unpack a local tgz/zip that contains artifacts_push."""
    if os.environ.get("CSRE_ARTIFACTS", "").strip():
        return None
    root = artifacts_root()
    daily = root / "dashboard" / "dashboard_daily.csv"
    if daily.exists() and daily.stat().st_size > 0:
        return None
    candidates = [
        APP_DIR / "csre_streamlit_pack.tgz",
        APP_DIR / "csre_dashboard_bundle.zip",
        Path(r"E:\策略") / "csre_dashboard_bundle.zip",
        Path(r"C:\Users\Admin\.cursor\projects\C-Users-Admin\uploads") / "csre_streamlit_pack.tgz",
    ]
    for src in candidates:
        if not src.exists() or src.stat().st_size < 1000:
            continue
        dest = APP_DIR
        try:
            if src.suffix.lower() == ".zip":
                with zipfile.ZipFile(src) as zf:
                    for info in zf.infolist():
                        pure = info.filename.replace("\\", "/")
                        if "artifacts_push/" not in pure:
                            continue
                        rel = pure[pure.index("artifacts_push"):]
                        target = (dest / rel).resolve()
                        if not str(target).startswith(str(dest.resolve())):
                            continue
                        if info.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as fh, open(target, "wb") as out:
                            out.write(fh.read())
            else:
                with tarfile.open(src, "r:*") as tf:
                    for member in tf.getmembers():
                        pure = member.name.replace("\\", "/")
                        if "artifacts_push/" not in pure:
                            continue
                        rel = pure[pure.index("artifacts_push"):]
                        target = (dest / rel).resolve()
                        if not str(target).startswith(str(dest.resolve())):
                            continue
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        extracted = tf.extractfile(member)
                        if extracted is None:
                            continue
                        with extracted, open(target, "wb") as out:
                            out.write(extracted.read())
        except (zipfile.BadZipFile, tarfile.TarError, OSError):
            continue
        if daily.exists() and daily.stat().st_size > 0:
            return str(src)
    return None


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    col = pick_col(df, DATE_COLS)
    out = df.copy()
    if col is None:
        out["date"] = pd.NaT
        return out
    out["date"] = pd.to_datetime(out[col], errors="coerce")
    return out


def attach_nav(df: pd.DataFrame) -> pd.DataFrame:
    out = parse_dates(df)
    nav_col = pick_col(out, NAV_COLS)
    ret_col = pick_col(out, RET_COLS)
    if nav_col is not None:
        out["nav"] = pd.to_numeric(out[nav_col], errors="coerce")
    elif ret_col is not None:
        ret = pd.to_numeric(out[ret_col], errors="coerce").fillna(0.0)
        out["nav"] = (1.0 + ret).cumprod()
    else:
        out["nav"] = np.nan
    if ret_col is not None:
        out["ret"] = pd.to_numeric(out[ret_col], errors="coerce")
    else:
        out["ret"] = out["nav"].pct_change()
    return out.dropna(subset=["date"]).sort_values("date")


def load_topk_curves(folder: Path) -> dict[int, pd.DataFrame]:
    curves: dict[int, pd.DataFrame] = {}
    if not folder.exists():
        return curves
    for path in sorted(folder.glob("equity_topk*.csv")):
        match = re.search(r"(\d+)", path.stem)
        if not match:
            continue
        df = read_table(path)
        if df is None:
            continue
        curves[int(match.group(1))] = attach_nav(df)
    return curves


def load_bundle() -> dict:
    maybe_extract_archives()
    root = artifacts_root()
    dash = root / "dashboard"
    topk = root / "topk_compare"
    daily_raw = read_table(dash / "dashboard_daily.csv")
    daily = attach_nav(daily_raw) if daily_raw is not None else None
    legs = read_table(dash / "dashboard_legs.csv")
    if legs is not None:
        legs = parse_dates(legs)
    trades = read_table(dash / "dashboard_trades.csv")
    if trades is not None:
        trades = parse_dates(trades)
    yearly = read_table(dash / "dashboard_yearly.csv")
    contrib = read_table(dash / "dashboard_contrib.csv")
    return {
        "root": root,
        "daily": daily,
        "legs": legs,
        "trades": trades,
        "yearly": yearly,
        "contrib": contrib,
        "universe": read_json(dash / "dashboard_universe.json"),
        "meta_doc": read_json(dash / "dashboard_data.json") or {},
        "topk": load_topk_curves(topk),
        "topk_metrics": read_table(topk / "metrics_by_topk.csv"),
    }


def params_from_meta(doc: dict) -> dict:
    params = dict(DEFAULT_PARAMS)
    if not isinstance(doc, dict):
        return params
    blobs = [doc]
    for key in ("meta", "params", "config"):
        if isinstance(doc.get(key), dict):
            blobs.append(doc[key])
    for blob in blobs:
        for key, value in blob.items():
            if key in params or key in DEFAULT_PARAMS:
                params[key] = value
    return params


def calc_metrics(nav: pd.Series) -> dict:
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    empty = {
        "tot": 0.0, "ann": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0,
        "win": 0.0, "final": 1.0, "days": int(len(nav)),
    }
    if len(nav) < 2:
        if len(nav):
            empty["final"] = float(nav.iloc[-1])
        return empty
    base = float(nav.iloc[0]) or 1.0
    norm = nav / base
    tot = float(norm.iloc[-1] - 1.0)
    periods = max(len(norm) - 1, 1)
    ann = (1.0 + tot) ** (ANN / periods) - 1.0
    dr = norm.pct_change().dropna()
    sharpe = float((dr.mean() / dr.std()) * np.sqrt(ANN)) if len(dr) and float(dr.std()) else 0.0
    dd = float((norm / norm.cummax() - 1.0).min())
    calmar = abs(ann / dd) if dd else 0.0
    win = float((dr > 0).mean()) if len(dr) else 0.0
    return {
        "tot": tot, "ann": ann, "sharpe": sharpe, "dd": dd, "calmar": calmar,
        "win": win, "final": float(norm.iloc[-1]), "days": int(len(nav)),
    }


def slice_by_date(df: pd.DataFrame | None, start, end) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame() if df is None else df.iloc[0:0]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    return df.loc[mask].copy()


def yearly_from_nav(nav: pd.Series) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()
    rows = []
    s = nav.dropna().copy()
    s.index = pd.to_datetime(s.index)
    for year, part in s.groupby(s.index.year):
        m = calc_metrics(part)
        rows.append({
            "year": int(year),
            "total_return": m["tot"],
            "ann_return": m["ann"],
            "sharpe": m["sharpe"],
            "max_dd": m["dd"],
            "days": m["days"],
        })
    return pd.DataFrame(rows)


def universe_for_day(obj, day) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    key = pd.Timestamp(day).strftime("%Y-%m-%d")
    records = None
    if isinstance(obj, dict):
        for bucket_name in (None, "by_date", "dates", "universe", "data"):
            bucket = obj if bucket_name is None else obj.get(bucket_name)
            if isinstance(bucket, dict) and key in bucket:
                records = bucket[key]
                break
            if isinstance(bucket, list):
                records = bucket
                break
        if records is None:
            for value in obj.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    records = value
                    break
    elif isinstance(obj, list):
        records = obj
    if isinstance(records, dict):
        rows = []
        for sym, info in records.items():
            if isinstance(info, dict):
                rows.append({"symbol": sym, **info})
            else:
                rows.append({"symbol": sym, "value": info})
        records = rows
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = parse_dates(df)
    if df["date"].notna().any():
        df = df[df["date"].dt.strftime("%Y-%m-%d") == key]
    return df


def numeric_metric_blocks(doc: dict) -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    if not isinstance(doc, dict):
        return found

    def _walk(node, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        numeric = {
            k: v for k, v in node.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        keys = " ".join(str(k).lower() for k in numeric)
        if numeric and any(tok in keys for tok in ("sharpe", "ann", "return", "dd", "calmar", "ret")):
            found.append((prefix or "metrics", numeric))
        for k, v in node.items():
            if isinstance(v, dict):
                _walk(v, f"{prefix}.{k}" if prefix else str(k))

    _walk(doc, "")
    return found


def series_on(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    col = pick_col(df, names)
    if col is None or "date" not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.to_datetime(df["date"])
    return s


def check_and_exit() -> None:
    bundle = load_bundle()
    daily = bundle["daily"]
    lines = [f"artifacts={bundle['root']}"]
    ok = daily is not None and not daily.empty and daily["nav"].notna().sum() >= 2
    if daily is None:
        lines.append("dashboard_daily.csv MISSING")
    else:
        lines.append(
            f"daily_rows={len(daily)} nav_points={int(daily['nav'].notna().sum())} "
            f"start={daily['date'].min().date()} end={daily['date'].max().date()}"
        )
        m = calc_metrics(daily.set_index("date")["nav"])
        lines.append(
            f"slice_total={m['tot']:.4f} ann={m['ann']:.4f} sharpe={m['sharpe']:.3f} "
            f"maxdd={m['dd']:.4f}"
        )
    for name, frame in (
        ("legs", bundle["legs"]),
        ("trades", bundle["trades"]),
        ("yearly", bundle["yearly"]),
        ("contrib", bundle["contrib"]),
    ):
        lines.append(f"{name}_rows={0 if frame is None else len(frame)}")
    lines.append(f"topk={sorted(bundle['topk'])}")
    lines.append(f"universe={'yes' if bundle['universe'] is not None else 'no'}")
    text = "\n".join(lines)
    print(text)
    sys.exit(0 if ok else 2)


if os.environ.get("CSRE_CHECK") == "1":
    check_and_exit()


my_font = setup_cn_font()
st.set_page_config(page_title="CSRE 截面残差动量", layout="wide", page_icon="📈")

bundle = load_bundle()
meta_doc = bundle["meta_doc"] if isinstance(bundle["meta_doc"], dict) else {}
params = params_from_meta(meta_doc)
daily_full = bundle["daily"]

st.title("CSRE 截面残差动量")
st.caption(
    "商品期货 · Push final 预计算展示 · "
    f"nf={params.get('nf')} · resid_z={params.get('resid_z')} · "
    f"factor_lb={params.get('factor_lb')} · cone_q={params.get('cone_q')} · "
    f"z={params.get('z')} · hl_max={params.get('hl_max')} · "
    f"vol={params.get('vol')} · fee={params.get('fee')} · ANN={ANN}"
)

if daily_full is None or daily_full.empty or daily_full["nav"].notna().sum() < 2:
    root = bundle["root"]
    st.error(
        "没有读到日频净值。请把 Push final 预计算文件放在看板目录的 artifacts_push 下"
        "（相对 streamlit_app.py，不使用云端路径）。"
    )
    st.code(
        "\n".join([
            str(root / "dashboard" / "dashboard_daily.csv"),
            str(root / "dashboard" / "dashboard_legs.csv"),
            str(root / "dashboard" / "dashboard_trades.csv"),
            str(root / "dashboard" / "dashboard_yearly.csv"),
            str(root / "dashboard" / "dashboard_contrib.csv"),
            str(root / "dashboard" / "dashboard_universe.json"),
            str(root / "dashboard" / "dashboard_data.json"),
            str(root / "topk_compare" / "equity_topk*.csv"),
            str(root / "topk_compare" / "metrics_by_topk.csv"),
        ]),
        language="text",
    )
    st.info(
        "也可以把 csre_streamlit_pack.tgz 或 csre_dashboard_bundle.zip 放在 csre_dashboard/ 后刷新，"
        "页面会只解出 artifacts_push。"
    )
    st.stop()

dmin = daily_full["date"].min().date()
dmax = daily_full["date"].max().date()
available_k = sorted(set(bundle["topk"]) | {PUSH_K})

with st.sidebar:
    st.header("CSRE 截面残差动量")
    st.caption("预计算 · 侧栏只切展示区间和 top_k，不重跑引擎")
    st.caption(f"数据覆盖：{dmin} ~ {dmax}")
    st.caption(f"数据目录：{bundle['root']}")

    k_labels = []
    for k in range(2, 11):
        mark = "Push final" if k == PUSH_K else ("有净值" if k in bundle["topk"] else "无文件")
        k_labels.append(f"{k} · {mark}")
    picked = st.selectbox("top_k", k_labels, index=0)
    top_k = int(picked.split("·", 1)[0].strip())

    preset = st.selectbox(
        "快捷区间",
        ["全样本", "YTD", "2020起", "近3年", "近1年", "2024全年", "2025至今", "2026YTD", "自定义"],
        index=0,
    )
    today = dmax
    if preset == "全样本":
        def_start, def_end = dmin, dmax
    elif preset == "YTD":
        def_start = max(dmin, pd.Timestamp(year=today.year, month=1, day=1).date())
        def_end = dmax
    elif preset == "2020起":
        def_start, def_end = max(dmin, pd.Timestamp("2020-01-01").date()), dmax
    elif preset == "近3年":
        def_start = max(dmin, (pd.Timestamp(today) - pd.DateOffset(years=3)).date())
        def_end = dmax
    elif preset == "近1年":
        def_start = max(dmin, (pd.Timestamp(today) - pd.DateOffset(years=1)).date())
        def_end = dmax
    elif preset == "2024全年":
        def_start = max(dmin, pd.Timestamp("2024-01-01").date())
        def_end = min(dmax, pd.Timestamp("2024-12-31").date())
    elif preset == "2025至今":
        def_start = max(dmin, pd.Timestamp("2025-01-01").date())
        def_end = dmax
    elif preset == "2026YTD":
        def_start = max(dmin, pd.Timestamp("2026-01-01").date())
        def_end = dmax
    else:
        def_start = st.session_state.get("_custom_start", dmin)
        def_end = st.session_state.get("_custom_end", dmax)

    if preset != "自定义":
        st.session_state["ui_start"] = def_start
        st.session_state["ui_end"] = def_end

    c1, c2 = st.columns(2)
    start_d = c1.date_input("开始", min_value=dmin, max_value=dmax, key="ui_start")
    end_d = c2.date_input("结束", min_value=dmin, max_value=dmax, key="ui_end")
    if preset == "自定义":
        st.session_state["_custom_start"] = start_d
        st.session_state["_custom_end"] = end_d
    if start_d > end_d:
        st.error("开始日期不能晚于结束日期")
        st.stop()

    st.divider()
    st.subheader("Push final 口径")
    st.write(f"signal_mode：{params.get('signal_mode')}")
    st.write(f"resid_z / factor_lb：{params.get('resid_z')} / {params.get('factor_lb')}")
    st.write(f"cone_q / z：{params.get('cone_q')} / {params.get('z')}")
    st.write(f"top_k / hl_max：{params.get('top_k')} / {params.get('hl_max')}")
    st.write(f"目标波动 / 手续费：{params.get('vol')} / {params.get('fee')}")
    st.write(f"pos_ewma / nf：{params.get('pos_ewma')} / {params.get('nf')}")
    st.caption("分腿、截面过滤、调仓流水只在 top_k=2（Push final）预计算里。其它 k 只切净值和指标。")

full_trace = top_k == PUSH_K
if full_trace:
    nav_src = daily_full
    trace_note = "Push final · top_k=2 日频全溯源"
else:
    alt = bundle["topk"].get(top_k)
    if alt is None or alt.empty:
        st.warning(f"没有 equity_topk{top_k}.csv，下面仍展示 Push final（k=2）净值。")
        nav_src = daily_full
        full_trace = True
        trace_note = "缺少该 k 的净值文件，回退到 Push final"
    else:
        nav_src = alt
        trace_note = f"top_k={top_k} 仅净值/指标；分腿与截面仍是 k=2"

nav_view = slice_by_date(nav_src, start_d, end_d)
if len(nav_view) < 2 or nav_view["nav"].notna().sum() < 2:
    st.error("所选区间净值点太少，请扩大日期。")
    st.stop()

nav_s = nav_view.set_index("date")["nav"].astype(float)
metrics = calc_metrics(nav_s)
nav_plot = nav_s / float(nav_s.iloc[0])
dd_plot = nav_plot / nav_plot.cummax() - 1.0

daily_view = slice_by_date(daily_full, start_d, end_d) if full_trace else nav_view
legs_view = slice_by_date(bundle["legs"], start_d, end_d) if full_trace else pd.DataFrame()
trades_view = slice_by_date(bundle["trades"], start_d, end_d) if full_trace else pd.DataFrame()

turn = series_on(daily_view, TURN_COLS)
cost = series_on(daily_view, COST_COLS)

st.caption(
    f"{start_d} ~ {end_d} · {trace_note} · 期末净值(归一) {metrics['final']:.4f} · "
    f"年化按 {ANN} 个交易日"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("区间累计收益", f"{metrics['tot']*100:.2f}%")
col2.metric("年化收益", f"{metrics['ann']*100:.2f}%")
col3.metric("夏普 (√242)", f"{metrics['sharpe']:.2f}")
col4.metric("最大回撤", f"{metrics['dd']*100:.2f}%")
col5, col6, col7, col8 = st.columns(4)
col5.metric("卡玛", f"{metrics['calmar']:.2f}")
col6.metric("日胜率", f"{metrics['win']*100:.1f}%")
col7.metric("期末净值", f"{metrics['final']:.4f}")
col8.metric("交易日", f"{metrics['days']} 天")
if turn is not None or cost is not None:
    r1, r2, r3, _ = st.columns(4)
    if turn is not None:
        r1.metric("日均换手", f"{float(turn.mean()):.2%}" if turn.mean() < 5 else f"{float(turn.mean()):.3f}")
    if cost is not None:
        r2.metric("区间成本累计", f"{float(cost.fillna(0).sum()):.4f}")
    if not legs_view.empty:
        sym = pick_col(legs_view, SYMBOL_COLS)
        if sym and "date" in legs_view.columns:
            r3.metric("日均腿数", f"{legs_view.groupby('date').size().mean():.1f}")

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📈 总览曲线",
    "🧭 每日信号溯源",
    "📊 持仓与分腿 PnL",
    "🔁 换手 / 调仓",
    "📅 分年表现",
    "🏅 品种贡献",
    "🧩 策略说明 / top_k",
])


def _line(ax, x, y, color, label, lw=1.8):
    ax.plot(x, y, color=color, lw=lw, label=label)
    ax.grid(True, alpha=0.3)
    ax.legend(prop=my_font, loc="upper left")


with t1:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.1, 1.1]})
    _line(axes[0], nav_plot.index, nav_plot.values, COLOR_NAV, f"净值 ({nav_plot.iloc[-1]:.2f})")
    axes[0].set_title(f"CSRE 净值 · top_k={PUSH_K if full_trace else top_k}", fontproperties=my_font)
    axes[0].set_ylabel("归一净值", fontproperties=my_font)
    _line(axes[1], dd_plot.index, dd_plot.values, COLOR_DD, "回撤")
    axes[1].set_ylabel("回撤", fontproperties=my_font)
    if turn is not None:
        axes[2].plot(turn.index, turn.values, color=COLOR_TURN, lw=1.2, label="换手")
    cone = series_on(daily_view, CONE_COLS)
    if cone is not None:
        axes[2].plot(cone.index, cone.values, color="#ff7f0e", lw=1.2, label="cone/门控")
    if turn is None and cone is None:
        axes[2].text(0.5, 0.5, "该区间没有换手或 cone 列", transform=axes[2].transAxes, ha="center", fontproperties=my_font)
    else:
        axes[2].legend(prop=my_font, loc="upper left")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylabel("换手 / 门控", fontproperties=my_font)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    show_cols = [c for c in daily_view.columns if c in ("date", "nav", "ret") or c not in ("date",)]
    with st.expander("日频表（当前区间）"):
        st.dataframe(daily_view, width="stretch", height=360, hide_index=True)

with t2:
    st.markdown("### 每日信号溯源")
    if not full_trace:
        st.info("截面过滤、腿上的 z/HL 只随 Push final（top_k=2）导出。请把侧栏 top_k 设为 2。")
    elif daily_view.empty:
        st.warning("这个区间没有日频行。")
    else:
        days = list(pd.to_datetime(daily_view["date"]).dt.date.unique())
        chosen = st.selectbox("选择交易日", days, index=len(days) - 1)
        day_row = daily_view[pd.to_datetime(daily_view["date"]).dt.date == chosen]
        st.dataframe(day_row, width="stretch", hide_index=True)
        day_legs = legs_view[pd.to_datetime(legs_view["date"]).dt.date == chosen] if not legs_view.empty else pd.DataFrame()
        st.markdown("#### 当日腿")
        if day_legs.empty:
            st.warning("这一天没有分腿记录。")
        else:
            st.dataframe(day_legs, width="stretch", hide_index=True)
        uni = universe_for_day(bundle["universe"], chosen)
        st.markdown("#### 截面 z / HL / 过滤原因")
        if uni.empty:
            st.info("dashboard_universe.json 里没有这一天，或文件未放置。")
        else:
            reason_col = pick_col(uni, REASON_COLS)
            if reason_col:
                vc = uni[reason_col].fillna("(空)").astype(str).value_counts()
                st.bar_chart(vc)
            st.dataframe(uni, width="stretch", height=420, hide_index=True)

with t3:
    st.markdown("### 持仓与分腿")
    if not full_trace:
        st.info("分腿 PnL 只在 top_k=2。")
    elif legs_view.empty:
        st.warning("没有 dashboard_legs.csv，或当前区间没有腿。")
    else:
        sym_col = pick_col(legs_view, SYMBOL_COLS)
        side_col = pick_col(legs_view, SIDE_COLS)
        f1, f2 = st.columns(2)
        view = legs_view
        if sym_col:
            symbols = ["全部"] + sorted(view[sym_col].dropna().astype(str).unique().tolist())
            picked_sym = f1.selectbox("品种", symbols)
            if picked_sym != "全部":
                view = view[view[sym_col].astype(str) == picked_sym]
        if side_col:
            sides = ["全部"] + sorted(view[side_col].dropna().astype(str).unique().tolist())
            picked_side = f2.selectbox("方向", sides)
            if picked_side != "全部":
                view = view[view[side_col].astype(str) == picked_side]
        st.dataframe(view, width="stretch", height=420, hide_index=True)
        pnl_col = pick_col(view, CONTRIB_COLS + RET_COLS)
        if sym_col and pnl_col:
            agg = view.groupby(view[sym_col].astype(str))[pnl_col].sum().sort_values()
            fig, ax = plt.subplots(figsize=(10, 4.2))
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in agg.values]
            ax.barh(agg.index, agg.values, color=colors)
            ax.set_title("区间分腿收益合计", fontproperties=my_font)
            ax.grid(True, axis="x", alpha=0.3)
            for lab in ax.get_yticklabels():
                lab.set_fontproperties(my_font)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

with t4:
    st.markdown("### 换手与调仓")
    if not full_trace:
        st.info("调仓流水只在 top_k=2。")
    else:
        fig, ax = plt.subplots(figsize=(12, 3.6))
        plotted = False
        if turn is not None:
            ax.plot(turn.index, turn.values, color=COLOR_TURN, lw=1.3, label="换手")
            plotted = True
        if cost is not None:
            ax.plot(cost.index, cost.fillna(0).cumsum().values, color=COLOR_DD, lw=1.3, label="成本累计")
            plotted = True
        if plotted:
            ax.legend(prop=my_font)
            ax.grid(True, alpha=0.3)
            ax.set_title("换手与成本", fontproperties=my_font)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            plt.close(fig)
            st.info("日频表里没有换手或成本列。")
        if trades_view.empty:
            st.warning("没有 dashboard_trades.csv。")
        else:
            st.dataframe(trades_view, width="stretch", height=420, hide_index=True)

with t5:
    st.markdown("### 分年表现")
    yearly = bundle["yearly"]
    if yearly is not None and not yearly.empty:
        st.caption("下表来自 dashboard_yearly.csv（全样本导出，不随侧栏切片重算）。")
        st.dataframe(yearly, width="stretch", hide_index=True)
        ycol = pick_col(yearly, ["year", "年份", "Year"])
        rcol = pick_col(yearly, ["total_return", "ret", "return", "ann_return", "收益", "ann"])
        if ycol and rcol:
            fig, ax = plt.subplots(figsize=(10, 3.6))
            vals = pd.to_numeric(yearly[rcol], errors="coerce")
            ax.bar(yearly[ycol].astype(str), vals, color="#1f77b4")
            ax.set_title("分年收益", fontproperties=my_font)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    built = yearly_from_nav(nav_s)
    st.caption("下表由当前区间净值按年重算（ANN=242）。")
    if built.empty:
        st.info("区间内无法分年。")
    else:
        st.dataframe(
            built.style.format({
                "total_return": "{:.2%}",
                "ann_return": "{:.2%}",
                "sharpe": "{:.2f}",
                "max_dd": "{:.2%}",
            }),
            width="stretch",
            hide_index=True,
        )

with t6:
    st.markdown("### 品种贡献")
    if not full_trace:
        st.info("品种毛贡献表对应 Push final。切换到 top_k=2 查看。")
    else:
        contrib = bundle["contrib"]
        if contrib is None or contrib.empty:
            st.warning("没有 dashboard_contrib.csv。")
        else:
            sym_col = pick_col(contrib, SYMBOL_COLS) or contrib.columns[0]
            val_col = pick_col(contrib, CONTRIB_COLS)
            if val_col is None:
                numeric_cols = [c for c in contrib.columns if pd.api.types.is_numeric_dtype(contrib[c])]
                val_col = numeric_cols[0] if numeric_cols else None
            if val_col is None:
                st.dataframe(contrib, width="stretch", hide_index=True)
            else:
                view = contrib.copy()
                view[val_col] = pd.to_numeric(view[val_col], errors="coerce")
                view = view.dropna(subset=[val_col]).sort_values(val_col, ascending=False)
                pos = view.loc[view[val_col] > 0, val_col].sum()
                neg = view.loc[view[val_col] < 0, val_col].sum()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("正贡献合计", f"{pos:.4f}")
                m2.metric("负贡献合计", f"{neg:.4f}")
                m3.metric("最大贡献", str(view.iloc[0][sym_col]))
                m4.metric("最大拖累", str(view.iloc[-1][sym_col]))
                top = pd.concat([view.head(10), view.tail(10)]).drop_duplicates()
                fig, ax = plt.subplots(figsize=(10, 4.6))
                ordered = top.sort_values(val_col)
                colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ordered[val_col]]
                ax.barh(ordered[sym_col].astype(str), ordered[val_col], color=colors)
                ax.set_title("贡献 / 拖累", fontproperties=my_font)
                ax.grid(True, axis="x", alpha=0.3)
                for lab in ax.get_yticklabels():
                    lab.set_fontproperties(my_font)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(view, width="stretch", height=420, hide_index=True)

with t7:
    st.markdown("### 策略说明")
    st.markdown(
        """
CSRE 是商品期货截面残差动量。本页默认展示 **Push final（top_k=2）** 的日频信号、净值、分腿和截面过滤原因。
侧栏里的 top_k=3..10 只切换已经算好的净值与指标，不会在页面里重跑研究流水线。

流水线口径（导出默认）：`nf=1`，`resid_z=8`，`factor_lb=90`，`cone_q=0.35`，`z=1.0`，
`top_k=2`，`hl_max=12`，目标波动 8%，`pos_ewma=1.0`，手续费 1.5bp，`ANN=242`，`signal_mode=momentum`。
"""
    )
    blocks = numeric_metric_blocks(meta_doc)
    if blocks:
        st.markdown("#### 导出文件里的 IS/OOS 指标")
        for name, nums in blocks:
            st.caption(name)
            st.json(nums)
    else:
        st.caption("dashboard_data.json 里没有识别到夏普/收益字段；原始 JSON 如下。")
        if meta_doc:
            st.json(meta_doc)

    st.markdown("#### top_k 净值叠加")
    if not bundle["topk"]:
        st.info("没有 artifacts_push/topk_compare/equity_topk*.csv。")
    else:
        fig, ax = plt.subplots(figsize=(12, 4.4))
        for k, curve in sorted(bundle["topk"].items()):
            sub = slice_by_date(curve, start_d, end_d)
            if len(sub) < 2:
                continue
            series = sub.set_index("date")["nav"].astype(float)
            series = series / float(series.iloc[0])
            lw = 2.4 if k == PUSH_K else 1.2
            ax.plot(series.index, series.values, lw=lw, label=f"k={k}" + (" Push final" if k == PUSH_K else ""))
        ax.set_title("top_k 归一净值", fontproperties=my_font)
        ax.grid(True, alpha=0.3)
        ax.legend(prop=my_font, ncol=2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    if bundle["topk_metrics"] is not None:
        st.dataframe(bundle["topk_metrics"], width="stretch", hide_index=True)

    st.markdown("#### 本机启动")
    st.code(
        "streamlit run csre_dashboard/streamlit_app.py --server.port 8507",
        language="powershell",
    )
    st.caption("趋势页继续用 8506：streamlit run D:/ALL/趋势策略优化版.py --server.port 8506")
