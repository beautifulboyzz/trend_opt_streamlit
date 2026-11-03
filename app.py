# -*- coding: utf-8 -*-
"""
趋势策略优化版 · Streamlit Cloud 展示页

云端只读 results/ 预计算结果；侧栏可选展示区间（切片重算指标，不重跑引擎）。
UI 对齐本地「趋势策略优化版.py」：指标卡 + 九个 Tab（白盒 Tab 为说明页）。

本地：
  streamlit run app.py --server.port 8506
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from report_plot_utils import add_fee_line

APP_DIR = Path(__file__).resolve().parent
RESULTS = APP_DIR / "results"
FONT_CANDIDATES = [
    APP_DIR / "fonts" / "SimHei.ttf",
    APP_DIR / "SimHei.ttf",
    APP_DIR / "Ubuntu_18.04_SimHei.ttf",
]

st.set_page_config(page_title="趋势策略优化版", layout="wide", page_icon="📈")

COLOR_NAV = "#2ca02c"
COLOR_BENCH = "#1f77b4"
COLOR_EXCESS = "#ff7f0e"


def setup_cn_font():
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                fm.fontManager.addfont(str(path))
            except Exception:
                pass
            prop = fm.FontProperties(fname=str(path))
            name = prop.get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name, "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    prop = fm.FontProperties(family=["DejaVu Sans"])
    plt.rcParams["axes.unicode_minus"] = False
    return prop


my_font = setup_cn_font()


@st.cache_data
def load_bundle():
    meta_p, nav_p = RESULTS / "meta.json", RESULTS / "nav.csv"
    if not meta_p.exists() or not nav_p.exists():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    nav = pd.read_csv(nav_p, parse_dates=["date"]).set_index("date").sort_index()
    daily = (
        pd.read_csv(RESULTS / "daily.csv", parse_dates=["date"]).sort_values("date")
        if (RESULTS / "daily.csv").exists() else None
    )
    weekly = pd.read_csv(RESULTS / "weekly.csv") if (RESULTS / "weekly.csv").exists() else None
    yearly = pd.read_csv(RESULTS / "yearly.csv") if (RESULTS / "yearly.csv").exists() else None
    contrib = pd.read_csv(RESULTS / "contrib.csv") if (RESULTS / "contrib.csv").exists() else None
    waves = pd.read_csv(RESULTS / "waves.csv") if (RESULTS / "waves.csv").exists() else None
    log = (RESULTS / "log.txt").read_text(encoding="utf-8") if (RESULTS / "log.txt").exists() else ""
    ind_index = []
    ind_path = RESULTS / "independent" / "index.json"
    if ind_path.exists():
        ind_index = json.loads(ind_path.read_text(encoding="utf-8"))
    return {
        "meta": meta, "nav": nav, "daily": daily, "weekly": weekly,
        "yearly": yearly, "contrib": contrib, "waves": waves, "log": log,
        "ind_index": ind_index,
    }


@st.cache_data
def load_independent(tag: str):
    """加载某年独立回测包（与本地「开始日=该年1月1日」对齐）。"""
    d = RESULTS / "independent" / str(tag)
    if not (d / "nav.csv").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
    nav = pd.read_csv(d / "nav.csv", parse_dates=["date"]).set_index("date").sort_index()
    daily = (
        pd.read_csv(d / "daily.csv", parse_dates=["date"]).sort_values("date")
        if (d / "daily.csv").exists() else None
    )
    waves = pd.read_csv(d / "waves.csv") if (d / "waves.csv").exists() else None
    contrib = pd.read_csv(d / "contrib.csv") if (d / "contrib.csv").exists() else None
    log = (d / "log.txt").read_text(encoding="utf-8") if (d / "log.txt").exists() else ""
    return {"meta": meta, "nav": nav, "daily": daily, "waves": waves, "contrib": contrib, "log": log}


def match_independent(start_d, end_d, ind_index: list) -> str | None:
    """若所选区间恰好是某年独立回测窗（1/1~年末或数据末日），返回 tag。"""
    if not ind_index:
        return None
    s, e = pd.Timestamp(start_d).normalize(), pd.Timestamp(end_d).normalize()
    for row in ind_index:
        rs, re_ = pd.Timestamp(row["start"]).normalize(), pd.Timestamp(row["end"]).normalize()
        # 允许结束日落在该独立窗末日（YTD / 全年）
        if s == rs and e == re_:
            return str(row["tag"])
        # YTD：开始=该年1/1，结束=独立窗末日
        if s.year == int(row["tag"]) and s.month == 1 and s.day == 1 and e == re_:
            return str(row["tag"])
    return None


def calc_metrics(nav: pd.Series) -> dict:
    nav = nav.dropna().astype(float)
    if len(nav) < 2:
        return {
            "tot": 0.0, "ann": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0,
            "win": 0.0, "final": 1.0, "days": 0,
        }
    nav = nav / float(nav.iloc[0])
    tot = float(nav.iloc[-1] - 1)
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    ann = (1 + tot) ** (365.25 / days) - 1
    dr = nav.pct_change().dropna()
    sh = float((dr.mean() / dr.std()) * np.sqrt(252)) if len(dr) and dr.std() else 0.0
    dd = float((nav / nav.cummax() - 1).min())
    cal = abs(ann / dd) if dd else 0.0
    win = float((dr > 0).mean()) if len(dr) else 0.0
    return {
        "tot": tot, "ann": ann, "sharpe": sh, "dd": dd, "calmar": cal,
        "win": win, "final": float(nav.iloc[-1]), "days": int(len(nav)),
    }


def filter_log_by_dates(text: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    if not text:
        return ""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    out: list[str] = []
    keep_day = False
    for ln in text.splitlines():
        if ln.startswith("第") and "周：" in ln:
            keep_day = False
            try:
                part = ln.split("：", 1)[1]
                span = part.split("收益")[0].strip()
                a_s, b_s = [x.strip() for x in span.split("~")]
                a_d = pd.Timestamp(a_s.split()[0]).normalize()
                b_d = pd.Timestamp(b_s.split()[0]).normalize()
                if not (b_d < start or a_d > end):
                    out.append(ln)
                    out.append("")
            except Exception:
                pass
            continue
        if ln.startswith("[") and "]" in ln[:14]:
            try:
                d = pd.Timestamp(ln[1:ln.index("]")]).normalize()
                keep_day = start <= d <= end
            except Exception:
                keep_day = False
            if keep_day:
                out.append(ln)
            continue
        if ln.startswith("="):
            if out and not out[-1].startswith("="):
                out.append(ln)
            keep_day = False
            continue
        if keep_day:
            out.append(ln)
    return "\n".join(out)


def build_log_from_daily(tdf: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    if tdf is None or tdf.empty:
        return ""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    sub = tdf[(tdf["date"] >= start) & (tdf["date"] <= end)].copy()
    if sub.empty:
        return "（所选区间无日明细）"
    sub["date"] = pd.to_datetime(sub["date"])
    logs: list[str] = []
    iso = sub["date"].dt.isocalendar()
    wi = 0
    for (y, w), g in sub.groupby([iso.year, iso.week], sort=True):
        wi += 1
        g = g.sort_values("date")
        week_ret = float(np.prod(1.0 + g["ret"].fillna(0.0).astype(float)) - 1.0)
        end_nav = g["nav"].iloc[-1]
        nav_s = f"{float(end_nav):.4f}" if pd.notna(end_nav) else "—"
        logs.append(
            f"第{wi}周：{g['date'].iloc[0].date()} ~ {g['date'].iloc[-1].date()} "
            f"收益: {week_ret*100:+.2f}% | 净值: {nav_s}"
        )
        logs.append("")
        for _, row in g.iterrows():
            logs.append(f"[{row['date'].date()}]")
            logs.append(f"📊 当天总收益率：{float(row['ret'])*100:+.2f}%")
            longs = str(row.get("long_hold") or "")
            shorts = str(row.get("short_hold") or "")
            if longs or shorts:
                logs.append("📈 当天持仓明细：")
                for item in longs.split("|"):
                    if item and ":" in item:
                        a, wt = item.split(":", 1)
                        try:
                            logs.append(f"  {a} (做多) ➔ 仓位: {float(wt):.1%}")
                        except Exception:
                            logs.append(f"  {a} (做多) ➔ 仓位: {wt}")
                for item in shorts.split("|"):
                    if item and ":" in item:
                        a, wt = item.split(":", 1)
                        try:
                            logs.append(f"  {a} (做空) ➔ 仓位: {float(wt):.1%}")
                        except Exception:
                            logs.append(f"  {a} (做空) ➔ 仓位: {wt}")
            else:
                logs.append("📈 当天持仓：空仓")
            stops = str(row.get("stops") or "").strip()
            if stops and stops.lower() not in ("nan", "none", ""):
                logs.append(f"⚠️ 当天止损：{stops} (进入隔离小黑屋🚫)")
            nxt_l = str(row.get("next_long") or "")
            nxt_s = str(row.get("next_short") or "")
            if nxt_l or nxt_s:
                logs.append("🔮 隔夜持仓：")
                for item in nxt_l.split("|"):
                    if item and ":" in item:
                        a, wt = item.split(":", 1)
                        try:
                            logs.append(f"  {a} (做多)：(仓位：{float(wt):.0%})")
                        except Exception:
                            logs.append(f"  {a} (做多)：(仓位：{wt})")
                for item in nxt_s.split("|"):
                    if item and ":" in item:
                        a, wt = item.split(":", 1)
                        try:
                            logs.append(f"  {a} (做空)：(仓位：{float(wt):.0%})")
                        except Exception:
                            logs.append(f"  {a} (做空)：(仓位：{wt})")
            else:
                logs.append("🔮 隔夜持仓：空仓")
            logs.append("")
        logs.append("=" * 60)
    return "\n".join(logs)


def cloud_only_tab(title: str):
    st.info(
        f"**{title}** 依赖全市场实时因子面板，云端预计算包不含此项。\n\n"
        "本地完整白盒请运行：\n"
        "`streamlit run D:/ALL/趋势策略优化版.py --server.port 8506`"
    )


bundle = load_bundle()
st.title("趋势策略优化版")
st.caption(
    "Enhanced Long-Short Momentum · 预计算展示 · "
    "Regime关空 + 空头对称滤网 + 全市场3×ATR熔断 + 涨跌停"
)

if bundle is None:
    st.error("未找到 results/。请本地运行 `python export_cloud_results.py` 后推送仓库。")
    st.stop()

meta = bundle["meta"]
nav_full = bundle["nav"]
dmin, dmax = nav_full.index.min().date(), nav_full.index.max().date()

with st.sidebar:
    st.header("趋势策略优化版")
    st.caption("预计算 · 整年/YTD 用独立回测（对齐本地）；其它区间为全样本切片")
    st.caption(f"数据覆盖：{dmin} ~ {dmax}")
    if bundle.get("ind_index"):
        st.success(f"已含独立回测年：{', '.join(x['tag'] for x in bundle['ind_index'])}")
    else:
        st.warning("尚未导出 independent/；YTD 数字会与本地独立回测不一致")

    preset = st.selectbox(
        "快捷区间",
        ["全样本", "YTD", "2020起", "近3年", "近1年", "2024全年", "2025至今", "2026YTD", "自定义"],
        index=0,
    )
    today = dmax
    if preset == "全样本":
        def_start, def_end = dmin, dmax
    elif preset == "YTD":
        def_start = max(dmin, pd.Timestamp(today.replace(month=1, day=1)).date())
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
    st.subheader("产品口径（导出默认）")
    st.write("涨跌停：开")
    st.write("全市场3×ATR熔断：常开")
    st.write(f"手续费：{meta.get('commission_bp', 1.5)}bp")
    st.write(f"单向持仓 / 板块上限：{meta.get('hold_num', 4)} / {meta.get('max_per_sector', 2)}")
    st.write("Regime关空 + 空头对称滤网：开")
    st.write("旧版4%止损 + Adv高点回撤：开")
    lb = meta.get("lb_stats") or {}
    st.caption(
        f"导出 {meta.get('exported_at', '')} · "
        f"全局预警 {meta.get('global_crash_events', 0)} 次 · "
        f"lb_buy={lb.get('buy_blocked', 0)} / lb_sell={lb.get('sell_blocked', 0)}"
    )
    if not any(p.exists() for p in FONT_CANDIDATES):
        st.warning("未找到 fonts/SimHei.ttf，图中中文可能方框。")

# 切片：优先命中「独立回测年」→ 与本地改开始日期对齐；否则全样本切片
ind_tag = match_independent(start_d, end_d, bundle.get("ind_index") or [])
mode_label = "连续净值切片（持仓承接上年）"
active = None
if ind_tag:
    active = load_independent(ind_tag)
    if active is not None:
        mode_label = f"独立回测（{ind_tag}年 · 对齐本地开始日）"
        nav_full_use = active["nav"]
        # 独立窗已是完整区间，直接用
        nav_slice = nav_full_use.copy()
        daily_src = active.get("daily")
        waves_src = active.get("waves")
        contrib_src = active.get("contrib")
        log_src = active.get("log") or ""
    else:
        ind_tag = None

if not ind_tag:
    mask = (nav_full.index.date >= start_d) & (nav_full.index.date <= end_d)
    nav_slice = nav_full.loc[mask].copy()
    daily_src = bundle["daily"]
    waves_src = bundle["waves"]
    contrib_src = bundle["contrib"]
    log_src = bundle["log"]

if len(nav_slice) < 5:
    st.error("所选区间交易日太少，请扩大日期范围。")
    st.stop()

nav_s = nav_slice["nav"].astype(float)
# 区间归一指标
m = calc_metrics(nav_s)
# 展示用绝对净值（保持导出累计路径，但从图起点归一更直观）
nav_plot = nav_s / float(nav_s.iloc[0])
nav_df_plot = pd.DataFrame({"nav": nav_plot}, index=nav_plot.index)
if "daily_cost" in nav_slice.columns:
    # 手续费线：对该切片重新累加
    nav_df_plot["daily_cost"] = nav_slice["daily_cost"].fillna(0.0).values

daily = daily_src
if daily is not None and not daily.empty:
    if ind_tag:
        dsub = daily.copy()
        dsub["date"] = pd.to_datetime(dsub["date"])
    else:
        dsub = daily[(daily["date"].dt.date >= start_d) & (daily["date"].dt.date <= end_d)].copy()
else:
    dsub = pd.DataFrame()

avg_n = float((dsub["n_long"] + dsub["n_short"]).mean()) if not dsub.empty else meta.get("avg_daily_holdings", 0)
avg_long = float(dsub["long_exposure"].mean()) if not dsub.empty else meta.get("avg_long_exposure", 0)
avg_short = float(dsub["short_exposure"].mean()) if not dsub.empty else meta.get("avg_short_exposure", 0)

waves = waves_src
if waves is not None and not waves.empty and "Start_Date" in waves.columns:
    wsub = waves.copy()
    wsub["Start_Date"] = pd.to_datetime(wsub["Start_Date"])
    wsub["End_Date"] = pd.to_datetime(wsub["End_Date"])
    if not ind_tag:
        wsub = wsub[(wsub["End_Date"].dt.date >= start_d) & (wsub["Start_Date"].dt.date <= end_d)]
    avg_hold = float(wsub["Hold_Days"].mean()) if len(wsub) and "Hold_Days" in wsub.columns else 0.0
else:
    wsub = pd.DataFrame()
    avg_hold = float((active or {}).get("meta", {}).get("avg_hold_days") or meta.get("avg_hold_days") or 0)

st.caption(
    f"{start_d} ~ {end_d} · 期末净值(归一) {m['final']:.4f} · "
    f"**{mode_label}** · 涨跌停开 · 全市场3×ATR熔断常开 · 手续费 {meta.get('commission_bp', 1.5)}bp"
)
if not ind_tag and start_d.year == end_d.year and start_d.month == 1 and start_d.day == 1:
    st.warning(
        "当前为「全样本连续净值」切片，持仓承接上年，"
        "与本地把开始日改成该年1月1日的独立回测会有差异。"
        "请先本地重跑 `export_cloud_results.py` 生成 independent/ 后推送，或选已导出的整年/YTD。"
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("多空独立累计总净收益", f"{m['tot']*100:.2f}%")
col2.metric("年化复合增长率(CAGR)", f"{m['ann']*100:.2f}%")
col3.metric("平均每天持仓个数", f"{avg_n:.1f} 个")
col4.metric("动态全时最大回撤", f"{m['dd']*100:.2f}%")
col5, col6, col7, col8 = st.columns(4)
col5.metric("夏普表现比率 (Sharpe)", f"{m['sharpe']:.2f}")
col6.metric("卡玛风险收益比 (Calmar)", f"{m['calmar']:.2f}")
col7.metric("每日胜率偏置", f"{m['win']*100:.1f}%")
col8.metric("运行结算总天数", f"{m['days']} 天")
r1, r2, r3, _ = st.columns(4)
r1.metric("平均多头仓位", f"{avg_long:.1%}")
r2.metric("平均空头仓位", f"{avg_short:.1%}")
r3.metric("平均持仓天数", f"{avg_hold:.1f} 个交易日")

t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "📈 资产组合净值曲线",
    "📊 个体盈亏贡献分布",
    "📝 自动化穿透日志",
    "🔬 每日全市场因子透视",
    "🧮 独立品种数学解构",
    "🏅 时序动量截面排序",
    "⏱️ 持仓周期精细画像",
    "🧭 多空收益拆解",
    "🧩 策略框架诊断",
])

with t1:
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(nav_plot.index, nav_plot, color=COLOR_NAV, lw=2, label=f"净值 ({nav_plot.iloc[-1]:.2f})")
    add_fee_line(ax1, nav_df_plot)

    excess = None
    if "bench" in nav_slice.columns and nav_slice["bench"].notna().any():
        bench = nav_slice["bench"].astype(float)
        bench_n = bench / float(bench.iloc[0])
        ax1.plot(bench_n.index, bench_n, color=COLOR_BENCH, lw=1.5, alpha=0.85,
                 label=f"基准 ({meta.get('bench_name') or '基准'})")
        excess = nav_plot - bench_n.reindex(nav_plot.index).ffill()

    ax2 = ax1.twinx()
    if excess is not None:
        ax2.plot(excess.index, excess, color=COLOR_EXCESS, lw=1.5, ls="--", label="超额")
        ax2.set_ylabel("超额", fontproperties=my_font, color="#666666")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if excess is not None else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, prop=my_font, loc="upper left")
    ax1.set_title("趋势策略_多空日频", fontproperties=my_font, fontsize=14)
    ax1.set_ylabel("累计净值", fontproperties=my_font)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    ydf = bundle["yearly"]
    if ydf is not None and not ydf.empty:
        st.markdown("#### 分年收益（全样本切片，非独立回测）")
        st.dataframe(ydf, use_container_width=True, hide_index=True)

with t2:
    st.markdown("### 📊 品种贡献驾驶舱")
    st.caption("先看谁贡献最大、谁拖累最大。" + ("当前为该年独立回测贡献。" if ind_tag else "云端默认全样本贡献；命中独立年后用该年贡献。"))
    cdf = contrib_src
    if cdf is not None and not cdf.empty:
        view = cdf.copy()
        tot_ret = m["tot"] if ind_tag else float(meta.get("metrics", {}).get("tot_pct", 0)) / 100.0
        view["Contribution_pct"] = view["Contribution"] / tot_ret if tot_ret else 0
        view["方向"] = np.where(view["Contribution"] >= 0, "贡献", "拖累")
        pos_sum = view.loc[view["Contribution"] > 0, "Contribution"].sum()
        neg_sum = view.loc[view["Contribution"] < 0, "Contribution"].sum()
        c2m1, c2m2, c2m3, c2m4 = st.columns(4)
        c2m1.metric("正贡献合计", f"{pos_sum:+.2%}")
        c2m2.metric("负贡献合计", f"{neg_sum:+.2%}")
        c2m3.metric("最大贡献品种", str(view.iloc[0]["Asset"]))
        c2m4.metric("最大拖累品种", str(view.sort_values("Contribution").iloc[0]["Asset"]))

        top = view.head(15)
        fig_c, ax = plt.subplots(figsize=(10, 4.5))
        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in top["Contribution"]]
        ax.barh(top["Asset"][::-1], top["Contribution"][::-1], color=colors[::-1])
        ax.set_title("品种贡献 Top15", fontproperties=my_font)
        ax.grid(True, axis="x", alpha=0.3)
        for lab in ax.get_yticklabels():
            lab.set_fontproperties(my_font)
        fig_c.tight_layout()
        st.pyplot(fig_c)
        st.dataframe(
            view.style.format({"Contribution": "{:+.4%}", "Contribution_pct": "{:.1%}"}),
            use_container_width=True, height=420,
        )
    else:
        st.warning("无贡献数据")

with t3:
    st.markdown("### 📝 自动化穿透日志工作台")
    log_kw = st.text_input("搜索关键词（品种/止损/日期/做多/做空）", value="")
    quick = st.selectbox("快捷过滤", ["全部日志", "只看周汇总", "只看止损", "只看持仓", "只看亏损日"])
    # 优先用 daily 重建（含隔夜仓），否则裁切导出 log
    rebuilt = build_log_from_daily(dsub if not dsub.empty else daily, start_d, end_d)
    raw = rebuilt if rebuilt else filter_log_by_dates(log_src, start_d, end_d)
    lines = raw.splitlines()
    if quick == "只看周汇总":
        lines = [ln for ln in lines if ln.startswith("第") and "周：" in ln]
    elif quick == "只看止损":
        lines = [ln for ln in lines if "止损" in ln or "全局预警" in ln]
    elif quick == "只看持仓":
        lines = [ln for ln in lines if "持仓" in ln or ln.startswith("  ")]
    elif quick == "只看亏损日":
        keep = set()
        cur = None
        for ln in raw.splitlines():
            if ln.startswith("[") and "]" in ln[:14]:
                cur = ln
            if "当天总收益率：-" in ln and cur:
                keep.add(cur)
        out = []
        cur_keep = False
        for ln in raw.splitlines():
            if ln.startswith("[") and "]" in ln[:14]:
                cur_keep = ln in keep
            if cur_keep or (ln.startswith("第") and "周：" in ln) or ln.startswith("="):
                out.append(ln)
        lines = out
    if log_kw.strip():
        lines = [ln for ln in lines if log_kw.strip() in ln]
    st.text_area("日志", "\n".join(lines), height=520)
    st.download_button(
        "下载区间日志", "\n".join(lines),
        file_name=f"trend_opt_log_{start_d}_{end_d}.txt",
        mime="text/plain",
    )

with t4:
    cloud_only_tab("每日全市场因子透视")
with t5:
    cloud_only_tab("独立品种数学解构")
with t6:
    cloud_only_tab("时序动量截面排序")

with t7:
    st.markdown("### ⏱️ 持仓周期精细画像")
    if wsub is not None and not wsub.empty:
        st.metric("平均持仓天数", f"{avg_hold:.1f} 个交易日")
        show = wsub.copy()
        if "Wave_Return" in show.columns:
            show["区间损益"] = show["Wave_Return"]
        cols = [c for c in ["asset", "direction", "Start_Date", "End_Date", "Hold_Days", "区间损益", "Wave_Return"] if c in show.columns]
        st.dataframe(show[cols].sort_values("Hold_Days", ascending=False), use_container_width=True, height=420)
        fig_w, ax = plt.subplots(figsize=(10, 3.8))
        ax.hist(show["Hold_Days"].dropna(), bins=20, color="#1f77b4", edgecolor="white")
        ax.set_title("持仓天数分布", fontproperties=my_font)
        ax.set_xlabel("交易日", fontproperties=my_font)
        fig_w.tight_layout()
        st.pyplot(fig_w)
    else:
        st.info("无波段数据")

with t8:
    st.markdown("### 🧭 多空收益拆解驾驶舱")
    if not dsub.empty:
        e1, e2, e3 = st.columns(3)
        e1.metric("区间均多仓", f"{avg_long:.1%}")
        e2.metric("区间均空仓", f"{avg_short:.1%}")
        e3.metric("止损触发日", f"{int((dsub['stop_count'] > 0).sum())}")
        fig8, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(dsub["date"], dsub["long_exposure"], color="#2ca02c", lw=1.2, label="多头敞口")
        ax.plot(dsub["date"], dsub["short_exposure"], color="#d62728", lw=1.2, label="空头敞口")
        ax.set_title("多空敞口时序", fontproperties=my_font)
        ax.legend(prop=my_font)
        ax.grid(True, alpha=0.3)
        fig8.tight_layout()
        st.pyplot(fig8)
        if "gap_long_pnl" in dsub.columns:
            st.caption(
                f"跳空多头PnL累计 {dsub['gap_long_pnl'].sum()*100:+.2f}% · "
                f"跳空空头PnL累计 {dsub['gap_short_pnl'].sum()*100:+.2f}%"
            )
        st.dataframe(
            dsub[["date", "ret", "long_exposure", "short_exposure", "n_long", "n_short", "stop_count", "stops"]],
            use_container_width=True, height=360,
        )
    else:
        st.info("无日明细")

with t9:
    st.markdown("### 🧩 策略框架诊断总控台")
    if not dsub.empty:
        health = pd.DataFrame([
            {
                "模块": "净值路径",
                "状态": "✅ 正常" if m["dd"] > -0.35 else "⚠️ 回撤偏大",
                "核心指标": f"夏普 {m['sharpe']:.2f} · 回撤 {m['dd']*100:.1f}%",
            },
            {
                "模块": "止损/熔断",
                "状态": "✅ 正常" if dsub["stop_count"].mean() < 1.5 else "⚠️ 止损偏多",
                "核心指标": f"日均止损 {dsub['stop_count'].mean():.2f} · 全局预警(全样本) {meta.get('global_crash_events', 0)}",
            },
            {
                "模块": "涨跌停",
                "状态": "✅ 已启用",
                "核心指标": f"lb_buy={lb.get('buy_blocked', 0)} / lb_sell={lb.get('sell_blocked', 0)}",
            },
            {
                "模块": "仓位",
                "状态": "✅ 正常",
                "核心指标": f"均多 {avg_long:.1%} · 均空 {avg_short:.1%} · 日均持仓数 {avg_n:.1f}",
            },
        ])
        st.dataframe(health, use_container_width=True, hide_index=True)
        fig9, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
        axes[0].plot(dsub["date"], dsub["nav"], color="#1f77b4", lw=1.6)
        axes[0].set_title("净值", fontproperties=my_font)
        axes[0].grid(True, alpha=0.3)
        axes[1].bar(dsub["date"], dsub["stop_count"], color="#d62728", width=1.0)
        axes[1].set_title("每日止损/熔断数量", fontproperties=my_font)
        axes[1].grid(True, alpha=0.3)
        fig9.tight_layout()
        st.pyplot(fig9)
    st.markdown("#### 口径说明")
    st.markdown(
        f"""
| 项目 | 取值 |
|---|---|
| 产品 | 趋势策略优化版 |
| 全市场3×ATR熔断 | 常开（低点≤昨收−3ATR 进小黑屋） |
| Regime关空 / 空头对称滤网 | 开 |
| 涨跌停 | 开 |
| 手续费 | {meta.get('commission_bp', 1.5)}bp |
| 导出区间 | {meta.get('start')} ~ {meta.get('end')} |
| 导出时间 | {meta.get('exported_at', '')} |
"""
    )
