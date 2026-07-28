# -*- coding: utf-8 -*-
"""回测净值图通用绘图工具（云端包自带，避免依赖 D:\\ALL）。"""
from __future__ import annotations

import pandas as pd


def add_fee_line(
    ax,
    nav_df: pd.DataFrame,
    *,
    monotonic: bool = True,
    lw: float = 1.2,
    alpha: float = 0.85,
    color: str = "#d62728",
    label_prefix: str = "手续费累计",
) -> bool:
    if "daily_cost" not in nav_df.columns:
        return False
    nav = nav_df["nav"]
    base = float(nav.iloc[0]) or 1.0
    fee_nav = base + nav_df["daily_cost"].fillna(0.0).cumsum()
    fee_plot = fee_nav / base if monotonic else fee_nav
    fee_simple = float(nav_df["daily_cost"].fillna(0.0).sum())
    label = f"{label_prefix} ({fee_plot.iloc[-1]:.2f}，付费 {fee_simple:.2f})"
    ax.plot(
        fee_plot.index, fee_plot, color=color, ls="--", lw=lw, alpha=alpha, label=label,
    )
    return True
