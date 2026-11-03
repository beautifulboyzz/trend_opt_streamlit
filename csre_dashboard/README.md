# CSRE Push final · 日频看板

商品期货截面残差动量。中文界面对齐本仓库趋势页：侧栏参数、指标卡、多 Tab。只读本地预计算文件，不重跑引擎。

默认展示 **Push final（top_k=2）** 的日频信号、净值、分腿和截面过滤原因。侧栏可切已经算好的 `top_k=2..10` 净值；完整溯源只有 k=2。

## 启动

在仓库根目录：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\streamlit run csre_dashboard\streamlit_app.py --server.port 8507
```

浏览器打开 http://localhost:8507

趋势页占用 8506，不要和本页抢端口：

```powershell
streamlit run D:\ALL\趋势策略优化版.py --server.port 8506
```

## 数据

全部相对 `csre_dashboard/streamlit_app.py`：

| 路径 | 用途 |
|------|------|
| `artifacts_push/dashboard/dashboard_daily.csv` | 日频净值、门控、持仓摘要 |
| `artifacts_push/dashboard/dashboard_legs.csv` | 每日腿、z、HL、次日收益 |
| `artifacts_push/dashboard/dashboard_trades.csv` | 调仓事件 |
| `artifacts_push/dashboard/dashboard_yearly.csv` | 分年指标 |
| `artifacts_push/dashboard/dashboard_contrib.csv` | 品种毛贡献 |
| `artifacts_push/dashboard/dashboard_universe.json` | 截面 z / HL / 过滤原因 |
| `artifacts_push/dashboard/dashboard_data.json` | 参数与 IS/OOS 指标 |
| `artifacts_push/topk_compare/equity_topk*.csv` | 各 top_k 净值 |
| `artifacts_push/topk_compare/metrics_by_topk.csv` | 各 top_k 指标 |

可选：把 `csre_streamlit_pack.tgz` 或非空的 `csre_dashboard_bundle.zip` 放在 `csre_dashboard/` 后刷新，页面只解出其中的 `artifacts_push`。

环境变量 `CSRE_ARTIFACTS` 可指向另一份 `artifacts_push` 目录。

## 口径

`nf=1, resid_z=8, factor_lb=90, cone_q=0.35, z=1.0, top_k=2, hl_max=12, vol=8%, pos_ewma=1.0, fee=1.5bps, ANN=242, signal_mode=momentum`
