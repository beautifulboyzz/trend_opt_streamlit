# Streamlit Cloud 部署包：趋势策略优化版（预计算展示）

**在线地址：** [https://trendoptapp-wc8karhp8e9bvmttm3tud7.streamlit.app/](https://trendoptapp-wc8karhp8e9bvmttm3tud7.streamlit.app/)

本地 HTML 入口：`index.html`（可双击打开，含跳转按钮与内嵌页）

云端只读 `results/`，不推全市场 CSV。UI 布局对齐本地 `趋势策略优化版.py`（指标卡 + 九个 Tab；因子白盒 Tab 在云端为只读说明）。

## 本地先导出结果

```bash
# 需本机数据与引擎：D:\数据集文件\期货全品种日线数据、D:\ALL\趋势策略优化版.py
python D:\ALL\trend_opt_streamlit\export_cloud_results.py
```

导出后应有：

```
trend_opt_streamlit/
  app.py
  requirements.txt
  fonts/SimHei.ttf
  results/
    meta.json
    nav.csv
    daily.csv
    weekly.csv
    yearly.csv
    contrib.csv
    waves.csv
    log.txt
```

本地预览（8506，8507 留给 CSRE 日频看板）：

```bash
cd D:\ALL\trend_opt_streamlit
streamlit run app.py --server.port 8506
```

CSRE 截面残差动量（Push final）在同一仓库的 `csre_dashboard/`，端口 8507：

```powershell
streamlit run csre_dashboard/streamlit_app.py --server.port 8507
```

说明与数据路径见 `csre_dashboard/README.md`。

## 推到 GitHub → Streamlit Cloud

1. 仓库根目录上传本文件夹全部内容（含 `fonts/SimHei.ttf` ≈9MB）
2. Streamlit Cloud → Main file = `app.py`
3. 侧栏可改展示区间（在预计算结果上切片，不重跑引擎）

### 注意

- 更新回测：本地重跑 `export_cloud_results.py` → 覆盖 `results/` → Commit
- 改完代码后 Cloud 需 Reboot / 等自动 redeploy
- 全市场因子透视 / 单品种白盒 / 动量截面排序 依赖实时面板，云端展示说明页；完整白盒请本地 `streamlit run D:\ALL\趋势策略优化版.py`

## 口径（对齐优化版默认）

- Regime 关空 + 空头对称滤网
- 全市场 3×ATR 熔断常开（无持仓也进小黑屋）
- 旧版 4% 止损 + Adv 高点回撤
- 涨跌停开 · 手续费 1.5bp · 流动性原版
- 区间默认 2020-01-01 ~ 数据末日
