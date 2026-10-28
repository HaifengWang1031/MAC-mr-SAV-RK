# 历史受迫方腔时间收敛结果

[结果 Notebook](results.ipynb) · [中文论文描述](description.tex) · [收敛结果专题](../../../convergence-results.md)

## 固定输入

- 批次：`20260922T023439-35a9711f`，位于 `runs/forced_ns_convergence/batches/`。
- 报告：`20260922T032418-3cd92bd4`，位于 `reports/forced_ns_convergence/`。
- 数值表：该报告的 `tables/errors.csv`；16 个被测运行、2 个共同参考，48 个观测误差行。
- 128² 单位无滑移方域，nu=0.01、gamma=1、零初值，SDIRK2 与 SDIRK2-mr-ccSAV。
- **历史运行的 `config.json` 记录外力 `[0, amplitude*sin(x)]`，amplitude=1。不是当前生产配置的 `m=2` 外力。** 不可用当前配置重跑后冒充该批次，亦不可套用当前外力的定常窗口说明。

Notebook 固定报告与误差表摘要，校验捕获批次、所有被测/参考记录的配置及 HDF5 摘要、同一历史来源、物理参数、完成状态和观测节点。只重绘保存误差，不重算场误差；不要求历史来源冒充当前源码身份。

## 执行

从项目根目录执行：

```sh
uv sync --locked --group notebook
uv run --locked --group notebook python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path = Path('experiments/forced_ns_convergence/studies/forced_time_convergence/results.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=120, kernel_name='python3').execute()
nbformat.write(notebook, path)
PY
```

迁移同一研究的数据时只修改带 `parameters` 标签的单元格中的项目根目录、显式报告目录及其已核验摘要。若更换研究批次，必须重新核对物理身份并相应重写叙述，不能只换摘要继续套用本文。缺失、损坏、来源不符或参考检查未通过时拒绝展示，不搜索最新记录、不补算。

## 结果与限制

已保存全步长图、k>=6 放大图、完整 L2/H1 半范数误差与观察阶表、原参考敏感性检查。T=2 的 L2 观察阶接近 2，H1 半范数逐步趋近 2；两方法在该批次中接近，不能推广为 SAV 在其他问题上不影响误差或稳定性。

参考 k=12/13 的六项保存检查通过 5% 阈值。这是细化对敏感性，不是严格参考误差界。中文片段使用支持中文的主文件并加载 `amsmath, amssymb` 后，以 `\input{.../description.tex}` 引入。
