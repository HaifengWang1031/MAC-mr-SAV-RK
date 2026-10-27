# 一阶格式的示例与验证

目的：固定空间离散，检查两种 Euler 格式的一阶速度精度、辅助变量误差及完整阶段残差。
默认配置采用 MAC 12x10（直接/Tensor）、Legendre size=6、Fourier 24x18、Channel 12x10（直接/FFT），
T=0.4、amplitude=0.1、80/160/320 步。SAV 分别测试 gamma=1/1000；普通格式每组只运行一次。

Fourier 使用连续非线性制造解，Channel 使用独立 NumPy 离散制造解；
MAC/Legendre 用固定算子构造半离散制造外力，使解析时间轨迹精确，这些成员检验时间格式而非空间算子。
NPZ 保存最终场、参照、时间、压力、r、残差及内积权重/质量矩阵。
父记录保存有效参数、源码/环境与每个文件的 SHA256，结果和报告分别位于 runs/reports。

```sh
uv run python -m experiments.imex_euler.validate run --config experiments/imex_euler/configs/validation.json
uv run python -m experiments.imex_euler.validate analyze --run runs/imex_euler_validation/<运行编号>
```

第二条命令只读保存数据，重算物理 L2 误差和观测阶，不构造模型、不求解、不自动绘图。
单线程可通过 OMP_NUM_THREADS、OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、VECLIB_MAXIMUM_THREADS、
NUMEXPR_NUM_THREADS、NUMBA_NUM_THREADS=1 启动；实际环境记入 manifest。

公开 Channel/Fourier 示例：

```sh
uv run python -m experiments.channel_ns.run --config experiments/channel_ns/configs/imex_euler.json
uv run python -m experiments.channel_ns.run --config experiments/channel_ns/configs/imex_euler_mrsav.json
uv run python -m experiments.periodic_ns.run --config experiments/periodic_ns/configs/imex_euler.json
uv run python -m experiments.periodic_ns.run --config experiments/periodic_ns/configs/imex_euler_mrsav.json
```

MAC 单次配置 `configs/mac_euler.json` 和 `configs/mac_euler_mrsav.json` 使用通用 workflow：

```python
import json
from pathlib import Path
from experiments.workflow import run_experiment
from experiments.analysis import analyze_runs

config = json.loads(Path('experiments/imex_euler/configs/mac_euler_mrsav.json').read_text())
directory = run_experiment(config)
report = analyze_runs([directory])
```

Channel/Fourier 分析沿用各自 `analyze --run <显式目录>`，可加 `--plot` 导出场图。
新格式首轮只支持固定或给定正步长；旧四格式的自适应和默认 campaign 保持原约定。
公式与压力语义见 [docs/imex-euler.md](../../docs/imex-euler.md)。
