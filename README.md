# MAC-mr-SAV-RK

二维不可压流的核心数值库：五类空间离散共用六种时间格式。阶段速度与压力满足同一移位
Stokes 系统；直接、Schur、Fourier 或带状求解是同一系统的不同求解方式，不是压力校正分裂。

当前 Git 文件树保留求解器、公共记录/只读后处理、数学说明及核心回归。研究配置、runner、
专属分析、任务测试及计算工具**仅本地保留在原路径**；它们不随新的源码检出提供。
[保留与恢复说明](docs/core-localization-plan.md)列出精确范围、备份和限制。Git 历史未重写。

## 当前算法与支持范围

格式身份、设计阶数和阶段数以 [registry.py](solver/schemes/registry.py) 为准：

| 时间格式配置名 | 速度设计阶数 | 阶段数 | SAV 标量校正 | 步长支持 |
| --- | --- | --- | --- | --- |
| `imex_euler` / `imex_euler_mrsav` | 1 | 1 | 每阶段一次，P₀=1 | 固定、给定序列 |
| `sdirk2` / `sdirk2_mrsav` | 2 | 2 | 每阶段一次，P₁(z)=1+z | 固定、给定序列、I/PI 自适应 |
| `sdirk3` / `sdirk3_mrsav` | 3 | 4 | 每阶段两次，P₂(z)=1+z+z² | 固定、给定序列、I/PI 自适应 |

普通版每阶段求一个 Stokes 右端，SAV 版求两个共享矩阵的右端。以先前阶段 r 为预测值，
固定进行 1/1/2 次标量商校正，使用倒数第二次值冻结 P、最后一次 r 恢复速度、压力及
lifting/SAV 功，再检查完整阶段方程。无多项式求根或 Newton 迭代；旧根槽及来源只读兼容。

| 空间模型 | 边界与离散 | 已实现的 Stokes 后端 |
| --- | --- | --- |
| [MAC](docs/numerics.md) | 矩形无滑移、二阶交错差分 | 稀疏直接；DST/DCT＋压力 Schur-CG |
| [Legendre](docs/spectral.md) | 张量积 Dirichlet 组合基、固定 lifting | 稀疏直接；广义特征分解＋压力 Schur-CG |
| [Fourier–Leray](docs/fourier.md) | 双周期、严格 2/3 去混叠 | FFT 波数空间显式求解 |
| [Channel MAC](docs/channel.md) | 流向周期、上下无滑移 | 稀疏直接；流向 FFT＋法向带状 LU |
| [Taylor–Hood FEM](docs/fem.md) | 仿射三角网格、一致质量 P2/P1 | 直接 LU；旧因子缺陷校正/GMRES；可选 PETSc FGMRES＋块 AMG |

另保留开放边界、固定移动壁 lifting、[Hele–Shaw/Brinkman](docs/hele-shaw.md)、
周期 MHD/受迫 MHD 与完整约束检查。模型、表示与格式的适用条件由各自说明限定；
可选后端存在不等于已获所有配置资格。[FEM 后端说明](docs/fem-multigrid.md)单列测量限制。

[自适应说明](solver/adaptivity/README.md)区分 SDIRK2(1)/SDIRK3(2)、I/PI、标量缺陷控制、
拒步和最小步长策略。Euler 无嵌入或 step-doubling 实现；局部容差不是全局误差上界。
阶段压力不自动成为同阶端点物理压力。固定网格时间测试、PDE 空间收敛和长期统计资格分开。
具体研究和远端生产授权仍以本地记录为准，本仓库精简不扩大科学资格。

## 环境与首个计算

Python 3.12，依赖由 `uv.lock` 固定。新环境可以使用 `uv sync --locked`；已有环境可直接
使用其 Python，不必重装。下面只构造公共 module，进行两个零解小步，不依赖研究包、
不写正式 runs/reports，也不是精度或长期稳定性的验证。

```sh
uv run --locked python - <<'PY'
import numpy as np
from solver.core import State
from solver.mac.grid import MACGrid
from solver.mac_ns import MACNavierStokes
from solver.integrate import integrate, step_sizes
from solver.schemes.registry import make_scheme

grid = MACGrid(8, 6)
initial = State(0., np.zeros((grid.ny, grid.nx+1)), np.zeros((grid.ny+1, grid.nx)))
result = integrate(MACNavierStokes(grid, nu=.01), make_scheme('sdirk2_mrsav'),
                   initial, step_sizes(.02, dt=.01), snapshots=[0., .02])
assert result.status == 'complete', result.error
print(result.final.t, result.diagnostics[-1]['divergence_inf'])
PY
```

保存与完整性 interface 见 [run_support](run_support/README.md)。已有完整 MAC 保存记录可
用公共只读命令分析，不生成新计算：

```sh
uv run --locked python -m postprocess.snapshots --run /path/to/run --fields speed vorticity
```

旧 `python tools/plot_speed_snapshots.py ...` 仍是同一实现的 adapter。
[postprocess](postprocess/README.md)说明支持的布局；不把 Fourier/Channel/FEM 系数猜作 MAC 场。

## 目录与本地研究

```text
solver/                 格式、Model、五类离散、完整线性求解及自适应
run_support/            公共记录、身份、校验和显式/历史来源策略
postprocess/            保存 MAC 场、图/排版、诊断及快照报告/CLI
tests/core/             算子、阶段、推进、约束、记录与公共后处理回归
tests/references/       测试私有的已知 ODE/PDE/独立离散参照
tests/core_distribution.py  无研究包的完整核心回归入口
tools/                  公共旧 CLI adapter；其他本地计算脚本不随检出提供
docs/                   数学、架构、规范、资格/历史证据与恢复说明
archive/README.md       原桌面程序的只读来源说明
```

完整本地工作区另外有 `experiments/`、`tests/tasks/`、五个会计算的 tools、五份研究资料、
`runs/`、`reports/` 和 scratch。它们保持原路径、单独备份，不移动或复制到公共 module；
Git 忽略不保证不会被分支切换/reset/clean 覆盖。不要对唯一研究副本执行 `git clean -fdx`。
专属配置/reader/冻结 worker 的旧路径在本机保留；干净检出不承诺导入这些本地包。

新 `provenance(task_paths=...)` 不按 Git 跟踪状态筛选；本地被忽略的真实任务依赖仍需入清单。
尚未迁移的 legacy 来源继续广泛扫描磁盘。不要用新来源重算旧 manifest，或绕过旧链/retarget
的身份校验。修改 pyproject 配置也真实改变新运行来源；已有运行/冻结源码不回写。

## 验证与阅读

```sh
uv run --locked pytest -q                # 检出的所有测试；本地 tasks 存在时也收集
uv run --locked mypy                     # 默认公共目标，不递归本地计算工具
uv run --locked python -m tests.core_distribution
```

拥有完整本地研究文件时，另行运行：

```sh
uv run --locked pytest -q tests/tasks
uv run --locked mypy solver run_support postprocess experiments tools tests/references tests/core_distribution.py
```

[测试导航](tests/README.md)区分机制和任务；默认 `testpaths=["tests"]` 未缩窄。隔离入口
复用已安装环境，不是最小依赖安装验证；可选后端 skip 仍单列，不授予新资格。

- [数学实现](docs/numerics.md)、[架构与修改导航](docs/architecture.md)。
- [确认范围](docs/spec.md)、[领域约定](CONTEXT.md)、[术语](GLOSSARY.md)和 [ADR](docs/adr/)。
- [实际验证与局限](docs/validation.md)、[受迫 MHD 历史审计](docs/inverse-cascade-toolchain-audit.md)。
- [本地化清单与备份恢复](docs/core-localization-plan.md)、[工具归属](tools/README.md)。

数值测试不能替代数学证明、完整 DNS、统计稳态或生产续算资格。研究与输出历史文件仅作
按日期解释的证据，不是新公共计算任务或默认参数。
