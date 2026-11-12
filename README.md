# MAC-mr-SAV-RK

二维不可压流的数值求解与实验项目。当前五类空间离散共用六种时间格式，阶段速度与压力满足同一移位 Stokes 系统；直接、Schur、Fourier 或带状求解是该系统的不同求解方式，没有压力校正分裂子步。

## 当前算法与支持范围

格式名称、设计阶数和阶段数以 [registry.py](solver/schemes/registry.py) 为准：

| 时间格式配置名 | 速度设计阶数 | 阶段数 | SAV 标量求解 | 步长支持 |
| --- | --- | --- | --- | --- |
| `imex_euler` / `imex_euler_mrsav` | 1 | 1 | SAV 每阶段一次冻结校正，P₀=1 | 固定、给定序列 |
| `sdirk2` / `sdirk2_mrsav` | 2 | 2 | SAV 每阶段一次冻结校正，P₁(z)=1+z | 固定、给定序列、I/PI 自适应 |
| `sdirk3` / `sdirk3_mrsav` | 3 | 4 | SAV 每阶段两次冻结校正，P₂(z)=1+z+z² | 固定、给定序列、I/PI 自适应 |

普通版每阶段求一个 Stokes 右端，SAV 版求两个共享同一矩阵的右端。SAV 从前一阶段r开始，每次只计算一个标量商；一/二/三阶每阶段固定1/1/2次，以倒数第二次标量值冻结P，使用最后一次r恢复速度、压力并检查阶段方程。默认比较组仍为四种 SDIRK 格式；[一阶实验](experiments/imex_euler/README.md)中的“六后端”指该研究选定的模型/后端组合，当前空间模型与后端清单见下表。

| 空间离散模型 | 边界与离散 | 已实现的 Stokes 后端 |
| --- | --- | --- |
| [MAC](docs/numerics.md) | 矩形无滑移、二阶交错差分 | 稀疏直接；DST/DCT 张量求逆＋压力 Schur-CG |
| [Legendre 谱 Galerkin](docs/spectral.md) | 单元素张量积、Dirichlet 组合基、固定 lifting | 稀疏直接；广义特征分解＋压力 Schur-CG |
| [Fourier–Leray](docs/fourier.md) | 双周期、严格 2/3 去混叠 | FFT 波数空间显式 Stokes 求解 |
| [Channel MAC](docs/channel.md) | 流向周期、上下无滑移、二阶交错差分 | 稀疏直接；流向 FFT＋法向带状 LU，保留全部 MAC 模式 |
| [FEM Taylor–Hood P2/P1](docs/fem.md) | 仿射三角网格、一致质量、反对称弱对流 | 直接 LU；旧因子缺陷校正/GMRES；可选 PETSc FGMRES＋块 AMG |

五类模型都可使用六格式的固定/给定步长，以及四种 SDIRK 嵌入格式的 I/PI 推进器。实验入口开放的配置不完全相同：通用 MAC 固定步入口使用直接后端，tensor 通过专属研究入口或后端注入使用；楔腔入口支持六格式固定步，默认 SDIRK2-mr-ccSAV，尚未开放自适应或给定步长序列。具体入口见[实验导航](experiments/README.md)。

FEM 另有[圆柱开放流动](experiments/cylinder_ns/README.md)、[移动顶盖楔腔](experiments/wedge_cavity/README.md)及 [Hele–Shaw 惯性 Brinkman](docs/hele-shaw.md)模型；独立标量 P2 Darcy 参照不采用 SAV。FEM 的弱载荷表示、一致质量 CG/METIS、可选 PARDISO 与 PETSc 设置见 [FEM 后端说明](docs/fem-multigrid.md)，实现支持与实测资格分别说明。

[自适应说明](solver/adaptivity/README.md)区分 SDIRK2(1)、SDIRK3(2)、可选标量局部缺陷控制、拒步及最小步长策略。Euler 尚无嵌入估计器或 step-doubling 实现；局部容差不代表全局误差上界。设计阶数不替代特定 PDE、边界和正则性条件下的收敛证明。

[受迫二维 MHD 逆级联边缘](experiments/inverse_cascade/README.md)使用既有 Fourier MHD
和两次固定校正 SDIRK3-mr-ccSAV，增加论文双场强迫、四阶／逆四阶耗散及通量统计。
当前是降 Reynolds 数的初步定性试算；粗步失真保留，尚非论文临界指数或完整 DNS 复现。

## 环境与首个计算

使用 uv 管理项目独立环境，Python 3.12；依赖版本由 `uv.lock` 固定。

```sh
uv sync --locked
uv run python experiments/cavity/run.py --config experiments/cavity/configs/re100.json
uv run python experiments/forced_ns_convergence/run.py --config experiments/forced_ns_convergence/configs/smoke.json
```

计算入口会打印结果目录。将其用于分析：

```sh
uv run python experiments/cavity/analyze.py --runs runs/cavity/<运行编号>
uv run python experiments/forced_ns_convergence/analyze.py --batch /绝对路径/batch.json
uv run python tools/plot_speed_snapshots.py --run runs/forced_ns/<运行编号>
```

正式配置在 `experiments/<实验名>/configs/`。`cavity` 是移动顶盖方腔；`forced_ns` 是定常体力 `(0, amplitude*sin(2*pi*m*x))` 驱动的方腔（同一 runner 也跑 `trig_ns` 系列配置）；`forced_rotation` 是 `(0,2π)²` 上有净力矩/无净力矩两种定常外力 A/B 的长期旋转实验，见 [README](experiments/forced_rotation/README.md)。`--rerun` 保留旧记录并新建一次计算；默认复用同配置、同代码、同环境且完整校验通过的结果。

## 目录

```text
solver/
  core.py                 State、History、Trial、Stage 与格式接口
  model.py                空间离散模型接缝：算子、求解与诊断协议
  momentum.py             共享阶段动量表示，默认使用模型系数
  mac_ns.py               MAC 速度形式 NS 模型（接缝实现之一）
  integrate.py            固定/给定序列共同推进
  mac/
    grid.py               交错网格、打包与内积
    operators.py          D、G、K 的相容离散
    kernels.py            Numba 对流、散度、内积核
    stokes.py             耦合稀疏求解、LU 缓存、后端接口
  schemes/
    imex_euler.py         一阶经典 IMEX Euler
    imex_euler_mrsav.py   一阶线性 mr-ccSAV，G=1-r、Q=1
    registry.py           格式工厂、阶数、阶段/候选容量与估计器元数据
    sdirk2.py             普通 IMEX-SDIRK2
    sdirk2_mrsav.py        增量 SDIRK2-mr-ccSAV
    sdirk3.py             普通四阶段 IMEX-SDIRK3
    sdirk3_mrsav.py       四阶段 SDIRK3-mr-ccSAV
    scalar_correction.py  共享冻结因子标量商，一/二/三阶固定1/1/2次
  spectral/               第二个离散：单元素 Dirichlet 组合 Legendre（无新依赖）
    basis.py              1D 精确矩阵（质量、刚度、散度、投影）与基函数递推
    assembly.py           Kronecker 张量积装配、Galerkin 载荷与求值
    lifting.py            非齐次壁面（移动顶盖）作为已知场，载荷入右端、矩阵不变
    stokes.py             移位 Stokes 鞍点求解与分解缓存
    model.py              谱离散的接缝实现
  fourier/                双周期 FFT 空间、Leray/Stokes、能量相容对流与 Model 实现
  channel/                流向周期、法向无滑移的 MAC 网格、算子、对流与 Model
  fem/                    scikit-fem P2/P1 空间、一致质量 Model 与无罚项 Stokes
  adaptivity/             I/PI 接受/拒绝、速度误差与可选标量缺陷控制
run_support/
  records.py              公共运行身份、保存、复用与完整性检查（不依赖实验）
  source.py               核心＋显式任务来源清单；旧来源策略兼容
postprocess/
  mac.py                  公共 MAC 保存场读取（不构造动力模型）
  plotting.py             通用字段/历史图与 TeX 样式
  diagnostics.py          显式完整/失败前缀的保存诊断分析
  snapshots.py            快照报告与独立命令行
experiments/
  controllers.py          共同 I/PI 构建、校验与 JSON 配置表示
  records.py              run_support 旧名兼容导出（旧入口仍用广泛来源策略）
  workflow.py             MAC 实验计算入口及历史工具名兼容导出
  <实验名>/               各实验的配置、计算、表示校验与只读分析
    records.py            periodic_ns/channel_ns/fem_ns/cylinder_ns/hele_shaw_airfoil 专属读取
runs/                     配置、manifest、HDF5、日志与批次记录
reports/                  分析来源、图、表和日志
tests/
  core/                   不依赖实验/工具的数值、公共记录及只读后处理回归
  references/             测试专用 ODE、MAC、Fourier、Channel 数学参照
  tasks/                  实验配置、运行/分析、扫描、动画与工具回归
  core_distribution.py    无实验/工具目录的隔离核心验证入口
tools/                    旧快照只读命令 adapter；其余计时/回放/探针/搜索仍属计算任务
archive/                  旧程序来源索引
```

核心记录与来源的独立入口见 [run_support](run_support/README.md)。当前只有周期 MHD
入口完成显式任务来源迁移；其他实验暂保留旧策略。测试已按核心/任务分离，
通用只读后处理已抽取到 [postprocess](postprocess/README.md)，与仍会计算的
[任务工具](tools/README.md)分开；实验/任务测试仍在跟踪中，本轮没有改变忽略规则。

## 验证与阅读

```sh
uv run --locked mypy
uv run --locked pytest -q                    # 当前检出的全部测试，含任务测试
uv run --locked pytest -q tests/core         # 核心回归
uv run --locked python -m tests.core_distribution  # 无实验/工具目录的完整核心回归
uv run --locked python -m postprocess.snapshots --run /path/to/run --fields speed vorticity
```

[测试说明](tests/README.md)列出归属和参照范围。隔离入口复用当前解释器和已安装依赖，
不安装新环境，也不启动研究计算；可选后端的跳过仍须单列报告。

- [数学实现](docs/numerics.md)：MAC 边界处理、压力规范、阶段公式及固定次数标量校正。
- [当前架构与修改导航](docs/architecture.md)：模块归属、依赖方向、扩展点、记录兼容与测试入口；修改模块分工时同步维护。
- [验证结果](docs/validation.md)：实际网格、收敛阶和局限。
- [确认的范围](docs/spec.md)、[领域约定](CONTEXT.md)。
- [二维 MHD 长时间行为探讨](docs/mhd-long-time-discussion.md)：用现有方法复现磁岛合并、粗化与衰减；记录已有证据和定量复现缺口，MRI 仅留作历史讨论。
- [二维受迫 MHD 模型与文献](docs/forced-2d-mhd-literature.md)：机械外力＋背景磁场、机械外力＋磁性外源；慢失稳与局域传播结构、级联转变及长期通量统计，含 [BibTeX](docs/forced-2d-mhd.bib)。仅调研，尚未实现。

当前不包含变网格或断点续算。移动顶盖、I/PI 自适应及可选 Stokes 后端的适用范围见各实验说明；一阶格式尚无自适应估计器。阶段压力按格式乘子解释，不将其视为与速度同阶的物理端点压力；开放 FEM 的瞬时半离散物理压力另行恢复并标记。数值测试不能替代长期稳定性或收敛性的数学证明。

## 经典算例可视化

[顶盖驱动方腔流 Re=100](experiments/cavity/README.md)：新增常速度移动顶盖边界载荷、网格对比与流线/涡量图。

从 [实验入口导航](experiments/README.md) 选择算例、验证或性能研究；[2026-09-24 历史记录盘点](docs/experiment-inventory.md) 不是当前支持清单。测试保留原则见 [代表性测试](tests/README.md)。

四种 SDIRK 格式的制造解时间收敛实验见 [实验说明](experiments/manufactured_convergence/README.md)。
