# MAC-mr-SAV-RK

二维不可压 Navier–Stokes 方程的无滑移 MAC 求解器。实现普通 IMEX-SDIRK2 与笔记中的增量 SDIRK2-mr-ccSAV，支持固定步长及给定步长序列。阶段速度和压力通过耦合 Stokes 系统一起求解，没有压力分裂子步。

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
  model.py                模型接缝：格式只通过它取算子、求解与诊断（15 个成员）
  mac_ns.py               MAC 速度形式 NS 模型（接缝实现之一）
  integrate.py            固定/给定序列共同推进
  mac/
    grid.py               交错网格、打包与内积
    operators.py          D、G、K 的相容离散
    kernels.py            Numba 对流、散度、内积核
    stokes.py             耦合稀疏求解、LU 缓存、后端接口
  schemes/
    sdirk2.py             普通 IMEX-SDIRK2
    sdirk2_mrsav.py        增量 SDIRK2-mr-ccSAV
    roots.py              全部数值实根及最小绝对值选择
  spectral/               第二个离散：单元素 Dirichlet 组合 Legendre（无新依赖）
    basis.py              1D 精确矩阵（质量、刚度、散度、投影）与基函数递推
    assembly.py           Kronecker 张量积装配、Galerkin 载荷与求值
    lifting.py            非齐次壁面（移动顶盖）作为已知场，载荷入右端、矩阵不变
    stokes.py             移位 Stokes 鞍点求解与分解缓存
    model.py              谱离散的接缝实现
  adaptivity/             仅预留接口说明
experiments/              各实验的配置、计算与分析入口
runs/                     配置、manifest、HDF5、日志与批次记录
reports/                  分析来源、图、表和日志
tests/                    数值和工作流回归检查
tools/                    步长计时与快照绘图
archive/                  旧程序来源索引
```

## 验证与阅读

```sh
uv run mypy
uv run pytest
```

- [数学实现](docs/numerics.md)：MAC 边界处理、压力规范、阶段公式及三次方程。
- [架构和存储](docs/architecture.md)：缓存、内存、运行身份与失败行为。
- [验证结果](docs/validation.md)：实际网格、收敛阶和局限。
- [确认的范围](docs/spec.md)、[领域约定](CONTEXT.md)。

首版不包含变网格、非齐次边界、迭代 Stokes 实现、自适应控制或断点续算。阶段压力是增量乘子；除定常 Stokes 实验外，不将其视为物理终点压力的二阶近似。数值测试不能替代长期稳定性或收敛性的数学证明。

## 经典算例可视化

[顶盖驱动方腔流 Re=100](experiments/cavity/README.md)：新增常速度移动顶盖边界载荷、网格对比与流线/涡量图。

实验记录导航见 [现存实验盘点](docs/experiment-inventory.md)，测试保留原则见 [代表性测试](tests/README.md)。
