# MAC-mr-SAV-RK

二维不可压 Navier–Stokes 方程的无滑移 MAC 求解器。实现普通 IMEX-SDIRK2 与笔记中的增量 SDIRK2-mr-ccSAV，支持固定步长及给定步长序列。阶段速度和压力通过耦合 Stokes 系统一起求解，没有压力分裂子步。

## 环境与首个计算

使用 uv 管理项目独立环境，Python 3.12；依赖版本由 `uv.lock` 固定。

```sh
uv sync --locked
uv run python experiments/decay/run.py --config experiments/decay/configs/default.json
```

计算入口会打印结果目录。将其用于分析：

```sh
uv run python experiments/decay/analyze.py --runs runs/decay/<运行编号>
```

正式配置在 `experiments/<实验名>/configs/`。`stokes_mms`、`ns_mms`、`decay` 分别验证 Stokes 空间误差、NS 制造解和无外力衰减。`--rerun` 保留旧记录并新建一次计算；默认复用同配置、同代码、同环境且完整校验通过的结果。

## 目录

```text
solver/
  core.py                 State、History、Trial、Stage 与格式接口
  mac_ns.py               速度形式 NS 模型
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
  adaptivity/             仅预留接口说明
experiments/              制造解、配置、计算与分析入口
runs/                     配置、manifest、HDF5、日志与批次记录
reports/                  分析来源、图、表和日志
tests/                    数值和工作流回归检查
tools/validate.py         有限规模验收计算
docs/                     数学约定、架构、验证、审查记录
archive/                  旧程序来源索引
```

## 验证与阅读

```sh
uv run mypy
uv run pytest
uv run python tools/validate.py
```

- [数学实现](docs/numerics.md)：MAC 边界处理、压力规范、阶段公式及三次方程。
- [架构和存储](docs/architecture.md)：缓存、内存、运行身份与失败行为。
- [验证结果](docs/validation.md)：实际网格、收敛阶和局限。
- [确认的范围](docs/spec.md)、[领域约定](CONTEXT.md)。

首版不包含变网格、非齐次边界、迭代 Stokes 实现、自适应控制或断点续算。阶段压力是增量乘子；除定常 Stokes 实验外，不将其视为物理终点压力的二阶近似。数值测试不能替代长期稳定性或收敛性的数学证明。

## 经典算例可视化

[顶盖驱动方腔流 Re=100](experiments/cavity/README.md)：新增常速度移动顶盖边界载荷、网格对比与流线/涡量图。
