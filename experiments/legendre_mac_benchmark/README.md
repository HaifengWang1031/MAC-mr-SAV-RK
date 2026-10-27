# Legendre–MAC 初步效率比较

目的：比较当前 Legendre 稀疏 LU、MAC 稀疏 LU、MAC 张量 Schur-CG 的实际成本，以及达到指定最终速度误差时的成本。求解器不作优化或公式修改。

统一光滑无滑移制造解：在矩形 `(lx, ly)=(1.3, 0.8)` 上取
`psi=0.1 exp(-t) sin(pi*x/lx)^2 sin(pi*y/ly)^2`，
`(u,v)=(psi_y,-psi_x)`，`p=0.1 exp(-t) cos(pi*x/lx) cos(pi*y/ly)`，`nu=0.1`。
连续体力独立由解析解生成。MAC 初值为顶点流函数的离散 curl，Legendre 初值为解析速度的质量投影。体力的 `exp(-t)` 与 `exp(-2t)` 两部分各自预处理后，在阶段中仅乘时间系数；预处理成本包含在初始化计时中。

`configs/preliminary.json` 集中设置物理参数、分辨率、格式、时间序列、重复次数、线性容差、求积及精度门槛。默认 SDIRK3 与 SDIRK3-mr-ccSAV（gamma=1），10 步到 T=0.02；固定 dt=0.002 与有固定种子的给定变步长序列总时间相同。**变步长序列不运行自适应控制器，不测拒步成本。**

计时使用单线程 `process_time`，取三次重复的中位数，并保留极值和 wall time。总 CPU 时间包括模型/后端/初值/体力初始化和整个公开 `integrate` 推进器（包含首次分解、阶段检查与诊断）；不包含导入、符号制造解、Numba 首次编译、误差求积、保存或绘图。记录单次热调用的对流、诊断和已缓存 Stokes 求解成本；这些微计时不与总时间相加，也不是精确的占比。固定步 LU 可复用；变步 LU 使用原有四项缓存。两种 MAC 后端保留完整散度和动量检查，Legendre 保留弱连续性检查及物理散度诊断。

共同物理 Gauss 网格上计算相对向量 L2 误差。Legendre 直接求值；MAC 分量在各自面位置作双线性重构并补上零切向壁值。因而 MAC 误差包括指定重构的误差。256 与 512 节点/方向检查误差积分的相对变化，超过 2% 的结果不得用于精度门槛比较。没有把 MAC 面内求和范数直接与谱质量范数作对比。精度门槛表只在测过的分辨率内选最省时的合格成员，没有外推；相近自由度的成本应另从 `timings.csv` 读取。

本机运行：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 UV_CACHE_DIR=/private/tmp/mac-audit-uv uv run --locked --offline python -m experiments.legendre_mac_benchmark.run
UV_CACHE_DIR=/private/tmp/mac-audit-uv uv run --locked --offline python -m experiments.legendre_mac_benchmark.analyze runs/legendre_mac_benchmark/<run-id>
```

计算只写 `runs/legendre_mac_benchmark/<run-id>/`：配置、完整步长序列、环境/源码身份、逐次结果、末态和校验值。分析显式读取该运行，核对校验值并由保存场复算误差，报告写 `reports/legendre_mac_benchmark/<run-id>/`（JSON、CSV、PNG），不启动计算。

本实验是短时光滑问题、现有代码及指定初始化/诊断策略的比较，不能推出长时、高 Reynolds 数、非光滑壁值或优化后谱算法的成本。没有计量峰值内存；误差包含时间离散误差。相同模式数与网格数意味着相近未知数规模，不意味着相同精度。Legendre 的弱连续性残差和求积节点上的物理散度必须分开读；MAC 的离散散度与 Legendre 的物理散度也不是同一种范数。

2026-10-04 本机主扫描 `20261004T040637-142e5d6d` 的 168 个案例完成，256/512 点误差求积最大相对变化 0.283%，两种 MAC 后端末态最大相对差 `9.83e-14`。SDIRK3 固定步长下，在本次扫描的分辨率中：相对误差门槛 `1e-2` 对应最省时 MAC 32² / 17.1 ms、Legendre 6 模态/方向 / 64.4 ms；门槛 `1e-3` 对应 MAC 张量 128² / 306 ms、Legendre 8 模态/方向 / 77.1 ms。时间均含初始化和 10 步推进，取三次中位数，不是单步时间。

补充运行 `20261004T040954-95fc7574`、`20261004T040956-5ab5593d` 将步长减为 0.001、0.0005，保持 T=0.02，使用同一流程及保存格式。两种三阶格式的 Legendre 16 模态固定步误差均从约 `2.031e-8` 降到 `2.786e-9`、`3.515e-10`，支持高模态误差平台主要受时间误差影响；不把主扫描非单调的微小误差当作空间收敛阶。这两次补充只检查时间误差，不加入三次重复的效率统计。
