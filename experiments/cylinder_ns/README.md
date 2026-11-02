# FEM 圆柱绕流：DFG 2D-2

使用本项目普通 / mrSAV 时间格式的 **Re=100、定常入口**圆柱绕流，不是入口随时间变化的 DFG 2D-3。

- 通道 `[0,2.2]×[0,.41]`；圆心 `(.2,.2)`、半径 `.05`。
- 密度 1，运动黏度 `.001`；抛物线入口峰值 `1.5`、截面平均值 `1`。
- 壁面/圆柱无滑移；出口为梯度黏性形式的自然牵引 `ν∂ₙu−pn=0`，不附加压力均值约束。
- scikit-fem Taylor–Hood P2/P1，一致质量矩阵，无压力罚项。圆柱为明确记录边数的仿射多边形；P2 速度不意味着曲边几何。
- 初值为相容的定常 Stokes 场，不是内部零速度的脉冲启动；只比较充分发展后的周期量。

## 运行与只读分析

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  uv run --locked --group notebook python -m experiments.cylinder_ns.run \
  --config experiments/cylinder_ns/configs/smoke.json

# 使用上一步打印的显式目录，不自动计算或选择 latest。
MPLBACKEND=Agg uv run --locked --group notebook python -m experiments.cylinder_ns.analyze \
  runs/cylinder_ns/<run-id>
```

`--root` 选择 runs/reports 的共同根，`--rerun` 明确新建一次运行；完整、校验通过且配置/源码/环境相同才复用。
数值输出位于 `runs/cylinder_ns/<run-id>/`，分析输出位于 `reports/cylinder_ns/<run-id>/`。
分析只读取保存的网格、P2/P1 系数和历史，不构造动力学模型或启动 Stokes/时间推进。

Re500加密SAV/回流耗散配置为`configs/re500_backflow_fine_optimized.json`：
nu=0.0002、resolution=2、96边圆柱、T=12，使用recycled/SuperLU/auto和单线程。
完整旧版/优化版串行效率对照的当前交接见
`runs/cylinder_re500_efficiency/20261030T135609-94ce170f/STATUS.md`；提交记录不能代替完成结果。

计算效率选项独立于物理参数与时间格式：`stokes_backend=recycled` 可复用邻近步长的
因子作预条件器，仍求解当前精确系统。默认 SuperLU 的 `stokes_ordering=auto` 在系统
未知量不少于20000时使用跨步缓存的METIS排序，小系统使用MMD；也可显式指定
`mmd` 或 `metis`。Linux x86_64 可安装 `uv sync --locked --extra pardiso`，再设
`stokes_factor_backend=pardiso, stokes_threads=4`；该后端使用自己的 `auto` 排序。
并行线程越多不一定越快，应按相同精度的完整步计时选择。所有原始残差检查保留，
旧配置与已有记录继续读取。基准和限制见 [效率记录](../../docs/fem-optimization.md)。

## 时间控制与记录

六个注册格式均支持固定/给定步长。普通及 SAV 的 SDIRK2/3 支持已有嵌入估计的 I/PI 自适应；Euler 不虚构嵌入估计器。
`mode=fixed` 使用 `dt` 或 `steps`；`mode=adaptive` 使用 `initial_step` 与 `controller`。
SAV 可选 `scalar_atol` 联合控制已实现的标量缺陷。控制器、拒绝、失败前缀、实际终点及最近接受节点快照均保存/校验。
`resolution` 控制内部/近壁/尾流分辨率，`circle_segments` 可独立控制圆周几何。
新生成网格默认约 `96*resolution` 条圆周边（向上取4的倍数、至少16），
近壁径向间隔为 `.0025/resolution`，五圈节点；旧记录始终读取实际保存网格。
这些密度不是精度保证：已完成的旧粗网格 T=12 虽形成稳定涡街，升力仍偏差约10–19%。

HDF5 保存实际网格拓扑、四类边界标签、完整 P2 分量、自由/约束 DOF 映射和固定入口值，读取不重新运行 Delaunay。
`final/stage_pressure` 仍是格式的阶段乘子，**不是物理端点压力**。
物理压力由瞬时半离散约束速度率方程独立重建；`final/physical_pressure` 与 `observations/` 明确记录时间及含义。
受力使用 `ν∇u−pI` 和指向流体的固体外法向，`CD/CL=2F/(Umean²D)`；压差探针为 `(.15,.199999)`、`(.25,.199999)`。
模型只约束完整弱连续性；报告另存非零的物理散度，不把两者混为一谈。

输出包含网格/局部网格、速度、单元平均涡量、阻力/升力/压差、步长/控制误差历史，以及至少三个不同快照时的 `wake.gif`。
涡量色标在 ±20 饱和；显示使用原始保存系数，无时间插值。输入校验和在只读分析前后核对。

## 性能对照与独立细化

计算与报告入口分离，全部消费显式种子/运行目录。冻结优化前后源码后执行：

```bash
# 从同一保存网格/末态重新启动短窗口；三次交错重复，另做共同接受调度重放和剖析。
uv run --locked --group notebook python tools/benchmark_fem_adaptive.py compare \
  --run runs/cylinder_ns/<developed-seed-id> \
  --baseline <frozen-before-source> --source <frozen-after-source> --duration .05 --repeats 3
MPLBACKEND=Agg uv run --locked --group notebook python -m experiments.cylinder_ns.performance \
  runs/fem_adaptive_benchmark/<explicit-id>

# 独立计算：同种子插值到新网格，固体内部零延拓，然后施加固定迹并弱 Leray 投影。
uv run --locked --group notebook python tools/benchmark_fem_adaptive.py reference \
  --run runs/cylinder_ns/<developed-seed-id> --resolution 2.5 --circle-segments 240 \
  --duration 3 --adaptive --control-config experiments/cylinder_ns/configs/native_adaptive.json \
  --backend recycled
# --prepare-only 只生成带摘要的输入，可复制到独立 SSH 目录，用同工具 worker 子命令执行。
# 另算更粗 bulk、同 bulk 更密 circle，以及相同初态/网格的 native_tight.json 参考，再只读汇总：
MPLBACKEND=Agg uv run --locked --group notebook python -m experiments.cylinder_ns.campaign \
  --coarse runs/cylinder_refinement_reference/<coarse-id> \
  --fine runs/cylinder_refinement_reference/<fine-id> \
  --tight runs/cylinder_refinement_reference/<tight-control-id> \
  --geometry runs/cylinder_refinement_reference/<circle-only-id> \
  --performance runs/fem_adaptive_benchmark/<explicit-id>
```

原生参考使用 NPZ/JSON 独立命名空间，不冒充标准 HDF5 或 checkpoint 续算。
重启 `duration` 从保存的非零时刻算起。记录保留种子、真实网格/分量映射、起止时刻、
最终/快照速度、物理受力/压差、来源、成本与每个工作器摘要；这些参考没有全压力系数历史。
只读报告不重新投影或求压力。空间、圆周几何和时间隔离条件均严格检查。
固定参考也可用 `--step`，但必须验证时间稳定性：实际 resolution=2、step=.0015 虽完成线性/阶段检查，
SAV 标量却升至约.86且周期失败，不能作为精度参考。自适应参考复用原控制器，另存完整标量历史与末态物理 P1 压力。

周期资格要求最后三个完整升力周期的周期相对范围 `<.5%`、峰峰值相对范围 `<1%`。
项目参考门槛为 CD/CL 极值和升力 Strouhal 的2%、压差极值的3%；独立空间/几何敏感性2%、
时间敏感性1%作用于相同七个主要观测量，不对接近零的平均升力作病态百分比认证。
时间平均、极值中点（midrange）和峰峰值另行报告。引用档案的数值样本不是公开认证区间。

实测成熟窗口（2474三角形、三次交错单线程）的优化前/直接/复用后端总 CPU 中位数
为 `9.2206 / 4.2105 / 3.6684 s`，包含建立与物理压力观测；复用后端约 **2.51倍**。
共同调度最终速度相对 L2 差最大 `9.86e-11`、接受/拒绝决定一致。原始记录
`20261027T220152-37e946c6` 及报告保存全部范围与内存；不能外推为所有尺寸的全局最优。

## 已计算的细网格结果

显式原生参考 `runs/cylinder_refinement_reference/20261027T220845-10013a71` 使用
resolution=2.5、240边、SDIRK3-mrSAV、step=.00075，从 T=12 的具名状态推进到15。
末三升力周期已稳定；CD范围 `[3.16721,3.23175]`、CL范围 `[-1.03134,.99417]`、
St=`.301652`，七项参照误差最大约 `.985%`。这是**参照/发展周期门槛通过**，
该固定参考是较早的交叉检查；最终总体资格由下述四项独立自适应计算给出，不混用来源。

```bash
# 同一只读分析入口也接受显式原生 NPZ 参考，生成正确细网格的图像和61帧涡街动画。
MPLBACKEND=Agg uv run --locked --group notebook python -m experiments.cylinder_ns.analyze \
  runs/cylinder_refinement_reference/20261027T220845-10013a71
```

图像位于 `reports/cylinder_reference/20261027T220845-10013a71/figures/`。
`production.json` 现选用2.5级网格及保守的自适应上限；完整T=12的这份新配置尚未重算，
不冒充上述原生参考或已完成的旧粗网格运行。

## 最终自适应验收（已完成）

以下四项均从显式保存的 T=12 种子推进至15，并各自满足末三个完整升力周期的稳定门槛：

| 角色 | resolution / 圆周边 | 三角形 | 接受 / 拒绝 | 原生运行编号 |
| --- | --- | ---: | ---: | --- |
| bulk 较粗 | 2 / 192 | 18185 | 3793 / 0 | `20261027T231430-6ec4424f` |
| bulk 较细 | 2.5 / 240 | 26963 | 3794 / 0 | `20261027T231436-febe9803` |
| 更紧时间控制 | 2.5 / 240 | 26963 | 11968 / 0 | `20261027T231444-c351fb3f` |
| 独立圆周细化 | 2.5 / 480 | 29123 | 3831 / 0 | `20261027T231451-c8274875` |

标准控制为速度 `atol=1e-8, rtol=2e-4`、标量 `1e-5`、max_step `.001`；
时间参考为 `1e-9, 2e-5`、标量 `1e-6`、max_step `.0005`。时间对照原生网格/初始系数逐项相同。
空间、几何、时间七项主要量的最大相对差分别 **.500947%、.024943%、.0004223%**，
通过预设2%、2%、1%门槛；较细/更紧/几何参照误差均小于1%。更紧算例
`St=.301652435`、CD `[3.16721556,3.23174987]`、CL `[-1.03134520,.99416962]`。

最终只读报告为 `reports/cylinder_campaign/20261027T231436-febe9803-33020ea1/`，
含 `analysis.json`、`cycles.csv` 及网格、速度、涡量、周期/细化图、`steps.png` 接受/拒绝步长与标量历史、
61帧 `figures/wake.gif`。
重绘只需上述 campaign 命令传入表中四个显式编号和性能编号 `20261027T220152-37e946c6`，不启动求解。
完整性/模型构造禁止/末态压力探针对照及最终回归见 [validation](../../docs/validation.md)。
最终源码、原始运行/报告和失败证据的可校验归档位于 `runs/cylinder_delivery/20261028-qualified/`，
解包/只读重绘说明在其中的 `DELIVERY.md`；运行/报告默认被Git忽略，需另行备份。
这些是已测窗口与主要观测量的资格，不是连续PDE误差上界、所有SAV轨迹阶数或普适性能最优的证明。

## 验证状态与限制

公开运行/读取/重建/分析、六格式、I/PI、自适应拒绝/失败前缀及数据破坏拒绝已完成聚焦测试。
独立数值参照包括非零入口 Poiseuille、开边界能量通量多项式和静水线性压力的圆柱合力。
**短时 smoke 不构成充分发展周期基准、网格/容差敏感性或性能最优的证明。**
完整 T=12 自适应计算、细网格参照、独立空间/几何/时间敏感性及重复优化计时均已执行并完成核验；
早期未通过的粗网格仍保留失败资格，不能以最终合格结果改写历史。最终数值、报告编号和实际范围以
[validation](../../docs/validation.md) 为准。
开边界 SAV 工作项推导和能量定理限制见 [ADR0001](../../docs/adr/0001-open-flow-sav-boundary-work.md)。
