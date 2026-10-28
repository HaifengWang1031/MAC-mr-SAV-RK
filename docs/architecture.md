# 架构与记录约定
2026-10-05 Legendre 优化：模型与直接/张量后端共用空间算子和有界求积缓存；公开 Space 采样方法保持独立返回值。可选张量后端只更换线性求解算法，默认 direct。legendre_optimization 以独立 NPZ/JSON 运行记录保存两版源码、场、阶段压力/根、计时与校验和，报告只读消费显式运行编号；原 HDF5 格式保持兼容。见 docs/legendre-optimization.md。

2026-10-05 TS 算例在 Channel HDF5 中追加 initial_mode 属性及 unit_velocity，不改变既有数组、阶段或根容量。
channel_instability 的父记录保存配置、连续特征值参照、验证成员和非线性成员的显式运行编号及校验和。
计算先通过线性增长门槛；分析只重读保存场与参照，不构造模型或重算特征值。
移动坐标仅在渲染时应用，所有运行场和真实时刻仍保留实验室坐标。

2026-10-04 一阶格式：solver/schemes/registry.py 提供名称、工厂、阶数、阶段数、候选容量及估计器元数据。
Euler 两格式返回一个阶段；公共写入器接受 n_stages/n_roots，HDF5 schema_version=1 的阶段维为1，候选容量1。
普通格式 count=0，SAV count=1 保存唯一线性解；旧2/4阶段与3/5容量保持原形状。Euler 新记录使用
euler_step_multiplier_mean_zero 压力标签，旧数据无需迁移。独立 imex_euler_validation 用带校验和的 NPZ
保存最终场、参照及内积，JSON/日志保存来源与状态，报告只读重算误差。参见 docs/imex-euler.md。

2026-10-04 Channel 增加第四个 Model 实现。内部为实数周期 MAC 自由度；公开 u 保留重复周期末列，v 的上下壁面零。`experiments/channel_ns` 使用专属 reader 和 `periodic_x_no_slip_y/mac_second_order/staggered_mac_periodic_x` 标签；复用基础 HDF5 并在 pending 文件补齐标签、attempts 后原子更名。Channel runs/reports 与旧实验隔离。DirectStokes 仅泛化算子类型，不改分解、缓存或压力规范公式。默认 direct，可选 FFTStokes 将同一 MAC 系统按流向模式拆成带状耦合系统；缓存系数键及全方程检查保持一致。见 docs/channel.md。

2026-10-04 周期模型增加第三个 Model 实现。`solver/fourier` 的 State.u/v 是实数 collocated_yx 场，Model.vector/内部阶段为复数 rFFT；Core.State、时间格式和旧 MAC/Legendre 布局不改。专属 `periodic_ns` reader 检查配置和 HDF5 的 boundary/discretization/field_layout/fft_norm/dealias 标记，不执行 MAC 面平均。基础 HDF5 沿用 `save_result`；周期写入器先在 pending HDF5 添加表示标签及自适应 attempts，再原子更名和最终化 manifest。周期及验证 runs/reports 单独命名；失败前缀可以诊断，收敛/计时汇总要求完整同源子记录。见 docs/fourier.md 和 experiments/periodic_ns/README.md。

2026-10-28 FEM 圆柱与自适应：`solver/fem` 的 mesh-backed space / `OpenFlowModel` 保留共享格式接缝，固定入口仿射 lifting、自然出口无压力规范；封闭矩形模型仍保留积分均值规范。完整弱连续性和真正质量逆动量系数残差不因直接/复用后端而削弱。P2 求积缓存与有界旧 LU 预条件器只加速相同离散系统，不近似步长或变更控制器。

`experiments/cylinder_ns` 单独拥有配置、物理观测和 HDF5 表示；真实网格/边界/DOF 映射保存后读取不重新三角化。阶段增量压力和独立瞬时约束速度率的物理压力分开命名。固定/自适应共同推进器的接受观测计时独立；观测失败仍保留已接受状态/快照。`fem_ns` 共享严格 attempts 契约，并保留旧固定有效配置的身份。

`tools/benchmark_fem_adaptive.py` 是显式计算入口：保存原生网格/状态重启、独立空间/几何及时间参考、交错重复与共同调度重放。原生细化先插值/固体零延拓、施加固定迹并弱 Leray 投影，记录种子和实际起止时间；不是 checkpoint 续算。其 NPZ/JSON campaign 命名空间与标准 HDF5 运行分离。`cylinder_ns.records/performance/campaign` 只验证已完成显式记录及所有输入/工作器摘要，重建基函数、周期量和图像；不构造动力学模型或启动任何求解。报告均写入独立 reports 命名空间。详见 [圆柱使用说明](../experiments/cylinder_ns/README.md)。

2026-10-27 FEM 增加第五个 Model：`solver/fem` 分离确定性 P2/P1 空间、一致质量模型与无罚项直接 Stokes；时间格式和共同推进器不改。内部为自由速度系数，State.u/v 为含边/壁自由度的完整标量 P2 分量系数，按库映射拆装而非连续两半。

`experiments/fem_ns` 使用专属 reader，沿用基础 HDF5 schema_version=1 并增加 fem_schema_version=1、FEM 布局/元素/边界/压力规范/库版本标签，以及 mesh、dofs 数据集。保存完整速度和最后接受步各阶段 P1 压力；快照只含速度/r，不含压力历史。pending 文件补齐元数据后更名并最终化 manifest。来源身份单独纳入 scikit-fem，不改变旧记录；加载需保存/安装库版本一致，复用另需同配置/源码/环境、完整状态和校验通过。rerun 新建尝试，失败保留接受前缀；不支持恢复或持久化事务保证。

重建/分析只从已存网格和基函数评估，不构造动力模型或积分；显式运行目录输入，`runs/fem_ns` 与 `reports/fem_ns` 分离。报告 JSON/CSV/日志区分弱连续性、物理积分点散度与阶段乘子。见 [FEM](fem.md) 和 [运行记录说明](../experiments/fem_ns/README.md)。

2026-10-05 Legendre 优化：模型与直接/张量后端共用空间算子和有界求积缓存；公开 Space 采样方法保持独立返回值。可选张量后端只更换线性求解算法，默认 direct。legendre_optimization 以独立 NPZ/JSON 运行记录保存两版源码、场、阶段压力/根、计时与校验和，报告只读消费显式运行编号；原 HDF5 格式保持兼容。见 docs/legendre-optimization.md。

2026-10-05 TS 算例在 Channel HDF5 中追加 initial_mode 属性及 unit_velocity，不改变既有数组、阶段或根容量。
channel_instability 的父记录保存配置、连续特征值参照、验证成员和非线性成员的显式运行编号及校验和。
计算先通过线性增长门槛；分析只重读保存场与参照，不构造模型或重算特征值。
移动坐标仅在渲染时应用，所有运行场和真实时刻仍保留实验室坐标。

2026-10-04 一阶格式：solver/schemes/registry.py 提供名称、工厂、阶数、阶段数、候选容量及估计器元数据。
Euler 两格式返回一个阶段；公共写入器接受 n_stages/n_roots，HDF5 schema_version=1 的阶段维为1，候选容量1。
普通格式 count=0，SAV count=1 保存唯一线性解；旧2/4阶段与3/5容量保持原形状。Euler 新记录使用
euler_step_multiplier_mean_zero 压力标签，旧数据无需迁移。独立 imex_euler_validation 用带校验和的 NPZ
保存最终场、参照及内积，JSON/日志保存来源与状态，报告只读重算误差。参见 docs/imex-euler.md。

2026-10-04 Channel 增加第四个 Model 实现。内部为实数周期 MAC 自由度；公开 u 保留重复周期末列，v 的上下壁面零。`experiments/channel_ns` 使用专属 reader 和 `periodic_x_no_slip_y/mac_second_order/staggered_mac_periodic_x` 标签；复用基础 HDF5 并在 pending 文件补齐标签、attempts 后原子更名。Channel runs/reports 与旧实验隔离。DirectStokes 仅泛化算子类型，不改分解、缓存或压力规范公式。默认 direct，可选 FFTStokes 将同一 MAC 系统按流向模式拆成带状耦合系统；缓存系数键及全方程检查保持一致。见 docs/channel.md。

2026-10-04 周期模型增加第三个 Model 实现。`solver/fourier` 的 State.u/v 是实数 collocated_yx 场，Model.vector/内部阶段为复数 rFFT；Core.State、时间格式和旧 MAC/Legendre 布局不改。专属 `periodic_ns` reader 检查配置和 HDF5 的 boundary/discretization/field_layout/fft_norm/dealias 标记，不执行 MAC 面平均。基础 HDF5 沿用 `save_result`；周期写入器先在 pending HDF5 添加表示标签及自适应 attempts，再原子更名和最终化 manifest。周期及验证 runs/reports 单独命名；失败前缀可以诊断，收敛/计时汇总要求完整同源子记录。见 docs/fourier.md 和 experiments/periodic_ns/README.md。

求解器直接位于 `solver/`，没有旧项目的 `nextgen/` 过渡目录。环境由 `uv sync --locked` 建立，所有入口使用 `uv run`。`uv.lock` 与 `.python-version` 属于代码来源身份；`.venv/` 不入库。

`MACNavierStokes` 管理网格、黏性、离散算子和外力函数。时间格式只执行单步试算，返回 `Trial`；共同推进器在试算成功后更新 `History`。每个状态含具名 u、v 面数组、时间和 r。格式不会改写传入状态数组。当前历史只需一个接受节点；未来自适应控制可使用现有试算接口，但首版没有接受/拒绝控制器。

`DirectStokes` 使用相同系数矩阵求一个或多个右端，以 `(mass, viscosity)` 为精确缓存键，LRU 默认保留 4 个分解。固定步长通常只分解一次；给定序列按实际系数复用。修改网格必须新建模型。线性后端可通过 `StokesBackend` 接口注入。Numba 只覆盖数组核，SciPy 负责稀疏矩阵和分解；没有 fastmath。预热、LU 分解、线性求解、推进及总耗时分别记录，预热耗时可能包含已有 JIT 缓存的加载。

推进器只保留当前状态、每步标量诊断、最后一次的阶段压力和指定快照；不自动保存每步全场。诊断及根记录随步数线性增长。指定快照映射到最近接受节点，理论距离相同选择较早节点；请求与实际时刻分别保存。失败时保留接受前缀，不把失败试算当作接受结果。

## 计算记录

`runs/<实验名>/<运行编号>/` 包含：

- `config.json`：默认值展开后的参数及 `actual_steps`。
- `manifest.json`：运行身份、代码 SHA256、依赖版本、平台、线程环境、开始/完成时间、状态和结果校验和。
- `results.h5`：时间、步长、诊断、最终 u/v/r、阶段压力、快照、每阶段实根候选/数目/所选 r/残差。
- `run.log`：结果与异常摘要。

身份由实际配置、代码和环境构成。计算前查找同身份完整记录并校验配置与 HDF5；损坏或失败记录不复用。`--rerun` 创建新运行编号，不覆盖旧结果。不支持恢复中断计算。阶段异常由推进器转成失败记录，保存最后有效节点。进程被强制终止或机器掉电时可能留下 `running` 或临时文件，它们不会被复用。

HDF5 的根候选容量对二阶格式为 3，对三阶格式为 5，不足部分填 NaN，计数说明有效位置。普通格式计数为 0；SAV 计数保留数值求根器给出的重数。实际节点、阶段离散方程的残差和每阶段 r 都保存。`final/stage_pressure` 保存最后接受步的两阶段或四阶段乘子（定常 Stokes 则只有一个物理压力场）。

批次配置为 `base` 加显式 `cases` 列表，记录位于 `runs/<实验名>/batches/<批次编号>/`，含配置、成员、状态和日志。批次不隐藏成员失败。

## 分析记录

分析只接受显式运行目录。`reports/diagnostics/<分析编号>/` 下包含 `analysis.json`、`figures/`、`tables/` 和 `analysis.log`，记录输入运行与校验和、分析代码来源、参数及状态。项目内输入路径相对于项目根目录记录，迁移项目后仍可定位。

时间收敛分析要求被测解、参考解和更细参考解同网格、同格式、同物理参数及终止时间，报告参考细化差与最小被测误差的比值。这是参考敏感性检查，不是严格误差上界。分析从保存的最终场计算，不重新推进。失败运行可用于普通诊断，不能用于时间收敛表。
