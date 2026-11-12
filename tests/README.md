# 代表性测试

默认运行 `uv run --locked pytest -q`，仍覆盖当前检出的全部核心与任务测试，不静默隐藏实验回归。
测试输出和临时实验由 pytest 的 tmp_path 管理，不写入正式 runs/reports。生产实验不能充当
单元测试的固定通过条件。

## 分层与执行

| 目录/入口 | 归属 |
| --- | --- |
| `core/` | 公开算子、线性求解、空间/时间阶、阶段/SAV/约束、自适应、公共记录与只读后处理；不依赖实验/工具 |
| `references/` | 测试私有 ODE、解析 MAC/Fourier、连续/离散 Channel 数学参照；不从测试文件借 helper |
| `tasks/` | 研究配置、campaign、专属 run/load/analyze、物理观测、绘图、工具与历史数据探针 |
| `core_distribution.py` | 只复制 solver/run_support/postprocess 与核心测试，在无实验/工具/任务测试的临时副本验证完整核心套件 |

```sh
uv run --locked pytest -q tests/core
uv run --locked pytest -q tests/tasks
uv run --locked python -m tests.core_distribution
uv run --locked mypy
```

隔离入口复用当前解释器、不安装环境；子进程使用 `-I`、禁止 research 包导入、禁用外部
pytest 插件自动加载。`--collect-only` 只验证收集，默认命令实际执行全部核心测试。
`core/test_core_distribution.py` 自动验证隔离收集；发布前仍须单独跑完整隔离命令。
可选后端仍按原规则 skip，并报告覆盖限制，不因任务分离取得新资格。

同名测试文件可以在 core/tasks 各有一份，**旧测试函数没有复制或丢弃**：例如
`test_sdirk3` 的 ODE/MAC/谱阶段归 core，保存布局/campaign 归 tasks；`test_fourier_mhd`
的解析通量/扩散归 core，研究运行/只读图归 tasks；边界功的解析常数归 core，专属记录标签归 tasks。
`test_cavity` 检验实验的顶盖载荷构造，归 tasks，而不是为了名字好看放入 core。
`test_fem_references` 的两项检验实际 workflow 默认值及预投影策略，归 tasks；独立 FEM
算子/求积/空间阶/六格式时间参照均留在 core。

测试参照只描述明确的数学问题，不是第二套研究配置/生产工厂；实验中的参照和工具实现
保留原归属，不反向依赖 tests。连续 PDE 导数、独立固定网格 stencil、半离散轨迹的
验证范围分别说明，不将时间验证冒充 PDE 空间/阶数证明。此阶段未改忽略规则、未取消
跟踪任务。通用只读工具已抽取到 postprocess，其他会计算的任务工具仍保留原依赖。

## 代表机制索引

下表保留机制名称；完整归属按上述目录查看，同名文件跨两层时分别承担数值与任务断言。

| 机制 | 保留的测试 |
|---|---|
| 实验目录不存在时的记录/来源、清单完整性与可迁移身份 | test_run_support |
| 旧来源编码 | core/test_record_compatibility |
| 旧导入、冻结 worker 及显式周期 MHD 来源 | tasks/test_record_compatibility |
| 无研究目录的隔离收集/完整核心执行 | core/test_core_distribution、core_distribution.py |
| 已知保存 MAC 场、快照统计/独立来源、失败前缀、完整性/布局拒绝与无计算模块 CLI | core/test_postprocess |
| TeX 定位/回退/异常恢复、继承 TeX 开关的实际渲染及单面板历史图 | core/test_plotting；core/test_postprocess 同时实际生成两类报告 |
| 旧绘图/分析/快照名的同实现 adapter | tasks/test_postprocess_compatibility |
| MAC 算子与边界 | core/test_model_seam、core/test_kernels；实验顶盖载荷另见 tasks/test_cavity |
| Stokes 求解、压力定标与分解复用 | test_stokes、test_numerics 的空间收敛 |
| 二阶、SAV 能量与冻结阶段方程 | test_numerics、test_schemes |
| 三阶 ODE 精度、MAC 定常态与四阶段数值 | core/test_sdirk3 |
| 四阶段保存布局与参考/campaign 标签 | tasks/test_sdirk3 |
| MAC Tensor/Direct 自适应拒步、步长、完整约束及末态 | core/test_tensor_stokes_adaptive；任务配置选择另见 tasks 同名文件 |
| 一阶 ODE、独立耦合线性系统、能量平衡与标量衰减 | test_imex_euler |
| 一阶六后端压力/完整约束、固定 lifting 与失败检查 | test_imex_models |
| 单阶段工作流、失败前缀、元数据与只读验证分析 | test_imex_workflow |
| 谱空间、lifting 与去混叠 | test_spectral |
| 谱求积缓存、张量质量逆、多右端与等价计时流程 | test_spectral_optimization, test_legendre_optimization |
| 谱张量 Stokes、真实 Schur、六格式与自适应拒步 | test_spectral_tensor |
| 编译后单步计时、冷热分解计数、四模型保存与只读分析 | test_step_efficiency |
| 周期 FFT/Parseval/伴随/严格频带与 Leray/Stokes 零模 | test_fourier_space、test_fourier_stokes |
| 周期解析对流、能量配对、四格式 MMS、I/PI 与标量控制 | test_fourier_model、test_fourier_integration |
| 周期布局、保存复用、失败前缀、只读分析与计时子记录 | test_periodic_workflow |
| Channel 保存场动画、Poiseuille 参照、周期粒子输运与只读导出 | test_channel_animation |
| 顶盖方腔动画、移动壁插值、保存场与只读导出 | test_cavity_animation |
| 周期动画、解析受迫剪切、双周期粒子与只读导出 | test_periodic_animation |
| SAV 圆柱对比、保存网格、短窗重算、只读动画、连续旋度恢复与遮罩平滑 | test_cylinder_wake_compare |
| Orr–Sommerfeld 独立特征值、MAC TS 初值/约束、增长及移动坐标 | test_channel_instability |
| 时间序列与失败前缀 | test_integrate |
| 保存、复用、失败记录与显式分析 | test_workflow、test_forced_convergence |
| 自适应批次、四分边界、容差选择与固定阶梯 | test_adaptivity、test_kolmogorov_adaptive |
| 显式强制下限、有限超容差接受、硬失败保留和旧配置身份 | test_min_step_policy |
| 出口回流独立能量积分、SAV配对、正向Poiseuille及保存标签 | test_backflow_outlet |
| FEM一致质量分量共享、解析弱式与真实散度 | test_fem_model、test_fem_quadrature |
| Hele–Shaw Darcy/Brinkman解析参照、反应迹载荷、六格式时间细化与SAV边界功 | test_hele_shaw |
| Hele–Shaw实际网格、固定/自适应保存与严格终点、只读动画 | test_hele_shaw_workflow（网格生成需可选gmsh） |
| FEM多右端缺陷校正、跨shift/模型排序、GMRES/鲁棒回退、原生线程恢复 | test_fem_recycled_stokes |
| FEM六格式开放解析解、固定/自适应保存和只读加载、旧配置身份 | test_fem_open_flow、test_fem_workflow、test_fem_adaptive_workflow、test_cylinder_workflow |
| 非零初值构造 | test_isotropic_ic |
| A/B 体力、角动量、反转和能量诊断 | test_forced_rotation |

同一机制保留一个代表用例，但不同格式、不同离散后端及不同失败机制仍分别验证。网格收敛所需的多个网格不是重复测试。

Channel 首版新增 `test_channel_operators`（周期布局、壁面算子、伴随和离散旋度）、
`test_channel_model`（独立通量参考、能量配对、平均流和完整 Stokes）、
`test_channel_integration`（四格式固定网格时间阶、连续空间阶、I/PI 与终点）、
`test_channel_workflow`（持久化、复用、独立分析、失败前缀及验证研究校验）。
`test_channel_fft` 对照全局直接解，覆盖奇偶／矩形网格、交错相位、零模式、批量载荷、四格式和自适应。

2026-09-24：46 → 44 个测试实例。谱弱形式与空间收敛仅保留非方形区域，移除重复方形参数；移除重复 G/D 断言及重复反转断言。角动量参考从 4096² 数值求积换为独立解析积分，减少内存占用。全套 44 passed；mypy 40 个源文件通过。求解器及实验参数未改动。

2026-09-27：自适应 campaign 改为 8 个自适应成员加参考解，不再产生 `fixed_step=0.01`；新增四等分输出边界、禁止亚 `min_step` 试步、源码一致性复用和容差推荐规则测试。固定阶梯入口要求基础批次无固定成员，且成员与参考解均完整、同源并通过校验和。
当前全套为 88 passed；mypy 64 个源文件通过。

Shared FEM scalability seams (2026-11-02): `test_fem_mass_solver.py` uses independent
SPD references for CG/METIS, component sharing and full SAV stages;
`test_fem_weak_load.py` checks the conservative normalization and all six formats
on four trace/reaction models; `test_fem_multigrid.py` checks exact mixed solutions,
true-defect correction and all four run/load workflows without PETSc during analysis;
`test_fem_scalability_workflow.py` replays a saved mesh/state and verifies fixed
representation and native-adaptive record/analysis handling.
