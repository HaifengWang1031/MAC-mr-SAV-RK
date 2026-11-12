## 当前能力索引（2026-11-04 文档核对）

2026-11-12 用户授权先修复主分支审查的三个问题，不合并或推送：接受态的下一步长/PI续算
上下文先于on_accept提交；MHD合法空压力范数为零，并以两模型/六格式零解检查接口；纯
bounded-history/联合PI续算测试从本地tasks移入core，函数体/阈值不变，新增接受后回调失败回归。
迁移前已单独外盘备份并独立恢复该本地测试；旧的412文件全字节保持声明对应83a043d本地化
快照，不适用于此次获准的单文件测试迁移。原工作区、旧运行/冻结源码和研究链/retarget限制
不改，新solver源码如实改变新来源，不修旧checkpoint，也不授予生产恢复或长期统计资格。
见docs/spec.md、docs/architecture.md与docs/validation.md。

2026-11-12 用户确认第四阶段本地化清单，外盘为 ~/GameStation（/Volumes/GameStation）。
两工作区的本地文件/Git 已独立备份和恢复校验，再按基线清单仅取消研究文件的 Git 跟踪，
保持本机原路径与字节；不改数值实现、旧记录、冻结源码或远端作业。公共 solver、
run_support、postprocess、核心测试/参照和必要文档仍随检出提供，研究包/任务测试/五个
计算工具与五份研究资料仅本地保留。默认 pytest 仍收集所有实际存在的测试；默认 mypy
仅检查公共目标，完整本地检查显式包含 experiments/tools。legacy 来源仍看磁盘实际依赖，
不按跟踪状态删文件；pyproject 配置变动如实改变新来源，不回填旧身份或放宽 retarget。
备份范围、原生 macOS 安全属性差异、恢复验证及非服务器备份限制见 docs/core-localization-plan.md。
下文研究路径均为可选本地材料；日期化科学资格/撤回记录及当前 ADR 约束继续有效。

2026-11-12 用户继续批准第三阶段通用只读工具分离：公共 MAC 保存场读取、图/排版、诊断和快照报告
归属 postprocess，experiments.plotting/analysis 和旧快照命令保留 adapter；研究绘图策略、
benchmark/probe/search 仍属任务计算侧，不统一搬迁。新报告显式记录分析实现来源，
旧广泛来源清单追加 postprocess 防漏依赖；旧记录/冻结源码不改。核心分发加入该公共
module，仍不包含或导入 tools/experiments；独立副本实际运行 python -m postprocess.snapshots，
禁止研究与数值计算模块导入，验证输入不变。本阶段不改数值公式、gitignore 或跟踪范围。

2026-11-11 用户继续批准第二阶段核心验证/任务测试分离：tests/core 不依赖
experiments、tools 或 tests/tasks；共享 ODE 与解析/独立离散参照归属 tests/references。
混合文件按机制拆分，任务测试全部保留在 tests/tasks；默认 pytest 仍覆盖全部现存测试。
用 python -m tests.core_distribution 在无任务目录、禁止相关导入的临时副本实际运行核心套件。
不改 solver、实验/工具 Python 实现、记录来源策略、旧数据或 gitignore；工具分离仍待后续。

2026-11-11 用户批准在独立优化分支先做运行记录与来源抽取：公共保存/读取归属
run_support，新增仅核心与显式任务清单的版本化来源；periodic_mhd 先行采用。
旧 experiments.records 为兼容导出，未迁移入口仍广泛扫描并纳入新公共实现；
旧记录身份和冻结源码不回写。本轮不删除实验/测试、不改 gitignore 或数值公式，
并同步检查新快照和数值身份不能遗漏 run_support。见 docs/architecture.md。

**2026-11-10 审计覆盖旧完成声明**：分析升 schema 3，修正 Geyer IPS 换算；旧 schema 2
τ/ESS 及资格结论撤回。旧 `reports/inverse_cascade/plan/configs` 高尺度配置不可执行。
当前 plan 明确各档 kf/Re-，执行须绑定有效配置/代码/证据的人工审批；37 条仍仅部分候选。
chain schema 2 拒绝 failed 续算，校验全部父段/元数据/采样相位/全场连续性；墙钟仅段边界暂停。
详见 `docs/inverse-cascade-toolchain-audit.md`。本轮只做本地验证，不连接服务器、不启动生产计算。

2026-11-07 用户要求用现有 SAV 复现 Seshasayanan 等（2014）的逆级联转变。
独立 experiments/inverse_cascade 使用论文双场定常强迫及相同的四阶／逆四阶耗散，
复用 FourierMHD 和共享普通／两次固定校正 SDIRK3；不改变旧衰减 MHD。
保存实际 E/Q 注入、大小尺度耗散、原 PDE 通量谱和标量校正，只读分析显式选记录、
启动剔除时间和块长。降低 Re 的试算与 kf=8/512² 网格分开；后者只做到 T*=100–300
（原文同组 T*=2000）。已验证：kf=8 下 192²–512² 网格与 dt*=0.005 一致、128² 不足；
能量逆通量在 mu=0.20 与 0.25 之间穿零（256² 八点）并被 512² 的 0.20/0.30 两点夹住；
种子、窗口、块长与 Re⁺/Re⁻ 依赖已检。2026-11-08 收紧：分析升到 schema 2，
原始／注入归一化通量、大尺度耗散占总耗散与大尺度耗散占注入分列，零分母输出 null；
新增积分相关时间、有效样本量与预登记稳态筛查，现有 256² 八点全部不通过，
故只能称有限窗趋势。长时统计改用分段 chain（原子 checkpoint + 显式续算校验），
分段与连续运行逐项一致。不意味着已验证临界值／指数、统计平衡或空间收敛
（储能/注入仍为 −0.05 至 −0.6）。另：SAV 标量阻尼 gamma 必须足够大；实测
gamma=1 在 kf=8 与早期 96²/kf=4 试算上都令 G 塌到 0 并压小非线性，那些记录按
max|G−1|>1e-3 判废。见实验 README 与 docs/validation.md。

2026-11-04 用户确认：共享mr-ccSAV阶段以固定次数冻结因子校正替代多项式求根及Newton。
一阶、二阶每阶段各一次，三阶每阶段两次；以先前阶段r为预测值。
最终耦合为P_{p-1}(r_penultimate)*(1-r_final)，速度、压力、lifting/SAV功及残差使用同一因子。
每阶段仍为两个同矩阵Stokes右端；不额外求解Stokes，也不按残差提前终止或增加校正次数。
新记录明确保存校正值与冻结因子，历史根候选只作兼容读取；旧运行来源与文件不重写。
这条用户决定取代下文历史记录中的实根枚举/最小模根选择，见docs/adr/0002-fixed-sav-corrections.md。

当前实现六种共享时间格式：IMEX Euler、SDIRK2、SDIRK3 各有普通版与 mr-ccSAV 版，
分别为一/二/三阶、单/两/四阶段。五类空间离散为无滑移 MAC、Legendre、双周期 Fourier、
Channel MAC 和 Taylor–Hood P2/P1 FEM。六格式支持固定/给定步长；四种 SDIRK 嵌入格式
支持 I/PI，自适应标量控制与可选后端由各实验入口限定。Euler 尚无自适应估计器。
完整清单见 [README](README.md#当前算法与支持范围)，模块归属见
[architecture](docs/architecture.md)，控制规则见[自适应说明](solver/adaptivity/README.md)。

下文保留历次确认记录；其中 V1、首期范围、已移除并行实现及旧验证数字按其日期解释，
不作为当前能力清单。当前 PETSc 选项是单进程 FEM 后端，不代表恢复已删除的 MAC MPI 实现。

2026-11-02 大规模FEM对照优化：新增显式petsc后端（COMM_SELF、FGMRES、速度块与压力辅助GAMG），
压力辅助逆区分质量/黏性/反应系数；外层保留精确原矩阵、全部连续性行及原压力规范。
可选mass_backend=cg仍求解一致质量；fem_load_mode=weak只改变动量载荷表示，共用六格式阶段及SAV根选择。
弱载荷残差分子仍用一致质量反解，分母以下界作更严格检查，不以对角残差代替原系数残差。
全局LU回退显式计数；默认direct/Riesz、旧记录身份及运行中的冻结源码Re10000不改。
已完成runs/fem_multigrid/20261102T083710的296样本对照：常规选项recycled+weak有实测收益，原默认不改。
大网格另提供mass_backend=metis的一致质量精确重排因子及CG；各批次适用范围、压力未合格AMG样本和限时失败均保留。
完整测试621/21(skip)与632/10(skip)，受影响71源文件mypy通过；见docs/fem-multigrid.md及final/summary.md。

2026-11-02 用户授权修复Hele-Shaw入口角点并保留SAV重跑Re_D=10000至T=20。
显式energy_stable_pressure入口在动量与SAV边界功中加入同一方向性动能应力，
默认natural及旧记录不改；生产使用现有回流耗散出口。新参数和边界标签另存，
先检查旧失控时刻，再从零重跑；单步修复不等于T=20已完成或高Re物理验证。

# Confirmed conventions
2026-11-01 用户批准Hele–Shaw翼型规划：新增独立压力驱动MWR惯性Brinkman模型，
闭合系数(6/5,54/35,6/5,12)，归一化后beta=9/7、sigma=10nu/b²、pi=p/(1.2rho)。
可选FEM反应项默认零，隐式一致质量阻力与所有lifting/阶段检查同步；旧模型矩阵零分支不改。
Gmsh仅为可选mesh生成依赖，读取真实保存网格。独立标量P2 Darcy参照不采用SAV。
首版两端压力型自然应力、翼型/侧壁零速度；压力功/对流通量/SAV配对及可选出口耗散分开。
物理快照压力在计算阶段独立恢复，模型力不冒充三维实验力；见docs/hele-shaw.md。

2026-11-01 顶盖楔腔修正与动画：独立 experiments/wedge_cavity 使用既有P2/P1与SDIRK2-mr-ccSAV。
新运行显式r0=0，顶盖两端归静止斜壁，非线性返回同一反对称求积的b(U,U,g)，
而不是用零法向通量代替lifting体积配对。旧r=1/顶盖端点记录保持原文件与legacy读取约定。
固定步末态和最后快照严格对齐T；动画只读显式完整记录，复用版式并直接编码，
阶段压力不在动画中当作物理端点压力使用。正确性修正与性能对照分开；见experiments/wedge_cavity/README.md。

2026-10-30 用户设定有限元，尤其变步长效率优化目标。现有recycled后端先对同一精确新矩阵
作多右端批量缺陷校正，停滞或失败回退原GMRES，再回退带完整检查的新LU。
对角质量缩放只筛选何时做完整检查，不代替一致质量系数残差、全部连续性行或压力规范。
多右端线性响应的物理散度直接收缩所需对角导数，仍覆盖全部求积点；
单右端保留完整场缓存供普通格式下一阶段复用；阶段/接受态诊断不删减。
SuperLU新增auto/mmd/metis排序，auto对不少于20000未知量的系统使用缓存的结构嵌套剖分。
所有分量置换还原后作原检查；无耦合同质量速度分量共用一个精确标量因子。
可选Linux x86_64 PARDISO、局部线程数、两次内部改进与SuperLU回退已通过85项原生验证。
这些新配置键只在显式提供时保存；旧身份不补默认键。派生自适应参考保留出口及因子选项。
已有来源与旧记录保留，性能以冻结基线的同状态/步长回放和原生自适应对照为准。
本轮实现与预定跨规模/格式验证完成，EIT-WSL真实网格PI积分中位加速3.274倍。
最优限于本轮测试候选与条件，非全局证明；详见 docs/fem-optimization.md 及对应final报告。

2026-10-28 用户要求继续测试可选回流耗散出口：outlet_treatment=backflow_damped
只在开放边界增加 .5*(U.n)^-U 的弱载荷，并同步调整 SAV 边界自功。
原自然出口默认和旧记录身份保持不变。新 case/边界/压力标签明确区分，
不继承原 DFG 自然出口参考资格；原物理通量和附加耗散分开记录。见 docs/backflow-outlet.md。

2026-10-28 强制最小步长：用户要求 Re1000 圆柱重跑在自适应建议低于下限时强制使用下限。
新增显式 min_step_policy=force：普通建议截断至 min_step，在该步长上允许有限的超容差试步被接受，
原始误差和接受标志完整保存并统计；非有限状态/误差、阶段求解/诊断失败不接受。
终点的短对齐步仍保留原误差检查。默认 fail 和所有未显式启用的旧实验保持原行为。
此策略只保证继续试算，不代表满足原局部容差或取得新的稳定性/精度资格；旧来源和失败记录保持不变。

2026-10-28 圆柱与自适应 FEM：在既有六格式和模型接缝下实现 DFG2D-2 Re100，固定抛物线入口、无滑移圆柱/壁面及梯度应力自然出口；开放压力不施加零均值。P2/P1、一致质量、反对称体积对流加边界通量、SAV 边界工作项见 ADR0001。四种嵌入格式沿用 I/PI 与可选标量控制；旧封闭 FEM 规范和固定记录保持兼容。实际网格、阶段增量压力和独立瞬时物理压力分开保存；独立空间/几何/时间及性能证据见 experiments/cylinder_ns/README.md、docs/validation.md，不声明新的开放边界能量定理或PDE时间阶证明。

2026-10-27 FEM 首期范围：用户批准独立 scikit-fem Taylor–Hood P2/P1 矩形三角网格 Model，复用六格式固定/给定步。齐次无滑移、一致质量、反对称弱对流、无压力罚项及物理积分零均值；完整弱连续性与 FE 梯度物理散度分开记录。专属 fem_ns 保存全 P2/P1 系数及网格/DOF 映射，支持显式运行编号加载重建和只读分析，不改旧模型/记录。当时不含自适应、非齐次边界或 PDE 时间阶证明；后续扩展见上条。见 docs/fem.md、docs/validation.md。

2026-10-05 Legendre 优化：用户批准按规划实施。求积/基函数缓存、精确一维闭式、张量质量逆、共享算子、非线性收缩复用和多右端 SuperLU 已接入；保留原空间与六格式语义、完整检查。新增可选 stokes_backend=tensor：速度广义特征分解加真实压力 Schur-CG，采用可分离质量 Schur/压力质量组合预条件器。默认 direct，按实测规模和步长选择；与 MAC/Channel/Fourier 路径独立。详见 docs/legendre-optimization.md。
2026-10-05 Channel 失稳可视化：用户要求看到失稳。独立 ts_wave 算例以中心速度和半高定义 Re，使用离散 Poiseuille 加基于同一 MAC 线性化的 TS 特征模态；epsilon 表示初始格心速度扰动峰值/Umax。原有算例参数语义不变。普通 SDIRK3/FFT 实际推进，独立连续 Orr–Sommerfeld 参照、短时步长/网格加密和低 Re 对照先核验增长，再生成非线性动画。TS 动画使用初始波速的移动坐标和相对速度流线，速度颜色仍为实验室系量；不等同于三维湍流。

2026-10-04 一阶扩展：用户确认经典 IMEX Euler 与 IMEX Euler-mr-ccSAV，新增 scheme=imex_euler/imex_euler_mrsav；显式旧时刻对流/外力，隐式黏性与压力，单阶段。SAV 固定 G=1-r、Q=1，两个同矩阵右端后直接求唯一线性标量解；默认 r0=0、gamma>=0。复用四模型接口，固定/给定步长及公开 MAC/周期/Channel 运行保存分析。注册表提供阶段数/容量/阶数，单阶段 HDF5 容量1，Euler 压力标记 euler_step_multiplier_mean_zero。旧四格式及默认研究组不变；本次不含一阶自适应，后续需独立 step doubling。见 docs/imex-euler.md。
2026-10-04 Channel：用户确认纯 MAC 的流向周期、上下壁面 u=v=0 模型。`solver/channel` 独立实现，u 保留 `(ny,nx+1)` 重复周期末列，仅 nx 列参与自由度和内积；v 为 `(ny+1,nx)` 且壁面行零。复用 DirectStokes（仅输入类型改为公共算子协议）、四格式和 I/PI；允许非零平均流与定常均值驱动力，压力仅全局零均值。专属 `experiments/channel_ns` 管理运行与只读分析。按规划先完成直接后端，再增加同一二阶 MAC 离散的可选 FFT 带状后端：保留全部模式和交错差分相位，零波数仅固定压力均值。默认 backend=direct，可选 fft。见 docs/channel.md；旧 MAC/Legendre/Fourier 语义及调用方式保持不变。
2026-10-04 周期扩展：用户确认在保持既有求解器使用方式的前提下实现二维双周期 Fourier–Leray 后端。`solver/fourier` 独立实现，复用四套时间格式和 I/PI 控制；公开 State/Stage 为实数 collocated `(ny,nx)` 场，内部为 forward-normalized rFFT 系数，严格 2/3 频带、压力零均值、瞬态保留速度均值、定常零均值兼容性检查。`experiments/periodic_ns` 独立计算/存储/分析及短时验证，不扩展旧 MAC workflow 的布局假设。详见 docs/fourier.md；原 MAC/Legendre 数值公式、数组及旧实验参数不改。

2026-10-04 自适应终点对齐：按用户要求，最后一步取 `initial.t+T-current.t`，成功接受后状态、时间记录、终点快照及回调均严格使用 `initial.t+T`。这个终点对齐步允许小于 `min_step`，包括浮点舍入级余量；仍执行原有阶段求解与速度/标量误差检验。普通步及内部严格输出节点的最小步长规则不改；此前“任何实际试步不得小于 min_step”的约定仅对终点对齐增加这一例外。

2026-10-04 固定对照补充：用户要求在既有256²、T=30的正式自适应六联图中增加tau=0.01/0.005/0.0025，沿用三次重复和原物理参数。已有18个成员及共同1e-4/5e-5参考保持不变，通过显式派生campaign增加9个新成员；数值执行使用原23ec5c78源码快照，补算入口单独记录校验和。新增步长均整除0.1输出间隔，旧参考覆盖新采样时刻；不重新计算参考。图扩展为一组自适应加八组固定对照，保留原曲线颜色/标记和最细固定步长参考核验未过的标记。

2026-10-02 EIT-WSL 自适应/固定对照：独立 adaptive_fixed_256.json 使用 MAC 256²、T=30，沿用 Kolmogorov 参数 nu=0.02、f=(-sin(4y),0)、epsilon=4、initial_modes=10、gamma=1000；atol_u=1e-8、rtol_u=2e-6、atol_r=1e-4，初步1e-5、min_step=1e-12、max_step=无穷（JSON null）。联合局部指标及分量控制阶数沿用已确认定义，I safety=0.9、倍率[0.5,1.2]。五组同格式固定对照tau=0.1/0.05/0.025/0.001/0.0005；三次交错单线程重复。自适应输出不截断自然步长，保存0.1间隔请求的最近自然节点及实际时刻，共同普通SDIRK3参考1e-4/5e-5访问实际采样时刻并核验5%敏感性。六联图和误差统计来自保存记录；失败前缀明确标识。T=0.1、单次重复的pilot只验证256²运行/存储/绘图流程，不代表T=30结果。此批次不使用独立固定收敛算例的nu=0.001与f=(0,sin x)。

2026-10-02 自适应标量控制：用户选定 SDIRK3-mrSAV 的联合指标 E=max(E_u,E_r)，其中速度沿用嵌入差及 atol_u+rtol_u*max(||u_old||_h,||u_trial||_h)，E_r=|r_trial-exp(-gamma*h)r_old|/atol_r。辅助变量为一阶更新，局部缺陷为 O(h²)，速度嵌入差为 O(h³)；两项分别产生步长建议并取较小值，PI 分别保留接受步误差历史。通过显式 ScalarErrorControl 启用，既有速度单独控制的默认行为保留。新容差搜索使用独立 joint_tolerance_scan.json，保持既有128² Kolmogorov物理参数，不使用独立256²固定收敛实验的初值或外力；固定绝对局部容差不等于全局 |r| 上界。正式扫描为T=30、三次交错单线程重复，共同细步参考需通过5%敏感性检查；短时pilot只验证流程和耗时。

2D incompressible velocity-pressure Navier–Stokes on a uniform rectangular MAC grid, homogeneous no-slip walls. u shape (ny,nx+1), v shape (ny+1,nx), pressure shape (ny,nx). Free normal-boundary velocities are eliminated. Scalar r is the paper variable, with Gu=1-r², Gr=1+r and r(0)=0.

V1: IMEX-SDIRK2 and incremental SDIRK2-mr-ccSAV, fixed or explicitly prescribed positive time steps. Numba array kernels, SciPy sparse direct Stokes solves, bounded factorization cache. 32²–128² validation plus rectangular grids. Exponential schemes, adaptive controllers, nonzero velocity walls and checkpoint continuation are outside V1.

Stage pressure is an incremental multiplier, not a demonstrated second-order physical endpoint-pressure approximation. Candidate roots are enumerated and residual-checked; select minimum absolute r, then the smaller signed r in a tie.

最新约定：不使用 nextgen/ 命名；直接在 solver/ 下组织模型、时间推进、MAC 算子及具体格式。使用 uv 管理环境，提交 uv.lock，标准命令为 uv sync --locked 和 uv run。

2026-09-20 可视化扩展：experiments/cavity 支持常速度移动顶盖；法向速度仍为零，黏性边界载荷显式加入阶段右端。原齐次边界实验保持原语义。

2026-09-21 并行扩展：solver/mac_parallel/ 提供同一 MAC 离散的分布式实现（PETSc/MPI，沿 y 分条带，作为可选 extra `mpi`，不装入时不导入 petsc4py）。沿用同一套 IMEX-SDIRK2 / SDIRK2-mr-ccSAV 阶段公式、同一压力定标与全散度检查；外层 FGMRES 或 BiCGStab，速度块与压力块各用 AMG 近似，停止判据同时要求动量残差与散度。串行路径与 V1 语义不变；该后端把散度解到迭代容差级而非机器零，超过 4 进程、二维域分解与多节点均未验证。

2026-09-23 受迫算例语义变更：experiments/forced_ns_convergence 的体力改为 f=(0, amplitude*sin(2*pi*m*x))（m 为跨越单位区间的完整波数，正整数），初值由恒零改为各向同性无滑移夹持梁流函数（experiments/forced_ns_convergence/isotropic.py，参数 ic_k_lo/ic_k_hi/ic_alpha/ic_seed/ic_target/ic_value/ic_symmetric/ic_project）。同一批次内所有步长级别与参考运行必须共用同一 ic_seed。频带受 k_hi <= min(nx,ny)/8 限制；nu=0.01 下 K∈[4,12] 的黏性寿命约 0.03–0.04，该初值只影响 T≲0.1 的瞬态。并行后端 solver/mac_parallel/ 仍用其自带 initial(amplitude)，未接入该初值。

2026-09-23 移除实验种类与并行后端：删除 experiments/{stokes_mms,ns_mms,decay,mac_parallel}、solver/mac_parallel/、experiments/convergence.py、tools/validate.py，以及 pyproject 的 `mpi` extra（mpi4py/petsc4py 已从 uv.lock 去掉）。workflow 只接受 cavity/forced_ns/trig_ns；cli 只保留 cavity 的批次入口；模型接缝的 realized 实现改为 MAC 与 spectral 两者。制造解字段 experiments/problems.py 保留，仅作测试与步长计时的独立参照。并行相关的历史约定与验收数字只存在于 git 历史，见 docs/validation.md。

2026-09-23 参数组确认：受迫方腔算例取 Omega=(0,1)^2、四周齐次无滑移、u0=0（ic_kind="zero"）、nu=0.01、F=1、m=2（f=(0,sin(4*pi*x))）、r0=0、gamma=1、MAC 128^2；production.json 与 smoke.json 已按此重写。初值两种：ic_kind ∈ {zero, isotropic_beams}，默认 zero。实测该参数组在 T≲3 即定常，故收敛实验的观测窗口 [1,1.5,2] 落在定常段，测的是定常残差而不是瞬态误差；要测瞬态需把窗口移到 T≲1，此项尚未改。

2026-09-23 A/B 旋转实验：新增 experiments/forced_rotation（experiment kind `rotation_ns`），外力 f_A=0.1(sin5y,-sin5x)（净力矩 2f0/k=0.04）与 f_B=0.1(cos5y,-cos5x)（零净力矩），域 (0,2π)²、零初值、r0=0。模型诊断新增 angular_momentum（域心、格心二阶求积）、power=(f,u)、dissipation=ν|∇u|²；配置新增 snapshot_every 输出节奏；cli.run_batch 不再维护自己的实验白名单，改由 workflow 校验。首轮探索参数 ν=1e-3、256²，长期配置 T=3000（τ=0.005，256² 约 21 h/次，无断点续算）。

2026-09-26 三阶格式扩展：依据 Obsidian `数值分析/SDIRK3-mr-ccSAV 无滑移NS/02,03,04,15`，solver/schemes 新增四阶段普通 SDIRK3 和增量 SDIRK3-mr-ccSAV。普通格式令 G=1，mrSAV 采用 G=1-r³、Q=1+r+r²、五次实根选择；r0=0。两者支持 MAC 与 spectral model seam、固定步长和给定序列。workflow 支持 scheme=sdirk3/sdirk3_mrsav，HDF5 对三阶格式存四阶段及每阶段最多五个候选实根。受迫收敛 campaign 可通过 schemes=["sdirk3","sdirk3_mrsav"] 选择第三阶对照组，默认二阶组不变。这里实现的是笔记增量格式；三阶 PDE 收敛取决于笔记所列附加正则性条件，ODE 验证不能替代该证明。

2026-09-27 自适应实验重置：删除旧 `runs/reports/kolmogorov_adaptive` 批次；基础 campaign 只保留 4 格式 × I/PI 的 8 个自适应成员和 SDIRK3 细步参考，删除已知无效的 `τ=0.01` 固定对照。固定阶梯独立使用 (0.0025,0.001,0.0005)。批次复用及两类分析均校验配置、源码哈希、完成状态和结果校验和；JIT 预热不计入积分 CPU 时间。严格输出对齐若将留下小于 `min_step` 的余量，则把到输出点的距离分成四个相等试步，浮点舍入余量直接吸收，任何实际试步不得小于 `min_step`。参考步长/误差指标与 mrSAV 标量控制暂不改。

2026-09-27 容差扫描：128²、T=0.1、单线程、3 次交错重复，扫描 rtol∈{1e-3,5e-4,2e-4,1e-4,5e-5}，保持 atol/rtol=2e-4、SDIRK3 参考步 1e-4，以输出速度最大相对误差≤1e-4 为门槛。120 个自适应计时全部完成，最小试步为 1e-5。八组统一可用容差仍为 rtol=5e-5、atol=1e-8，故 production 不改；若允许各组异容差，最快可用档的聚合 CPU 中位数比统一 5e-5 低 16.7%，但不用于公平同容差比较。原始记录见 `reports/timings/20260927T141542-82e76f6e-a87d30ee-adaptive-tolerance-scan/timings.json`。

2026-10-02 三阶无滑移受迫收敛参数更新：独立配置 `experiments/no_slip_reference_convergence/configs/sdirk3_256_gamma1000.json` 只比较 SDIRK3 与 SDIRK3-mr-ccSAV，域 (0,2π)²、MAC 256²、nu=0.001、f=(0,sin x)、gamma=1000、r0=0、T=4、输出 0.5/1/2/4；tau=0.1*2^(-k)，k=0,...,8 且无小数档。共同普通 SDIRK3 参考 k=10，以 k=11 做5%参考敏感性检查。初值使用用户指定的 4 sin²(x/2) sin²(y/2) 加权整格点流函数和，包含 0<k1²+k2²<=100 的全部316个有符号模式；顶点流函数离散旋度生成 MAC 速度，无随机种子、归一化或额外投影。保留稀疏直接 Stokes 后端及1e-9残差阈值。既有旧配置与结果不改；新初值和外力只通过显式配置选择。固定网格误差表及参考敏感性检查不能替代空间收敛或无滑移 PDE 三阶证明。

2026-10-05 新的 H-valued-force 整数收敛批次：f=((1-cos(x))*sin(y), -sin(x)*(1-cos(y)))，C=1，force_profile=clamped_streamfunction；四种 SDIRK2/3 方法，k=0,...,8。沿用 MAC 256²、nu=0.001、gamma=1000、r0=0、用户原 lattice 初值、T=4 和 0.5/1/2/4 输出、tau=0.1*2^(-k)、直接 Stokes 1e-9。共同普通 SDIRK3 参考 k=10/11 随新外力重新计算；保留旧结果，待整数结果后再选小数加密区间。生产仅在 MAC MINI 上运行，远端原有时间格式未更新为本地其他扩展。
