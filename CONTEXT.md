# Confirmed conventions
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
