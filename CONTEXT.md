# Confirmed conventions
2D incompressible velocity-pressure Navier–Stokes on a uniform rectangular MAC grid, homogeneous no-slip walls. u shape (ny,nx+1), v shape (ny+1,nx), pressure shape (ny,nx). Free normal-boundary velocities are eliminated. Scalar r is the paper variable, with Gu=1-r², Gr=1+r and r(0)=0.

V1: IMEX-SDIRK2 and incremental SDIRK2-mr-ccSAV, fixed or explicitly prescribed positive time steps. Numba array kernels, SciPy sparse direct Stokes solves, bounded factorization cache. 32²–128² validation plus rectangular grids. Exponential schemes, adaptive controllers, nonzero velocity walls and checkpoint continuation are outside V1.

Stage pressure is an incremental multiplier, not a demonstrated second-order physical endpoint-pressure approximation. Candidate roots are enumerated and residual-checked; select minimum absolute r, then the smaller signed r in a tie.

最新约定：不使用 nextgen/ 命名；直接在 solver/ 下组织模型、时间推进、MAC 算子及具体格式。使用 uv 管理环境，提交 uv.lock，标准命令为 uv sync --locked 和 uv run。

2026-09-20 可视化扩展：experiments/cavity 支持常速度移动顶盖；法向速度仍为零，黏性边界载荷显式加入阶段右端。原齐次边界实验保持原语义。

2026-09-21 并行扩展：solver/mac_parallel/ 提供同一 MAC 离散的分布式实现（PETSc/MPI，沿 y 分条带，作为可选 extra `mpi`，不装入时不导入 petsc4py）。沿用同一套 IMEX-SDIRK2 / SDIRK2-mr-ccSAV 阶段公式、同一压力定标与全散度检查；外层 FGMRES 或 BiCGStab，速度块与压力块各用 AMG 近似，停止判据同时要求动量残差与散度。串行路径与 V1 语义不变；该后端把散度解到迭代容差级而非机器零，超过 4 进程、二维域分解与多节点均未验证。
