# Confirmed conventions
2D incompressible velocity-pressure Navier–Stokes on a uniform rectangular MAC grid, homogeneous no-slip walls. u shape (ny,nx+1), v shape (ny+1,nx), pressure shape (ny,nx). Free normal-boundary velocities are eliminated. Scalar r is the paper variable, with Gu=1-r², Gr=1+r and r(0)=0.

V1: IMEX-SDIRK2 and incremental SDIRK2-mr-ccSAV, fixed or explicitly prescribed positive time steps. Numba array kernels, SciPy sparse direct Stokes solves, bounded factorization cache. 32²–128² validation plus rectangular grids. Exponential schemes, adaptive controllers, nonzero velocity walls and checkpoint continuation are outside V1.

Stage pressure is an incremental multiplier, not a demonstrated second-order physical endpoint-pressure approximation. Candidate roots are enumerated and residual-checked; select minimum absolute r, then the smaller signed r in a tie.

最新约定：不使用 nextgen/ 命名；直接在 solver/ 下组织模型、时间推进、MAC 算子及具体格式。使用 uv 管理环境，提交 uv.lock，标准命令为 uv sync --locked 和 uv run。

2026-09-20 可视化扩展：experiments/cavity 支持常速度移动顶盖；法向速度仍为零，黏性边界载荷显式加入阶段右端。原齐次边界实验保持原语义。
