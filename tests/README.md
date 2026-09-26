# 代表性测试

默认运行 `uv run --locked pytest -q`。测试输出和临时实验由 pytest 的 tmp_path 管理，不写入正式 runs/reports。生产实验不能充当单元测试的固定通过条件。

| 机制 | 保留的测试 |
|---|---|
| MAC 算子与边界 | test_model_seam、test_cavity、test_kernels |
| Stokes 求解、压力定标与分解复用 | test_stokes、test_numerics 的空间收敛 |
| 二阶、SAV 能量与实根 | test_numerics、test_schemes |
| 三阶 ODE 精度、MAC 定常态与四阶段工作流 | test_sdirk3 |
| 谱空间、lifting 与去混叠 | test_spectral |
| 时间序列与失败前缀 | test_integrate |
| 保存、复用、失败记录与显式分析 | test_workflow、test_forced_convergence |
| 非零初值构造 | test_isotropic_ic |
| A/B 体力、角动量、反转和能量诊断 | test_forced_rotation |

同一机制保留一个代表用例，但不同格式、不同离散后端及不同失败机制仍分别验证。网格收敛所需的多个网格不是重复测试。

2026-09-24：46 → 44 个测试实例。谱弱形式与空间收敛仅保留非方形区域，移除重复方形参数；移除重复 G/D 断言及重复反转断言。角动量参考从 4096² 数值求积换为独立解析积分，减少内存占用。全套 44 passed；mypy 40 个源文件通过。求解器及实验参数未改动。
