# Legendre 第一轮优化验证

第二轮可用 `--config experiments/legendre_optimization/configs/tensor.json`，
其基线应先捕获当前优化 direct 源码，例如使用 `capture_source(Path(...))`。
第二轮比较 8/32/64 模态的 direct 与 tensor；其余制造解与计时约定一致。

目的：在同一离散方程、时间格式、连续制造解和给定时间序列下，比较优化前后
Legendre 后端的数值结果和初始化加完整积分的总成本。对照源码在修改前捕获，
计算记录再次保存两版源码及校验和；公共时间格式、推进器和模型接缝共用当前代码。

配置集中在 `configs/first_round.json`：矩形 `(1.3,0.8)`、nu=0.1、振幅0.1，
8/16/32模态，全部六种时间格式，10步、固定步长0.002及等总时间给定变步长。
三次交错重复，单数值线程。连续制造解与原 Legendre/MAC 比较一致。

计算：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
uv run python -m experiments.legendre_optimization.run \
  --baseline runs/legendre_optimization/baseline-20261005T045900/spectral
```

分析只接受显式运行编号，重读保存系数，以 NumPy 的独立 Legendre 求值重构物理场，
同时核对速度、零均值阶段压力、辅助变量、根候选、时刻和诊断：

```sh
uv run python -m experiments.legendre_optimization.analyze runs/legendre_optimization/<run-id>
```

主要计时是 model/初值/外力初始化加公开 integrate 的完整推进，包含首次分解、
每次求解后的完整检查和每步诊断。导入、符号制造、物理误差重构、存储、热调用和
插桩积分不计入主要计时。`profiles.json` 单独保存函数调用及累计/自身墙钟耗时，
累计项互相包含，不能相加当作主要计时比例。优化版另存装配、纯LU、三角求解、
后端检查耗时；旧版保留原有装配加LU计时，其他成本通过独立插桩定位。

内存样本在两个独立进程中测量最大模态的 SDIRK3-mr-ccSAV/给定变步长成员；
RSS包含解释器、导入、初始化、推进及热调用。另存结束时缓存中的 LU 与耦合矩阵
数组字节数，不将其称为整个模型内存。日志逐成员记录进度，异常保留失败状态与
已完成记录。分析要求完整记录和全部校验和匹配。

本实验只验证该短时光滑制造解和当前环境。给定变步长不包含控制器和拒步成本，
阶段乘子不是物理终点压力精度验证。求积敏感性门槛2%，速度相对差门槛1e-10；
阶段压力使用 `||delta p||_L2 <= 1e-12 + 1e-10 ||p_baseline||_L2`，同时保留
未经缩放的绝对和相对差。接近零的增量乘子相对误差可能放大舍入差，绝对门槛
只用于版本等价检查，原求解器方程残差门槛保留。
基准通过不替代高Re或有角点奇异性的空间收敛研究。
