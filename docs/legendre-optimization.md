# Legendre 优化实现

2026-10-05：在原单元 Dirichlet-combined Legendre 离散上完成两轮优化。
速度系数仍按 `[x mode,y mode]` 排列，压力使用普通 Legendre 基且常数系数为零。
六种时间格式、反对称对流、求积阶数、lifting 载荷和阶段压力语义沿用原定义。

## 第一轮：默认直接后端

- `quadrature.py`：模型拥有有界 LRU 求积缓存，保存节点、权重、基函数、导数及
  加权投影矩阵。缓存数组只读；公开 Space 方法仍返回独立、可写的节点数组。
  Stokes 的 `size+8` 检查网格和模型的去混叠网格分别保留。
- `basis.py`：直接使用 `phi_k'=-(2k+3)P_{k+1}`、一维质量矩阵的三条非零
  对角线、对角刚度、压力/速度交叉质量及散度闭式。质量对角线仍由精确分数
  累加后转为 float64。
- `mass.py`：两个一维带状 Cholesky 分解实现 `Mx kron My` 的质量逆作用；
  不形成显式逆，不更改系数/弱载荷约定。
- `model.py`：模型与后端共享一次装配的算子，速度及其导数共享张量收缩，
  非线性投影复用权重和乘积。内置固定 LidLifting 的采样按网格缓存，系数或
  空间变化后刷新；自定义 lifting 保留其采样调用。
- `stokes.py`：同一矩阵的多个右端一次提交 SuperLU，逐列保留动量、连续性和
  物理散度检查；只有第一列携带边界载荷。LU 为有界 LRU，系数键保持精确浮点。
  装配、纯分解、三角求解、后端检查及右端/求解次数分别统计。

## 第二轮：可选张量后端

```python
model = SpectralModel(space, nu)  # 默认 direct
model = SpectralModel(space, nu, stokes_backend="tensor",
                      krylov_tolerance=1e-12, max_iterations=200)
```

速度块 `H=m Mv+kappa Kv` 利用一维广义特征分解。空间基底初始化一次，实际
阶段系数只更新 `m+kappa*(lambda_x+lambda_y)` 的倒数。每个压力 CG 迭代使用
真实 Schur 作用 `S=B H^{-1} B^T`，然后重构速度并沿用直接后端的完整检查。
CG 超过迭代上限时显式失败。

预条件器为 `m S0^{-1}+kappa Mp^{-1}`，其中 `S0=B Mv^{-1} B^T`，Mp 为
普通 Legendre 压力质量矩阵。S0 的逆也有可分离结构：令一维 D 为散度块、P
为压力/速度交叉质量，`A=D M^{-1} D^T`、`C=P M^{-1} P^T`，则
`S0=Ax kron Cy+Cx kron Ay`，可做一次一维广义特征分解。仅全局常数模态
置零，输出压力常数系数也显式置零。

这个预条件器近似的是变系数阶段 Schur 的逆；外层 CG 解的仍是原耦合方程。
固定顶盖的非齐次连续性右端、修正列的齐次边界以及物理散度诊断均保留。
迭代容差不是物理空间截断误差估计，物理散度也不等于弱连续性残差。

## 验证与运行

基础机制分别由 `test_spectral_optimization.py`、`test_spectral_tensor.py` 和原
`test_spectral.py` 覆盖：矩形域解析重构、符号积分、已知模态质量载荷、已知
速度/压力/连续性载荷、多右端逐列错误检查、lifting、全部六格式及 I/PI 拒步。
自适应常规步长与终点余量分别比较；终点时刻均严格达到指定值。

`experiments/legendre_optimization` 保存基线和当前源码、完整配置、时刻、末态、
阶段压力与根、诊断、三次交错计时、独立插桩和独立进程内存样本。分析只读取
保存场，使用 NumPy 的独立 Legendre 求值及连续制造解计算物理误差。
第一轮比较原实现与优化 direct；第二轮比较优化 direct 与 tensor。
具体编号、耗时与验证边界见 `docs/validation.md`。

已完成的报告：

- 原实现与优化 direct（本地材料：`../reports/legendre_optimization/20261005T052848-aeb8c268/comparison.md`）
- 优化 direct 与 tensor（本地材料：`../reports/legendre_optimization/20261005T053105-e9d345fc/comparison.md`）
- 优化后 Legendre 与 MAC 的共同误差门槛比较（本地材料：`../reports/legendre_mac_benchmark/20261005T053507-3daa7b53/comparison.md`）

默认保留 direct。后端选择应计入初始化、实际步长变化、压力迭代数及总成本；
少模态、固定步长时 LU 复用通常有优势。变步长下零耦合 LU 分解次数不代表
张量后端在所有规模上更快。
