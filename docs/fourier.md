# 周期 Fourier–Leray 求解器

`solver/fourier/` 是五类空间离散之一，通过共同 `Model` 接口复用六种时间格式：
`imex_euler/imex_euler_mrsav`、`sdirk2/sdirk2_mrsav`、`sdirk3/sdirk3_mrsav`。
六格式支持固定/给定步长，四种 SDIRK 格式支持 I/PI 自适应；Euler 尚无自适应估计器。
格式元数据见[注册表](../solver/schemes/registry.py)，控制规则见[自适应说明](../solver/adaptivity/README.md)。
MAC 与 Legendre 的构造、数组、阶段公式和旧实验入口保持原语义。

## 表示与算子

二维双周期矩形域 `(0,lx)×(0,ly)`；节点为 `(j lx/nx,i ly/ny)`，不重复终点。`State.u/v` 均为实数 `(ny,nx)`。内部速度为复数 `(2,ny,nx//2+1)`，压力为 `(ny,nx//2+1)`。`rfft2/irfft2` 使用 `norm="forward"` 和显式逆变换尺寸，支持矩形奇偶网格。

初始化、外力和非线性统一使用严格频带 `|mx|<nx/3, |my|<ny/3`。偶数网格的 Nyquist 模被排除；不能把全频输入仅在乘积之后截断，否则低频别名仍可能存在。实际保留范围及全谱/半谱模数记录在配置中。`to_spectral` 默认过滤；Stokes 和非线性拒绝频带外的内部载荷。

物理波数 `k=(2πmx/lx,2πmy/ly)`，`K=|k|²`，`G=i k`，`D=i k·`。内积为

\[
\langle a,b\rangle_h=lx\,ly\sum_{m_y,m_x\ge0}w_{m_x}
 \operatorname{Re}\bigl(\overline{\hat a}\cdot\hat b\bigr),
\]

其中普通正 `mx` 权重 2，零列与偶数网格 Nyquist 列权重 1；只对 x 使用半谱，y 仍为完整有符号谱。这等于网格物理 L2 求积，`G=-D*`。`max_abs` 是系数最大模，用于阶段相对残差；物理散度单独通过完整逆 FFT 检查。

非线性使用 `1/2[(u·∇)u+div(u⊗u)]`，保留原始梯度部分，产品严格去混叠。频带内实数场满足离散反对称能量配对；散度为零时与通常的 advective Galerkin 对流一致。周期模型的 lifting 配对为零。

## Stokes 与零模

求解 `(mass I+viscosity K)u+G pi=b, Du=0`。非零波数上

\[
\hat\pi=-i(k\cdot\hat b)/|k|^2,\qquad
\hat u=(I-kk^T/|k|^2)\hat b/(mass+viscosity|k|^2).
\]

压力均值固定为零。`mass>0` 时速度均值为 `b0/mass`，因此保留初始均值，并允许平均外力驱动平均速度。`mass=0` 时要求外力零均值（仅允许舍入级残留），固定解的速度均值为零；不相容时明确失败。系数非负且不能同时为零。

`solve_columns` 在同一组波数上批量计算多载荷，不组装稀疏矩阵、不需要 LU 或迭代压力预条件器。每个解检查完整动量残差和完整物理散度。外力及非线性不提前做 Leray 投影，否则压力梯度及 SAV 阶段缺陷会丢失。

求解返回原始谱乘子，时间格式仍按原公式除以 `dt`。`model.stage` 仅把组合后的压力转换为实数场输出；它是 **incremental stage multiplier**，没有证明为高阶物理终点压力。

## 公共使用

```python
import numpy as np
from solver.fourier import FourierSpace, FourierModel
from solver.integrate import integrate, step_sizes
from solver.schemes.sdirk3_mrsav import SDIRK3MRSAV

s = FourierSpace(64, 48)  # 默认 (0,2*pi)^2
load = s.velocity(-np.sin(4*s.y), np.zeros(s.shape), project=False)
model = FourierModel(s, nu=.02, force=lambda t: load)
z0 = s.velocity(.1*np.sin(s.y), .1*np.cos(s.x))
initial = model.state(0., z0, r=0.)
result = integrate(model, SDIRK3MRSAV(gamma=1000.), initial,
                   step_sizes(.1, dt=.001), snapshots=[0., .05, .1])
assert result.status == 'complete', result.error
```

外力回调返回 **已过滤、未投影** 的谱速度载荷；可通过 `s.velocity(u,v,project=False)` 或 `s.to_spectral(stack)` 构造。`s.velocity` 默认投影，适合初值。`model.vector(state)` 返回谱场，`model.state(t,z,r)` 返回实数场。零外力无需提供回调。

`integrate_adaptive` 的用法与既有模型相同；二阶格式设置 `estimator_order=2`，三阶格式为 3，误差范数直接使用上述物理 L2 内积。最后接受节点仍严格等于 `initial.t+T`，终点短步继续执行误差与阶段检查。`ScalarErrorControl` 沿用既有局部缺陷定义，实验入口仅为 SDIRK3-mrSAV 提供联合控制选项；它不代表全局 `|r|` 误差上界。

独立算例、存储、分析及验证入口见 periodic_ns README（本地材料：`../experiments/periodic_ns/README.md`）。长期湍流、高雷诺数、3D、混合边界及 MPI 尚未验证。周期 MAC 同条件对照尚未实现，不能用既有无滑移 MAC 计时推断周期算法加速比。

## 双周期不可压电阻 MHD

同一 Fourier 空间也由 [FourierMHD](../solver/fourier/mhd.py) 使用。动量非线性
以通量散度计算后作 Leray 投影。感应项以反对称通量计算，其散度在精确算术下为零。
阶段算子对速度和磁场分别除以 `1 + tau*nu*k2/4` 和 `1 + tau*eta*k2/4`。
两个响应均检查完整投影残差和两场物理散度；共同 SAV 标量以两场 $L^2$ 内积之和
计算校正，两次校正复用同一对响应。阶段压力已消元，MHD 记录单独保存磁场与磁散度。
固定步 Orszag–Tang 入口、物理诊断及验证范围见实验说明（本地材料：`../experiments/periodic_mhd/README.md`）。
