# 一阶 IMEX Euler 与 IMEX Euler-mr-ccSAV

配置名称为 `imex_euler` 和 `imex_euler_mrsav`；Python 类为
`solver.schemes.IMEXEuler`、`solver.schemes.IMEXEulerMRSAV(gamma=1.)`。
两者只依赖现有 Model 接口，返回一个阶段的 Trial，支持固定步长和给定正步长序列。
旧 SDIRK2/3 格式、配置名称、默认比较组及存储容量保持原约定。

## 离散公式

记 $B^n=N(U^n)$、$f^n=f(t_n)$、$H_\tau=I+\nu\tau K$。普通格式为

\[
H_\tau U^{n+1}+\tau\nabla_h p^{n+1}
=U^n+\tau f^n-\tau B^n,\qquad DU^{n+1}=0.
\]

对流及已知外力在旧时刻采样，黏性和不可压约束隐式求解，每步一个 Stokes 右端。
普通格式不推进辅助变量，输出 r=0。

mr-ccSAV 固定采用 $G(r)=1-r$、$Q(r)=1$：

\[
H_\tau U^{n+1}+\tau\nabla_h p^{n+1}
=U^n+\tau f^n-\tau(1-r^{n+1})B^n,
\]
\[
(1+\gamma\tau)r^{n+1}=r^n-\tau(B^n,U^{n+1})_h,
\qquad DU^{n+1}=0.
\]

默认 r0=0。单步接口接受有限的旧 r；周期和 Channel 的现有配置可指定 r0，制造解要求 r0=0。
$\gamma$ 必须有限且非负。固定 lifting 的标量配对加上 `nonlinear_with_lifting`
返回的已知边界做功；以下 $\alpha$ 相应包含该项。非零壁面载荷继续由模型处理，只加入第一个求解列。

## 线性求解与检查

令 $R=U^n+\tau f^n$，同一矩阵求两个右端：

\[
H_\tau V+\nabla_h q_V=R,\qquad H_\tau W+\nabla_h q_W=B^n.
\]

齐次边界下 $DV=DW=0$。定义 $\alpha=(B^n,V)_h$、$\beta=(B^n,W)_h$，则

\[
r^{n+1}=\frac{r^n-\tau\alpha+\tau^2\beta}{1+\gamma\tau+\tau^2\beta},
\quad U^{n+1}=V-\tau(1-r^{n+1})W,
\quad p^{n+1}=q_V/\tau-(1-r^{n+1})q_W.
\]

相容齐次校正问题有 $\beta=(W,H_\tau W)_h\ge0$，分母至少为 1。
仅对舍入量级的负 beta 做归零；显著负值、非有限或非正分母导致阶段失败。
不调用多项式求根器。代入最终场重新检查完整动量、标量和全部代数散度残差，
也单独记录物理空间散度；Legendre 的弱连续性残差与物理散度不混同。

齐次壁面下，以 $U^{n+1}$ 和 $r^{n+1}-1$ 分别测试得到

\[
E^{n+1}-E^n+\tfrac12\|U^{n+1}-U^n\|_h^2
+\tfrac12(r^{n+1}-r^n)^2
+\tau\nu(U^{n+1},KU^{n+1})_h
+\tau\gamma r^{n+1}(r^{n+1}-1)
=\tau(f^n,U^{n+1})_h,
\]

其中 $E^n=\tfrac12\|U^n\|_h^2+\tfrac12(r^n-1)^2$。
非线性项由 $(1-r)+(r-1)=0$ 抵消；阻尼做功并非始终非负，
验收检查这个平衡式，不把修正能量逐步单调作为一般条件。

## 注册、输出及使用

`solver/schemes/registry.py` 提供格式工厂、阶数、阶段数、候选容量、估计器阶数及压力语义。
Euler 阶数和阶段数均为 1，候选容量为 1。HDF5 沿用 schema_version=1：
`stages/*` 为 `(接受步数,1)`，`roots/candidates` 为 `(接受步数,1,1)`；
普通格式 count=0，SAV count=1 并保存唯一线性解。`final/stage_pressure` 第一维为 1。
旧二阶/三阶记录仍为两/四阶段及三/五候选容量，旧数据无需迁移。

后端压力乘子除以实际步长后输出，并保持全局零均值。
新记录标记 `pressure_semantics=euler_step_multiplier_mean_zero`；这是 Euler 步的离散压力，
验证未宣称物理压力收敛阶。失败记录保留已接受前缀，复用及分析继续检查来源和校验和。

MAC、Legendre、Fourier 和 Channel 都可直接使用同一类：

```python
from solver.schemes import IMEXEulerMRSAV
from solver.integrate import integrate

result = integrate(model, IMEXEulerMRSAV(gamma=1.), initial, [.01, .02, .01])
```

公开 Channel/Fourier 配置入口支持两个名称，示例见
[一阶验证与示例配置](../experiments/imex_euler/README.md)。Euler 的 `embedded_velocity=None`。
本次支持固定/给定步长；Channel/Fourier 在配置校验时拒绝新格式的嵌入自适应请求。
后续 step doubling 需要独立估计器，不能借用旧 SDIRK 的嵌入阶数。

## 验证范围

测试覆盖独立耦合线性系统、backward Euler 模态与标量衰减、非线性 ODE 一阶收敛、
完整约束失效、六后端纯梯度压力、固定 lifting、单阶段保存/读取/复用/只读分析及失败前缀。
周期与 Channel 的非线性时间阶测试扩展到 Euler；自适应测试只用于四套旧嵌入格式。
六后端细化验证保存最终场、解析时间参照及内积权重/质量矩阵，分析不构造模型也不启动积分。
实际结果及局限记在 [validation.md](validation.md)。
