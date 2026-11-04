# MAC 离散与 SDIRK2 阶段实现

本页给出无滑移 MAC 的空间算子和二阶增量格式推导。当前求解器共实现六种时间格式与
五类空间模型，完整清单见[项目总览](../README.md#当前算法与支持范围)；
一阶公式见 [IMEX Euler](imex-euler.md)，三阶阶段矩阵见
[SDIRK3](../solver/schemes/sdirk3.py)与 [SDIRK3-mr-ccSAV](../solver/schemes/sdirk3_mrsav.py)，
二/三阶嵌入输出及 I/PI 见[自适应说明](../solver/adaptivity/README.md)。

在矩形上用均匀 MAC 网格。u 数组形状 `(ny,nx+1)`，v 为 `(ny+1,nx)`，压力为 `(ny,nx)`；第一个轴是 y。法向边界速度直接为零，切向边界采用关于墙面的奇延拓。自由向量按 u 内面、v 内面顺序拼接，速度内积为 `hx*hy*dot(a,b)`。

## 空间算子与 Stokes 求解

单元散度为面通量差除以网格宽度，记为 D；梯度 G=-Dᵀ；K 为正的离散负向量 Laplacian。切向半网格边界使对应一维 K 边界对角为 3，法向方向对角为 2。K 对称正定。

每个阶段求解

\[
(mI+\kappa K)U+Gq=b,\qquad DU=0.
\]

删去一个压力自由度及一个线性相关的散度约束，求解后把压力平移到零均值，并检查**全部**散度行和完整动量残差。不是压力校正分裂。阶段矩阵中的 q=τπ；输出的 π 是增量压力乘子。

上述消元与稀疏 LU 对应默认 [DirectStokes](../solver/mac/stokes.py)。可选
[TensorStokes](../solver/mac/tensor_stokes.py)用 DST/DCT 求速度移位算子的逆、
预条件 CG 求真实压力 Schur 补，并对原动量与全部散度行重新检查。
它求解同一 MAC 系统，不改变时间格式；实验入口是否开放该后端见各实验说明。

对流使用双控制体中心通量，减去半个双控制体散度乘以当地输运速度，使 `(N(U),U)_h=0` 在舍入误差内成立。切向虚拟值使用奇延拓；法向墙通量为零。测试用独立 NumPy 通量表达式及随机非散度自由场检查 Numba 核和反对称性。

## 增量格式

设 η=1−1/√2，δ=−1/√2，因此 δ−η=−1。对每一步，第一阶段：

\[
R_u=U^n+\eta\tau f(t_n),\quad R_r=r^n,\quad \widetilde N=\eta N(U^n).
\]

第二阶段：

\[
\begin{aligned}
R_u&=U_1-\nu\tau(1-2\eta)KU_1+\tau[-f(t_n)+(1-\delta)f(t_n+\eta\tau)],\\
R_r&=[1-\gamma\tau(1-2\eta)]r_1,\\
\widetilde N&=-N(U^n)+(1-\delta)N(U_1).
\end{aligned}
\]

每阶段先用同一Stokes矩阵分别求右端Rᵤ与Ñ，得到V、W，记
α=(Ñ,V)ₕ、β=(Ñ,W)ₕ≥0，d=1+γητ。存在lifting时α另加同一已知边界功。

令p为格式阶数，P₀(z)=1、P₁(z)=1+z、P₂(z)=1+z+z²。对预测值z，冻结C=P_{p-1}(z)，
阶段方程和消元结果为

\[
U+\nu\eta\tau KU+\tau G\pi=R_u-\tau C(1-r)\widetilde N,\qquad DU=0,
\]
\[
dr=R_r-\tau C(\widetilde N,U)_h,
\qquad U=V-\tau C(1-r)W,
\]
\[
\Phi(z)=\frac{R_r-\tau C\alpha+\tau^2C^2\beta}{d+\tau^2C^2\beta}.
\]

分母严格为正。首阶段预测值为节点r，后续阶段预测值为前阶段r；
一阶与二阶每阶段固定一次更新r^[1]=Phi(r^[0])，三阶每阶段固定两次更新，
额外计算r^[2]=Phi(r^[1])。次数不会因残差已经很小而提前终止。
最终令m=1/1/2，C=P_{p-1}(r^[m-1])、r=r^[m]，统一使用耦合C(1-r)恢复速度和压力：

\[
U=V-\tau C(1-r)W,\qquad
\pi=\pi_V/\tau-C(1-r)\pi_W.
\]

二阶的d采用上述η；三阶采用各阶段D[i,i]，Euler采用1+γτ。
每次Phi只做标量计算，两次校正也仅复用原来的V/W，不新增Stokes求解。
各空间模型通过共同动量adapter保留质量、lifting和开放边界SAV功；
对最终冻结系数的完整动量、标量和全部连续性行重新验收。

生产格式不再构造三次/五次多项式、不枚举实根、不做Newton或按容差追加迭代。
有限次校正与原自洽多项式不同，故冻结标量残差也不是该多项式根残差。
当前Stage记录预测值、校正序列和冻结因子；历史根候选只用于旧数据读取。
详见[ADR0002](adr/0002-fixed-sav-corrections.md)及[保存契约](architecture.md#保存布局与分析约定)。

普通 IMEX-SDIRK2 使用同样的增量系数，直接以 Rᵤ−τÑ 求速度。它不推进辅助变量；输出 r=0 用于统一诊断结构。

修正能量是 `||U||²/2+(r−1)²/2`，辅助变量平衡点仍为 r=0。γr(r−1) 并非总为非负，因此不把无外力修正能量的每步单调性当作一般验收准则。实际记录动能、修正能量、H1 半范数、散度和 r；离散阶段能量抵消另有数值测试。

## 制造解

流函数 ψ=A sin²(πx/Lx) sin²(πy/Ly)e^(−t)，速度 `(ψ_y,−ψ_x)`，压力 `A cos(πx/Lx)cos(πy/Ly)e^(−t)`。从连续公式构造外力，初值以顶点流函数的离散旋度构造，保证离散散度为零；它与连续面采样相差 O(h²)。定常 Stokes 取 t=0 并去掉时间及对流项。空间误差对连续面速度及中心压力计算；时间误差用同网格细时间参考解，避免空间误差掩盖时间阶。

公式来源：Obsidian 系列第 03 篇“mr-ccSAV 扩展与 SDIRK2 格式”和第 04 篇“阶段存在性与条件唯一性”。原 `Stokes_BC.py` 仅作为 MAC 布局参考，新实现独立组装相容矩阵。
