# Hele–Shaw翼型流动及SDIRK3-mr-ccSAV离散

本文对应2026-11-02入口角点修正后的实现：板间平均的MWR惯性Brinkman模型，
三角形P2/P1有限元，四阶段增量SDIRK3-mr-ccSAV。
正式Re_D=10000成员使用带动能修正的压力入口和已有回流耗散出口。
前置T=1.1成员保持自然出口，用于单独检查入口修正。
下文的“当前参数”指正式T=20成员，不表示长算已经完成。

## 1. 区域、未知量及薄缝近似

设弦长为c，完整板间距为b，平均平面速度为U=(U_x,U_y)，物理压力为p。
平面区域为

\[
 \Omega=(-3c,6c)\times(-2c,2c)\setminus\overline{\mathcal A_\alpha},
\]

其中A_alpha为闭尾缘NACA0012翼型，绕四分之一弦点取攻角alpha=10度。
代码中的正攻角表示机头上仰，即由前缘指向尾缘的方向顺时针旋转alpha。
当前几何用120点多边形近似；P2速度插值不改变几何近似的阶数。

边界分为入口Gamma_in、出口Gamma_out、侧壁Gamma_w及翼型表面Gamma_a。
Gamma_in对应x=-3c，Gamma_out对应x=6c，Gamma_w对应y=+-2c。
这些侧壁是平面内边界；板间方向z不显式划分网格。

采用单一抛物线剖面

\[
 \boldsymbol u_{3D}(x,y,z,t)\approx
 \left(\varphi(z)\boldsymbol U(x,y,t),0\right),\qquad
 \varphi(z)=\frac32\left(1-\frac{4z^2}{b^2}\right),
 \quad -b/2\le z\le b/2.
\]

它的板间平均为1，且在上下板为零。记板间平均为角括号，则

\[
 \langle\varphi\rangle=1,\quad
 \langle\varphi^2\rangle=\frac65,\quad
 \langle\varphi^3\rangle=\frac{54}{35},\quad
 \langle|\varphi'|^2\rangle=\frac{12}{b^2}.
\]

这给出单剖面加权残差模型的系数；模型出处及其约化假设见
[Thin gap approximations for microfluidic device design，式18–19a](https://arxiv.org/html/2511.03956v2)。
当前实现没有求解该文的更高阶板间模态。

## 2. 物理模型与归一化

设nu为运动黏度、rho为密度。无额外体力时，模型为

\[
 \frac65\partial_t\boldsymbol U
 +\frac{54}{35}(\boldsymbol U\cdot\nabla)\boldsymbol U
 -\frac65\nu\Delta\boldsymbol U
 +\frac{12\nu}{b^2}\boldsymbol U
 +\nabla(p/\rho)=0,
 \qquad \nabla\cdot\boldsymbol U=0.
\]

令a_t=6/5并除以a_t，定义

\[
 \pi=\frac{p}{a_t\rho},\qquad
 \beta=\frac97,\qquad
 \sigma=\frac{10\nu}{b^2}.
\]

于是实际交给时间格式的方程为

\[
 \boxed{
 \partial_t\boldsymbol U
 +\beta(\boldsymbol U\cdot\nabla)\boldsymbol U
 -\nu\Delta\boldsymbol U+\sigma\boldsymbol U+\nabla\pi=0,
 \qquad \nabla\cdot\boldsymbol U=0.}
\]

这里的面内黏性系数仍是nu。板间摩擦sigma U是隐式线性项，
不是显式对流项的一部分。初值为U(0)=0。
物理压力和归一化压力的关系始终为p=a_t rho pi。
后面的辅助变量及数值范数使用当前计算单位；本算例取$c=U_D=\rho=1$。

## 3. 修正后的边界条件

在翼型和侧壁上施加

\[
 \boldsymbol U=0\quad\text{on }\Gamma_a\cup\Gamma_w.
\]

令n为流体外法向，U_n=U dot n，并定义非负的入流量

\[
 U_n^-=\max(-U_n,0).
\]

为区分不同实验，用chi表示启用动能修正的开放边界部分：正式长算在入口和出口均取chi=1；
只修正入口的前置检查在入口取1、出口取0。
在$\Gamma_o=\Gamma_{\rm in}\cup\Gamma_{\rm out}$上使用

\[
 \boxed{
 \nu\partial_n\boldsymbol U-\pi\boldsymbol n
 =-\frac{p_{\rm ext}(t)}{a_t\rho}\boldsymbol n
  -\frac\beta2\chi U_n^-\boldsymbol U.}
\]

这是梯度应力形式的压力型边界条件。它改变了原纯自然应力条件，
不能理解为沿边界逐点给定内部静压p=p_ext。
当U_n>=0时，修正项为零。
入口保留未知速度，通过外部压力型载荷驱动；没有再给定入口速度剖面。
该方向性牵引的背景及与其他开放边界条件的区别见
[Ni–Yang–Dong 2019，引言](https://arxiv.org/html/1807.07056v1)。
本实现不是该文OBC-A/B/C的复现，不继承其物理精度结论。

出口外部压力参考为0。入口压力载荷缓慢启动：

\[
 p_{\rm ext,in}(t)=\Delta p\,S(t/t_r),\qquad
 S(s)=\begin{cases}
 10s^3-15s^4+6s^5,&0\le s\le1,\\
 1,&s\ge1.
 \end{cases}
\]

当前Delta p=27、t_r=1。这个启动过程是随时间变化的压力载荷，
速度Dirichlet迹仍恒为零，不需要时变速度lifting。

## 4. 有限元弱形式及边界功

采用连续P2速度、连续P1压力：

\[
 X_h=\{\boldsymbol v_h\in[C^0(\overline\Omega)]^2:
       \boldsymbol v_h|_K\in[P_2(K)]^2,
       \boldsymbol v_h=0\text{ on }\Gamma_a\cup\Gamma_w\},
\]

\[
 Q_h=\{q_h\in C^0(\overline\Omega):q_h|_K\in P_1(K)\}.
\]

Q_h不附加零均值约束：压力型自然应力已经固定压力常数。
不可压条件为(div U_h,q_h)=0，对全部q_h成立；这是弱连续性，不是逐点零散度。

为保留对流自功恒等式，使用反对称形式

\[
 b_s(\boldsymbol w,\boldsymbol u,\boldsymbol v)
 =\frac12\big[(\boldsymbol w\cdot\nabla\boldsymbol u,\boldsymbol v)
              -(\boldsymbol w\cdot\nabla\boldsymbol v,\boldsymbol u)\big].
\]

完整非线性泛函包括体积对流、开放边界通量及动能修正：

\[
 \boxed{
 \mathcal C(\boldsymbol U;\boldsymbol v)
 =\beta b_s(\boldsymbol U,\boldsymbol U,\boldsymbol v)
 +\frac\beta2\int_{\Gamma_o}
       (U_n+\chi U_n^-)\boldsymbol U\cdot\boldsymbol v.}
\]

压力驱动作为独立外载荷：

\[
 \mathcal F(t;\boldsymbol v)
 =-\frac1{a_t\rho}\int_{\Gamma_o}
                 p_{\rm ext}(t)\boldsymbol v\cdot\boldsymbol n.
\]

因此半离散动量及连续性为

\[
 (\partial_t\boldsymbol U_h,\boldsymbol v_h)
 +\nu(\nabla\boldsymbol U_h,\nabla\boldsymbol v_h)
 +\sigma(\boldsymbol U_h,\boldsymbol v_h)
 -(\pi_h,\nabla\cdot\boldsymbol v_h)
 +\mathcal C(\boldsymbol U_h;\boldsymbol v_h)
 =\mathcal F(t;\boldsymbol v_h),
\]

\[
 (\nabla\cdot\boldsymbol U_h,q_h)=0.
\]

由于b_s(U,U,U)=0，非线性自功完全来自边界：

\[
 \mathcal P(\boldsymbol U)
 :=\mathcal C(\boldsymbol U;\boldsymbol U)
 =\frac\beta2\int_{\Gamma_o}(U_n+\chi U_n^-)|\boldsymbol U|^2.
\]

正式长算chi=1，因此

\[
 \mathcal P(\boldsymbol U)
 =\frac\beta2\int_{\Gamma_o}\max(U_n,0)|\boldsymbol U|^2\ge0.
\]

这控制了原压力入口的负向对流自功。它是空间边界项的符号性质，
本身不是整个变步长SDIRK3格式的稳定性定理。
物理对流功、入口动能修正及出口回流耗散在诊断中分别保存。

## 5. 矩阵形式与SAV辅助方程

用u表示自由速度系数向量，定义一致质量矩阵M、梯度刚度矩阵A、
弱散度矩阵B以及物理线性耗散矩阵

\[
 L=\nu A+\sigma M.
\]

令n(u)为非线性弱载荷向量，即n(u)^T v=C(U_h;V_h)，
令f(t)为压力载荷向量。则物理半离散系统为

\[
 M\dot{\boldsymbol u}+L\boldsymbol u-B^T\boldsymbol\pi
       +\boldsymbol n(\boldsymbol u)=\boldsymbol f(t),
 \qquad B\boldsymbol u=0.
\]

代码接口返回的是M^(-1)n(u)；下文为避免质量内积混淆，始终采用弱载荷向量n(u)。

对于当前零速度Dirichlet迹，定义SAV边界功

\[
 \ell(\boldsymbol u)=-\mathcal P(\boldsymbol U_h).
\]

它从同一套边界求积计算，满足

\[
 \boldsymbol n(\boldsymbol u)^T\boldsymbol u+\ell(\boldsymbol u)=0.
\]

当前三阶格式使用

\[
 G(r)=1-r^3,\qquad Q(r)=1+r+r^2,\qquad G(r)=(1-r)Q(r),\qquad r(0)=0.
\]

增广系统为

\[
 \boxed{\begin{aligned}
 M\dot{\boldsymbol u}+L\boldsymbol u-B^T\boldsymbol\pi
       +G(r)\boldsymbol n(\boldsymbol u)&=\boldsymbol f(t),\\
 B\boldsymbol u&=0,\\
 \dot r+\gamma r
       &=-Q(r)\big[\boldsymbol n(\boldsymbol u)^T\boldsymbol u+\ell(\boldsymbol u)\big].
 \end{aligned}}
\]

连续时间自配对时，括号为零，因此r'=-gamma r，r(0)=0给出r=0，恢复物理模型。
离散阶段中，“已知对流”和“当前速度”来自不同阶段，括号一般不为零，
所以不能在实现中强制r=0或把该跨阶段配对直接删掉。
压力驱动f不乘G；入口/出口动能修正属于n，随完整非线性项一起乘G并参与ell。

## 6. 四阶段增量SDIRK3-mr-ccSAV

令tau=t_(n+1)-t_n>0，并设u_0=u^n、r_0=r^n。
本节的r_0是本时间步的已知起点，不是每步重新设置为初始值0。
阶段时刻为t_n+c_i tau，其中

\[
 (c_0,c_1,c_2,c_3,c_4)=\left(0,\frac14,\frac45,1,1\right).
\]

增量系数为

\[
 D=\begin{pmatrix}
 \frac14&0&0&0\\
 \frac3{10}&\frac14&0&0\\
 -\frac{31}{220}&\frac1{11}&\frac14&0\\
 \frac{31}{198}&\frac5{132}&-\frac49&\frac14
 \end{pmatrix},
\]

\[
 \widehat D=\begin{pmatrix}
 \frac14&0&0&0\\
 -\frac{33}{50}&\frac{121}{100}&0&0\\
 \frac{239}{400}&-\frac{71}{100}&\frac5{16}&0\\
 -\frac1{48}&-\frac{17}{66}&\frac{235}{528}&-\frac16
 \end{pmatrix}.
\]

D的第j列对应未知阶段u_j，j=1,...,4；Dhat的第j+1列对应已知阶段u_j，j=0,...,3。
它们是增量形式的系数，不是直接写在普通Butcher形式中的A和Ahat。
每行和为c_i-c_(i-1)；最后一行和为0，但最后一个校正阶段仍必须计算。

在第i阶段，先用已知阶段构造

\[
 \boldsymbol c_i=\sum_{j=0}^{i-1}\widehat d_{i,j+1}\boldsymbol n(\boldsymbol u_j),\quad
 e_i=\sum_{j=0}^{i-1}\widehat d_{i,j+1}\ell(\boldsymbol u_j),\quad
 \boldsymbol f_i=\sum_{j=0}^{i-1}\widehat d_{i,j+1}\boldsymbol f(t_n+c_j\tau).
\]

随后求u_i、r_i及阶段增量压力lambda_i：

\[
 \boxed{\begin{aligned}
 M(\boldsymbol u_i-\boldsymbol u_{i-1})
 +\tau\sum_{j=1}^{i}d_{ij}L\boldsymbol u_j
 -\tau B^T\boldsymbol\lambda_i
 +\tau G(r_i)\boldsymbol c_i&=\tau\boldsymbol f_i,\\
 B\boldsymbol u_i&=0,\\
 r_i-r_{i-1}+\gamma\tau\sum_{j=1}^{i}d_{ij}r_j
 +\tau Q(r_i)\big(\boldsymbol c_i^T\boldsymbol u_i+e_i\big)&=0.
 \end{aligned}}
\]

这里f_i、c_i及e_i使用同一组显式增量系数。
尤其e_i不是用当前u_i重新计算的-P(U_i)，而是已知边界功的增量组合。
第4阶段结束后令

\[
 \boldsymbol u^{n+1}=\boldsymbol u_4,\qquad r^{n+1}=r_4.
\]

## 7. 每阶段如何化为两次线性求解和一个标量方程

把已知线性项移到右端，记

\[
 R_i=M\boldsymbol u_{i-1}+\tau\boldsymbol f_i
              -\tau\sum_{j<i}d_{ij}L\boldsymbol u_j,\qquad
 s_i=r_{i-1}-\gamma\tau\sum_{j<i}d_{ij}r_j.
\]

所有阶段的对角元都是1/4，所以同一时间步只需要同一个矩阵

\[
 H_\tau=M+\frac\tau4 L
       =\left(1+\frac{\sigma\tau}4\right)M+\frac{\nu\tau}4A.
\]

首先解两个共享矩阵的Stokes问题：

\[
 \begin{pmatrix}H_\tau&-B^T\\B&0\end{pmatrix}
 \begin{pmatrix}V_i\\\xi_i\end{pmatrix}
 =\begin{pmatrix}R_i\\0\end{pmatrix},\qquad
 \begin{pmatrix}H_\tau&-B^T\\B&0\end{pmatrix}
 \begin{pmatrix}W_i\\\zeta_i\end{pmatrix}
 =\begin{pmatrix}\boldsymbol c_i\\0\end{pmatrix}.
\]

由线性叠加可得

\[
 \boldsymbol u_i=V_i-\tau G(r_i)W_i,\qquad
 \boldsymbol\lambda_i=\xi_i/\tau-G(r_i)\zeta_i.
\]

因此完整速度非线性问题只剩一个标量未知r_i。计算

\[
 a_i=\boldsymbol c_i^T V_i+e_i,\qquad
 b_i=\boldsymbol c_i^T W_i=W_i^TH_\tau W_i\ge0,
\]

将速度表达式代入辅助方程，得到

\[
 \boxed{
 \left(1+\frac{\gamma\tau}{4}\right)r_i-s_i
 +\tau a_i(1+r_i+r_i^2)
 -\tau^2b_i(1+r_i+r_i^2)(1-r_i^3)=0.}
\]

它通常为五次多项式。按r^5到常数排列的系数为

\[
 \left[\tau^2b_i,\ \tau^2b_i,\ \tau^2b_i,\
 \tau a_i-\tau^2b_i,\
 1+\frac{\gamma\tau}4+\tau a_i-\tau^2b_i,\
 \tau a_i-\tau^2b_i-s_i\right].
\]

实现枚举数值实根并复核残差，选择绝对值最小的根；并列时取较小的有符号根。
不把r强制截断到[0,1]，也不将辅助变量每步重置为0。
当高次系数恰好为零时按实际次数求解，不人为保留五次项。

每步总计4次双右端Stokes求解及4个标量方程。
同一步的8个线性响应共享同一阶段矩阵；非线性标量r不改变这个矩阵。
不同时间步长会改变矩阵；现有复用后端仍针对新步长的精确系统作完整残差检查。

## 8. 阶段压力与物理压力

lambda_i是增量形式中的约束乘子，不能仅把最后一个lambda_4乘比例系数，
就称为已验证三阶的物理终点压力。
程序对保存速度另作瞬时物理半离散压力恢复：

\[
 \begin{pmatrix}M&-B^T\\B&0\end{pmatrix}
 \begin{pmatrix}\dot u_*\\\pi_*\end{pmatrix}
 =\begin{pmatrix}\boldsymbol f(t)-L\boldsymbol u-\boldsymbol n(\boldsymbol u)\\0\end{pmatrix},
 \qquad p_*=a_t\rho\pi_*.
\]

这个诊断使用物理模型G=1，定义为给定速度场对应的瞬时约束速度率和压力，
不是SAV阶段增量压力的重命名，也不是已证明高阶的压力后处理。

## 9. 自适应步长

四阶段速度同时给出二阶嵌入组合

\[
 \widehat{\boldsymbol u}^{n+1}
 =\frac23\boldsymbol u_0-\frac{128}{99}\boldsymbol u_1
  +\frac{50}{33}\boldsymbol u_2+\frac19\boldsymbol u_3.
\]

一致质量范数为||v||_M=sqrt(v^T M v)。速度和标量指标分别为

\[
 E_u=\frac{\|\boldsymbol u_4-\widehat{\boldsymbol u}^{n+1}\|_M}
 {\mathrm{atol}_u+\mathrm{rtol}_u\max(\|\boldsymbol u_0\|_M,\|\boldsymbol u_4\|_M)},
 \qquad
 E_r=\frac{|r_4-e^{-\gamma\tau}r_0|}{\mathrm{atol}_r}.
\]

只有阶段/根/连续性检查通过且max(E_u,E_r)<=1才接受试步。
速度与标量分别保留PI历史，并用局部缺陷阶数3和2提出步长建议，取较小者。
这不表示r具有三阶全局精度，也不表示局部atol_r直接给出了全局|r|上界。
最后一步对齐T；普通步长低于下限时停止，未启用强制下限步长越过误差控制。
为终点对齐产生的更短末步仍接受同样的误差检查。

## 10. 本次正式运行参数与验证范围

|项目|当前正式成员|
|---|---|
|弦长、完整板间距|c=1，b=.02|
|密度、运动黏度|rho=1，nu=1e-4|
|对流系数、板间阻力|beta=9/7，sigma=2.5|
|攻角|10度|
|入口压差、爬升时间|Delta p=27，t_r=1|
|名义Darcy速度|U_D=b² Delta p/(12 rho nu *9c)=1|
|名义雷诺数|Re_D=U_D c/nu=10000|
|实际雷诺数|Re_c(t)=出口平均流向速度*c/nu|
|空间离散|P2/P1，主网格129390三角形|
|SAV参数|gamma=1，初始r=0|
|时间区间|0到20|
|速度容差|atol_u=1e-9，rtol_u=1e-4|
|标量容差|atol_r=1e-4|
|初始/最小/最大步长|1e-4 / 1e-7 / .005|
|输出|每.05请求一个快照，保存最近接受节点的实际时刻|

velocity_limit=20和scalar_limit=.05是本次显式运行资格门槛：超限后停止并保留接受前缀，
不裁剪速度或r，不改变SAV方程。它们不是稳定性定理中的先验界。
前置T=1.1计算与正式T=20计算均从零初值独立开始，不是checkpoint续算。

本文核对了运行冻结源码中的模型、边界、阶段系数、压力定标及自适应公式；
抛物线剖面矩、增量系数行和、G=(1-r)Q及五次多项式展开作了独立精确代数检查。
这些检查不替代完整mr-ccSAV/PDE收敛证明。三阶速度基格式、辅助变量精度、阶段压力精度
以及当前高Re翼型解的时间/空间收敛需要分别说明。
模型本身仍是单剖面二维约化模型，未作本翼型高Re的三维校核。

## 11. 代码对应

- 物理模型与入口修正：[solver/fem/hele_shaw.py](/Users/wanghaifeng/Desktop/MAC-mr-SAV-RK/solver/fem/hele_shaw.py)。
- 同一transport的动量/标量边界配对及压力恢复：[solver/fem/open_flow.py](/Users/wanghaifeng/Desktop/MAC-mr-SAV-RK/solver/fem/open_flow.py)。
- 增量系数、已知项及嵌入组合：[solver/schemes/sdirk3.py](/Users/wanghaifeng/Desktop/MAC-mr-SAV-RK/solver/schemes/sdirk3.py)。
- 双右端求解、五次方程及完整阶段检查：[solver/schemes/sdirk3_mrsav.py](/Users/wanghaifeng/Desktop/MAC-mr-SAV-RK/solver/schemes/sdirk3_mrsav.py)。
- 本次运行配置：[experiments/hele_shaw_airfoil/inlet_recovery.py](/Users/wanghaifeng/Desktop/MAC-mr-SAV-RK/experiments/hele_shaw_airfoil/inlet_recovery.py)。

核对的冻结任务：runs/hele_shaw_wake/20261102T071827-ef7aba44。
