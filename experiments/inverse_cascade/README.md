# 受迫二维 MHD：逆级联边缘（Seshasayanan et al., 2014）

**先看简明报告：**[磁场会怎样改变能量的传递方向？](../../reports/inverse_cascade/readable_20261110/README.md)
——用直观说明、两张中文图和数据表解释当前结果，无需先读方程。

**最新长窗结果（2026-11-11）：**[512²、μ=0.20、T*=1000 图文整理](../../reports/inverse_cascade/long512_mu020/results_20261111/README.md)。
计算已完成，通量分块门槛和低通收支通过，但大尺度耗散变化仍超门槛，整体统计资格未通过。

> **2026-11-10 审计更正**：旧 `reports/inverse_cascade/plan/configs/` 高尺度配置有误，
> 不得用于执行；旧 schema 2 的 τ、ESS 及统计资格结论撤回。当前分析为 schema 3。
> [验收记录与剩余限制](../../docs/inverse-cascade-toolchain-audit.md)。本轮未连接服务器、
> 未启动生产计算；以下历史物理表格仅为有限窗记录，不是当前统计资格证明。

目的：使用已有 SDIRK3-mr-ccSAV 研究磁性强迫增大时逆向能量通量减弱，
以及小平均通量与强瞬时输运的差别。普通 SDIRK3 是数值对照。
控制量为强迫幅值比 mu，而不是实际功率比。先检查输运和收支，暂不拟合临界指数。

原文：[PRE 90, 051003(R)](https://doi.org/10.1103/PhysRevE.90.051003)，
本次核对[正式 PDF](https://www.phys.ens.psl.eu/~alexakis/PAPERS/InverseCascade_PhysRevE.90.051003.pdf)
式 (1)–(5)、表 I 和图 1–4。程序是独立算例，不改变旧的无驱动 MHD 记录和算法。

## 方程和单位

盒子为 `(0,2*pi)^2`，即论文 L=1；磁场采用 Alfvén 速度单位。
`solver/fourier/forced_mhd.py` 复用原 FourierMHD 的联合非线性和完整双场散度检查。
两场使用相同的正耗散乘子

`D(k) = nu_plus*|k|^4 + nu_minus*|k|^-4`。

零模的逆幂定义为零，零平均初值、零平均外源保持零平均；没有隐式清零操作。
强迫由论文的涡量和磁通源准确转换：

- `f_u = f0*(-cos(kf*x)*sin(kf*y), sin(kf*x)*cos(kf*y))`；
- `f_B = mu*f0*(sin(kf*x)*cos(kf*y), -cos(kf*x)*sin(kf*y))`。

这里 `kf` 为坐标方向模数，实际注入壳层半径是 `sqrt(2)*kf`。
配置 `T`、`dt` 和日志时间是论文时间单位 `t0=1/sqrt(f0*kf)` 下的数值；
HDF5 时间为物理计算时间，除以保存的 `t0` 即可转换。`gamma` 以 `1/t0` 为单位。
`nu_plus=sqrt(f0)*kf^(-7/2)/Re_plus`，
`nu_minus=sqrt(f0)*kf^(9/2)/Re_minus`。二者的幂由原文 n=m=2 得到。
外力是加性项，不乘 SAV 非线性耦合因子。两场使用同一个现有两次固定校正的 SAV 标量。

原文没有完整指定初始噪声、每例启动时间及输出采样方式。这里明确选择种子 2014 的
低模随机流函数／磁通函数：每方向最多 `kf+1` 模，生成格点无关的连续三角多项式，
速度 RMS 为 `noise_rms*sqrt(f0/kf)`，磁场 RMS 再乘 mu。mu=0 精确成为流体对照。
这不是原文提供的初始场。非零扰动用于打破定常强迫的对称不变子空间。

## 配置分档

| 配置 | 用途 | 网格、尺度和耗散 | 时间 |
|---|---|---|---|
| `configs/pilot.json` | 降低成本的探索扫描，不是论文 DNS | 96², kf=4, Re+=80, Re-=1000；mu=.05/.2/.3/.8 | T*=80, dt*=.005, gamma=1 |
| `configs/pilot_checks.json` | mu=.2 的普通格式、半步长、128² 检查 | 其余与 pilot 一致 | 同一采样间隔 .1 t0 |
| `configs/study/` | 正式 kf=8 试算：资格、mu 扫描、种子、窗口、Re 依赖、512² | 见下方结果各表 | 见下方结果各表 |
| `configs/paper512_candidate.json` | **尚待资格验证的候选配置**，不是已完成结果 | 512², kf=8, Re+=1400, Re-=23000, mu=.21 | T*=100，dt*=.005，gamma=1（**未按新结论更新，仅作旧候选**）|

原文表 I 四档为 `(kf*L,N,Re-,T*) = (8,512,2.3e4,2000),
(16,1024,7.4e5,600), (32,2048,2.3e7,342), (64,4096,7.6e8,100)`，
全部 `Re+=1400`。图 2–3 用 kf*L=16，图 4 用 64。
`paper512_candidate` 仅对齐最小组的空间／耗散参数；T*=100 不是表中完整波动采样时间。
该文件早于 `gamma` 结论，保留原样以免改写历史；实际 512² 算例用
`configs/study/n512_*.json`（`gamma=1000`、显式耦合门禁）。
步长和初值也不是论文披露的数值。必须先通过分辨率、时间步和统计检查再延长。

计算保留每步标量诊断和阶段历史，内存随步数增长；只保留初末全场，不保留每步全场。
每 `sample_every` 步保存通量和能谱（增加一次非线性评估），`T/dt` 要整除采样间隔。
**长时统计用分段 chain**（`segments.py`）：每 `segment_steps` 个接受步提交一段
并写 checkpoint。正常段边界暂停可重启；崩溃遗留/锁须人工检查，不自动恢复。
`failed` 不可续算；未到计划终点的正常暂停 chain 标为 `running`，
不得当作完成样本。单文件路径仍保留给短片试算，两者物理推进完全相同（分段只改存储）。
已接受的可捕获失败前缀保存为 failed，强制杀死可能仅留下 running 记录。

较早的 `reports/inverse_cascade/final_*`、`qual_*` 等目录是 schema 1 的旧产物，
仍可用于历史对照，但**以 `reports/inverse_cascade/v2/` 与 `summary.json` 里的
`analysis_schema_version=3` 为准**；schema 2 的 τ/ESS/资格结论撤回，旧目录不重写。

## 计算和只读分析

```sh
uv run --locked python -m experiments.inverse_cascade.run --scan \
  --config experiments/inverse_cascade/configs/pilot.json
uv run --locked python -m experiments.inverse_cascade.run --scan \
  --config experiments/inverse_cascade/configs/pilot_checks.json
# 取计算入口实际打印的 batch.json 路径，不自动搜索“最新结果”
uv run --locked python -m experiments.inverse_cascade.analyze \
  --batch runs/inverse_cascade_scans/EXPLICIT_BATCH/batch.json \
  --burn-in 40 --block 5 --K 4 --output reports/inverse_cascade/EXPLICIT_REPORT
```

也可给分析入口显式多个运行目录／run ID，不用 `--batch`；单例计算省略 `--scan`。
扫描串行执行，完整匹配有效配置、代码和环境并通过 reader 的旧成员才复用。

正式 kf=8 试算用 `configs/study/` 下的单例配置与三个启动器；已完成的同名配置会经
`reusable_run` 复用，因此重复执行启动器不会重算：

```sh
# 资格检查（128/192/256 网格、192 的 dt 阶梯）与 mu=0.2 基准 + 512² 首个单点
sh experiments/inverse_cascade/configs/study/launch_phase1.sh
# 其余 mu、种子、T*=200 窗口、Re 依赖
sh experiments/inverse_cascade/configs/study/launch_wave1.sh
sh experiments/inverse_cascade/configs/study/launch_sweep.sh
sh experiments/inverse_cascade/configs/study/launch_qual.sh
# 重算论文最小 DNS 组合（512²）的单点，最长的 T*=300 单例约 5.2 h
uv run --locked python -m experiments.inverse_cascade.run \
  --config experiments/inverse_cascade/configs/study/n512_mu0200_T300.json
```

启动器只是按顺序调用同一个 `run.py`；并行度、日志目录均在其中显式写出。
本机八核下 4–5 个并发进程即可饱和，再增加只会互相拖慢（FFT 受内存带宽限制）。
每例保存在 `runs/inverse_cascade/<id>/`，日志 `run.log` 每 `log_every` 步报告一次。
结果文件发布、校验和及 manifest 使用项目共同工作流。计算失败不会出现在完成报告中。

保存内容：

- 公共每步 `diagnostics`、阶段残差、固定次数标量校正及初末快照；
- `final/magnetic`、`snapshots/magnetic`，顺序 Bx/By；
- `spectra/{times,K,Pi_E,Pi_Q,E_u,E_B}`；K 为整数径向边界，低通保留 `|k|<=K`，
  壳层为 `(K-1,K]`，零模归入 0，rFFT 双权重给出面积平均；
- `spectra/{E_low,Q_low,power_E_low,power_Q_low,diss_*_low}`：低通累积能量、注入和
  两条耗散通道，用于逐波数核对 `dX_<K/dt = P_<K - D+_<K - D-_<K - Pi_X(K)`；
  旧记录没有这些列，读取时视为可选，不做回填；
- 能量／磁通平方的实际注入与大小尺度分耗散，四个向量分量的均值。

正通量表示低模流出（向小尺度），负值表示向大尺度。
这里记录的是在接受状态上评估的**原 PDE 非线性通量**，不把它冒充 SAV 修改后的
离散阶段通量。分析另外给出 `max_abs_coupling_minus_one`、r 和物理收支缺陷；
SAV 若明显改变非线性，应缩步并与普通格式比较。

## 分析输出的量的定义（读图前先看）

| 字段 | 定义 | 何时为 `null` |
|---|---|---|
| `mean_Pi_E`, `mean_Pi_Q` | 窗口平均的**原始** PDE 通量 | 永不为空 |
| `mean_Pi_*_over_injection` | 除以该场的窗口平均注入 `<P_E>`／`<P_A>` | 注入非正或非有限（如纯流体对照的 A 通道）|
| `large_*_fraction` | 大尺度耗散／该场**总耗散**（原文图 1 的 `eps-/eps`）| 总耗散为零 |
| `large_*_over_injection` | 大尺度耗散／该场**平均注入** | 注入为零 |
| `E_storage_over_injection` | 储能速率／注入，非稳态时为非零 | 注入为零 |
| `tau_int_Pi_E`, `n_eff_Pi_E` | Geyer 初始正序列积分相关时间与有效样本数 | 常量序列或样本不足 |
| `lowpass_window_defect_E` | 低通收支窗口缺陷／总注入，合格阈值 1e-3 | 旧记录无低通列 |

两类比值在非稳态下**不相等**，不得互换标注。原文的 `eps_E` 是稳态注入，而本仓库
除 `<P_E>` 外还同时给出两个比值，正是为了不把瞬态混进稳态口径。

分析输出 `mean_flux.png`、`instantaneous_flux.png`、`window_statistics.png`、
`spectra.png`、`transition.png`（同一同质组才算）、`statistics.csv`、`summary.json`，
`analysis_schema_version=3`。`mean_flux_raw` 单独绘原始量，`mean_flux` 只绘可归一化量，
两者均有 PNG/PDF，不混轴。`transition.png` 只在比较组**同一网格、dt、T、格式、
Re⁺/Re⁻ 且 mu 互不相同**时绘制，否则 `summary.json` 记录拒绝原因而不画汇总曲线。

分块使用完整、不重叠 `(left,right]` 区间；块长需为采样间隔整数倍，至少四块才画
Student-t **95% 启发式块区间**（不是 ±1σ）。`block_means` 只用完整块，尾块丢弃量
记入 `tail_dropped_t0`；`mean`/`std` 则用 burn-in 之后的全部样本。两者口径不同，
不得混用。报告同时给出绝对标准差、正负占比、标准差／均值、积分相关时间与
有效样本量，不因小分母自动推断绝对波动发散。

稳态筛查（`stats.stationarity_screen`；入口使用固定门槛，不搜索通过的窗口）：

- burn-in 后全跨度及两个连续、非重叠的半跨度分别检查；
- 储能速率绝对值 ≤ 平均注入的 2%；
- 通量半窗差 ≤ 相关修正合并标准误的 2 倍，且 ≤ **平均注入**的 5%；
- 注入、小尺度耗散、大尺度耗散各自半窗变化 ≤ 各自均值的 5%；
- 各评估窗至少 20 个完整块、块长 ≥10τ、ESS≥100，才可 `statistics_qualified=true`；
- 常量/不足/未截断相关估计或非正注入不制造资格，原因及末两四分窗保存到 JSON。

τ 使用 biased FFT 自协方差的 Geyer IPS：`g=-1+2Σ(rho[2k]+rho[2k+1])`，
`τ=Δt*g/2`，`ESS=N/g≤N`。主图区间中心为**完整块均值**，与全样本 `mean` 分列。
这只是操作筛查，不是平稳性、统计收敛或论文物理结论的证明。

## 验收边界

1. 独立受迫剪切精确解验证隐式高／负阶耗散和两种阶段格式；
2. 三角多项式独立导数验证双场能量和 Q 通量及正负符号，完整频带总输运为零；
3. 原文涡量源／磁通源、幅值比和格点无关初值验证；
4. 公开 run/load/reuse/scan/analyze 和校验和验证；
5. 实际试算要另外检查统计平衡、普通／SAV、半步和网格敏感性。

即便探索性扫描出现方向变化，也不代表复现原文的临界位置、指数、谱指数、
无限 Reynolds 数极限或普适间歇性。已执行结果及未覆盖边界见 `docs/validation.md`。

## SAV 标量阻尼 `gamma` 与耦合门禁（务必先读）

本实验的 SAV 标量使用仓库约定的 `G=1-r`、`Q=1`。**持续强迫会驱动 r 漂移**：
标量方程里的 `alpha=<convection, V>` 中的 V 含外力响应，因此外力越强，r 的不动点越远离 0。
若 `gamma` 太小，`r→1` 使 `G=(1+r+r²)(1-r)→0`，阶段速度方程里的非线性项被压小一到两个量级，
表现为“跑得完、残差合格、能量却持续增长不饱和”。

实测（`256²、kf=8、Re+=1400、mu=0.2、dt*=0.005、T*=100`）：

| gamma | r 终值 | 最大 |G−1| | 结果 |
|---|---|---|---|
| 1 | 0.9918 | 0.988 | 非线性被抑制，E 由 0.7 单调涨到 12.7，判定为无效 |
| 1000 | 1.6e-4 | 4.9e-7 | 与同配置普通 SDIRK3 的能量轨迹一致（t*=40 时 1.138 vs 1.145） |

**`gamma=1` 在早期 96²/kf=4 试算上同样失效**：从记录重新测得的 `max|G-1|` 为 1.0
（G 一度塌到 0），所以那批数据也不能作为“参数不同但有效”的证据。
`experiments.inverse_cascade.inventory` 直接从每条记录重算该偏离，并按 1e-3 阈值
排除，不依赖配置里写的 gamma。

因此本目录所有正式配置使用 `gamma=1000`，并显式设置 `coupling_tolerance=1e-3`：
积分结束后若 `max|frozen_factor*(1-r)-1|` 超过阈值，运行记为 failed 且不发布
`results.h5`（分段 chain 则逐段检查）。该门禁只验证离散自洽，**不证明物理准确**；
真正的步长判据仍是显式的 dt 阶梯比较（见下）。

`coupling_tolerance` 是可选键，只在显式提供时写入配置，因此不影响旧记录身份。

## 分辨率与步长资格（kf=8、mu=0.2、T*=100、窗口 t*>40、K=8）

| 配置 | 平均 Π_E | 块区间半宽 | 判定 |
|---|---|---|---|
| 128² | −0.0314 | 0.0043 | 与 192² 差 0.0086 > 半宽，**不足** |
| 192² | −0.0228 | 0.0035 | — |
| 256² | −0.0221 | 0.0037 | 与 192² 差 0.0007，**通过** |
| 192², dt*=0.005 | −0.02275 | 0.0035 | — |
| 192², dt*=0.0025 | −0.02231 | 0.0035 | 差 0.00044，**通过** |

`kf=8` 在 `128²` 上只剩约 5 倍尺度范围，通量明显偏大，故该分辨率不作物理论述。
生产窗口取 `192²–256²、dt*=0.005`。`gamma` 正确后 192² 的耦合偏离为 2e-6 量级。

## 已完成的 kf=8 结果（2026-11-07）

统一参数：`kf=8,f0=1,Re+=1400,Re-=23000,gamma=1000,dt*=0.005,seed=2014`，
分析窗口 `t*>40`、块长 `t*=5`、代表截断 `K=8`（`K/kf=1`，低于强迫壳层 `sqrt(2)kf`）。
全部算例的 `max|G-1|` 在 `1e-8–1e-5`；`128²` 的失败例（`gamma=1`）单独保留为反例。

### 1. mu 扫描（256²，T*=100）

| mu | 平均原始 Π_E | 95% 块区间 | 原始 Π_Q | 大尺度 E／总耗散 | 大尺度 Q／总耗散 |
|---|---|---|---|---|---|
| 0.05 | −0.0548 | 0.0054 | −1e−8 | 0.866 | 0.004 |
| 0.10 | −0.0454 | 0.0041 | −3e−8 | 0.785 | 0.005 |
| 0.15 | −0.0381 | 0.0037 | −6e−8 | 0.587 | 0.005 |
| 0.20 | −0.0221 | 0.0037 | −3e−7 | 0.403 | 0.004 |
| 0.25 | **+0.0038** | 0.0036 | −1e−6 | 0.201 | 0.009 |
| 0.30 | +0.0173 | 0.0051 | −9e−6 | 0.075 | 0.016 |
| 0.40 | +0.0226 | 0.0140 | −9e−5 | 0.062 | 0.063 |
| 0.80 | −0.0024 | 0.0025 | −7e−4 | 0.015 | 0.883 |

能量逆输运在 `mu=0.20` 与 `0.25` 之间穿零（`0.25` 仅约 1σ 为正）；磁通平方的逆输运
随 mu 开启，大尺度 Q 耗散占比由 0.004 升到 0.883。这两条方向与原文图 1–2 一致。
`mu=0.80` 的平均 Π_E 回到零附近而 `std/|mean|=4.1`，即“均值很小、瞬时波动主导”。

### 2. 网格与步长（mu=0.2，T*=100）

| 配置 | 平均 Π_E | 块区间半宽 |
|---|---|---|
| 128² | −0.0314 | 0.0043 |
| 192² | −0.0228 | 0.0035 |
| 256² | −0.0221 | 0.0037 |
| 512² | −0.0192 | 0.0042 |
| 192², dt*=0.0025 | −0.0223 | 0.0035 |

192²/256²/512² 三档两两之差（0.0007、0.0029、0.0035）都不超过各自的块区间半宽，
`128²` 明显偏离。步长减半使均值变化 0.00044。因此 `192²–512²、dt*=0.005` 是本次可用的
数值档位，`128²` 不作物理结论。

### 3. 种子、观测窗口与块长

- 种子：`mu=0.2`（192²，seed 2014/7/99）平均 Π_E = −0.0228/−0.0218/−0.0209，极差 0.0019，
  小于块区间半宽 0.0035；`mu=0.8` 两个额外种子给出 −0.00315/−0.00308，与 256² 的 −0.00241 一致。
- 窗口：`mu=0.2` 的 `T*=100`（−0.02275±0.0035）与 `T*=200`（−0.02220±0.0016）均值相差 0.0006，
  后者区间半宽减半；`mu=0.8` 同样在区间内一致（−0.00241 与 −0.00341±0.0019）。
- 块长：5→10 不改变均值（定义使然，均值取全部后处理样本），只改变区间半宽，
  最大变化为 `mu=0.4` 的 0.0140→0.0235。穿零区间 [0.20,0.25] 在两种块长下都成立。

### 4. Reynolds 数依赖（mu=0.2，192²）

| 变化 | 平均 Π_E | 大尺度 E 耗散占比 | 储能/注入 |
|---|---|---|---|
| Re+=700 | −0.0343 | 0.508 | −0.19 |
| Re+=1400（基准） | −0.0228 | 0.413 | −0.21 |
| Re+=2800 | −0.0090 | 0.374 | −0.37 |
| Re-=5000 | −0.0272 | 0.371 | −0.08 |
| Re-=23000（基准） | −0.0228 | 0.413 | −0.21 |
| Re-=100000 | −0.0134 | 0.559 | −0.60 |

固定 mu 时提高 `Re+` 会显著削弱逆向通量，与原文“`mu_c` 随 `Re+` 变化”的说法一致。
`Re-=1e5` 的储能/注入达 −0.60，说明该点远未统计平衡，只能作为趋势而非定量结论。

### 5. 必须一起读的限制

- 所有算例的储能/注入为 −0.01 到 −0.60，即**窗口内仍未达到统计平衡**；把 `mu=0.2`
  从 `T*=100` 延长到 `T*=200` 使该值从 −0.21 降到 −0.08，均值不变。原文同组 `T*=2000`。
- 用 `stats.screen` 对现有 256² 八点扫描逐点检查（`reports/inverse_cascade/v2/mu_sweep`）：
  **八点全部不通过**（`mu=0.05/0.15/0.20` 因储能，`mu=0.10/0.20/0.25/0.30/0.40`
  另有前后半窗漂移）。因此这些点只能支持“有限窗趋势”，不能称为稳态通量。
- **撤回 schema 2 的 τ/ESS 数值及相应资格结论**：旧 Geyer 换算有误。
  `v2/window`、`v2/window512` 的 `interval_qualified` 不再作为资格证据；必须用 schema 3
  对显式运行及固定窗口重分析。T*=100、burn-in=40、block=5 只有 12 个完整块，
  不满足 20 块的事实不受修复影响。
- 因此本文只能报告“穿零区间 + 方向性 + 有限窗不确定性”，不能报告 `mu_c` 数值、
  指数 `gamma_E/gamma_A`、无穷 `Re-` 极限或谱指数。
- `96²、kf=4、Re+=80` 的早期试算实测 `max|G-1|=1.0`，已按耦合门禁判废，
  只是流程证据（见 `inventory.md`），不能与正式算例混在一张 μ 曲线上。
- 分块区间是假设块均值平稳独立时的启发式估计，未做自相关时间估计以外的独立性检验。
- `K=8` 是**坐标方向模数** `kf`，径向强迫壳层在 `sqrt(2)kf≈11.3`；图中竖虚线标出该处。
  单点 `K=8` 的符号不替代整条通量谱的形状检查。

### 6. 512² 上的穿零区间（原文最小 DNS 组合）

| mu | 平均 Π_E | 95% 块区间 | 负通量样本占比 | 大尺度 E 耗散占比 |
|---|---|---|---|---|
| 0.20 | −0.0192 | 0.0042 | 0.80 | 0.367 |
| 0.30 | **+0.0155** | 0.0034 | 0.10 | 0.134 |

与 256² 的同名点（−0.0221±0.0037、+0.0173±0.0051）在区间内一致。
因此**符号改变在 256² 与 512² 上都被夹在 `mu∈(0.20,0.30)`**，
256² 的八点扫描进一步把上界收在 0.25 附近（该点仅 1σ 为正）。
512² 的长时间波动统计（原文该组 `T*=2000`）未做：本机单点 `T*=100` 已需 1.7 h，
按同速率 `T*=2000` 约 68 h/点。

### 7. 512² 上的窗口阶梯（mu=0.2，块长 t*=10）

| 窗口 | 平均 Π_E | 95% 块区间半宽 | std/\|mean\| | 大尺度 E 耗散占比 | 储能/注入 |
|---|---|---|---|---|---|
| T*=100 | −0.0192 | 0.0040 | 1.17 | 0.367 | −0.17 |
| T*=300 | −0.0209 | **0.0012** | 1.05 | 0.283 | **−0.05** |

窗口增至三倍，均值变化 0.0017（远小于区间半宽），区间半宽缩小 3.2 倍，
储能失衡由 17% 降到 5%，即系统确实在向统计平衡靠近但尚未达到。
这正是本文最想给出的“长时间统计”证据：**平均通量稳定、不确定性随窗口收缩**，
而不是靠单次短窗判断输运方向。直接回答了三件事：
弱/强磁性驱动的输运方向不同、转变区附近均值小而波动大、窗口增长时均值与不确定性如何变化。

## 论文级稿件与矢量图

`reports/inverse_cascade/manuscript.tex` 是本实验的中文论文稿（ctexart + amsmath +
booktabs），把模型方程、数值配置、资格检查、μ 扫描、Re 依赖、512² 复核与
“与原文逐图对照”汇成一份可编译文档；图引用 `v2/` 下分析产出的矢量 PDF。

**仓库里的 `manuscript.pdf` 是 2026-11-07 版本的旧构建，早于本轮修正，不得作为
当前结果引用**；本机没有 TeX 二进制，无法就地重编译。装好 TeX 后按下方命令重建，
并检查 PDF 里的图号与表值是否来自 `v2/`。

```sh
cd reports/inverse_cascade
PATH=/Library/TeX/texbin:$PATH latexmk -xelatex manuscript.tex   # -> manuscript.pdf
```

分析同时输出 PNG 与 PDF；图内数学由 `text.usetex` 交给 LaTeX 排印，
无 TeX 时自动回退 Computer-Modern mathtext（见 `experiments/plotting.py`
的 `latex_style`）。注意 MacTeX 的二进制目录 `/Library/TeX/texbin` 通常不在
非登录 shell 的 `PATH` 里，仓库里的工具解析器会自己去这些标准位置找。

## 与原文逐图对照（现状，不是“已复现”）

核对依据为原文图注原文（Fig. 1–4 caption）与正文，不含推测。

| 原文 | 内容 | 本仓库对应 | 当前判定 |
|---|---|---|---|
| Fig. 1 | `eps-_E/eps_E`、`eps-_A/eps_A` vs mu，四档 kfL，插图放大 + 幂律拟合 | `transition.png` 第 2、3 面板（只有 kfL=8） | 端点趋势一致；**多尺度收敛与指数拟合未做** |
| Fig. 2 | `Pi_E(k)/eps_E` 上、`Pi_A(k)/eps_A` 下，kfL=16 | `mean_flux.png`（kfL=8，N=256） | 通量结构与正负号一致；**尺度档与平台宽度未对齐** |
| Fig. 3 | 插图：瞬时 + 平均通量（mu=0.21, kfL=16）；主图：`sigma_E/eps-_E`、`sigma_A/eps-_A` vs mu | `window_statistics.png`、`instantaneous_flux.png`；`transition.png` 第 4 面板 | **主图未复现**：本仓库短窗下该比值在 mu>0.3 单调上升，临界处无峰，故标注 `no divergence claim` |
| Fig. 4 | `E_u`、`E_b` 谱，kfL=64，mu=0.21–0.26 | `spectra.png`（kfL=8，mu=0.05–0.80） | **仅定性**：磁能随磁强迫抬升；临界区谱斜率变化未复现 |

原文明示的两条边界（`mu≈0.22` 能量逆级联终止、`mu≈0.25` A 的逆级联开启）在本仓库
只落实为“穿零区间 `mu∈(0.20,0.25)`（256²）+ 512² 的 0.20/0.30 夹住”，
`eps-_A/eps_A` 从 `mu=0.25` 的 0.009 升到 0.063（0.40）与 0.883（0.80）。
**这不是 μ_c 测定，也不表示两个边界已被分辨为不同值。**

## 分段 chain、证据清单与计算矩阵

以下长算仅展示 API，不是本轮执行授权；正式运行先完成资源及资格审批。

```sh
# 长时统计：分段 + 断点续算（显式给出 chain 目录，不自动搜索）
uv run --locked python - <<'PY'
from experiments.inverse_cascade.segments import run_segmented
run_segmented({'nx':512,'ny':512,'kf':8,'mu':.2,'T':500.,'Re_plus':1400.,'Re_minus':23000.,
               'gamma':1000.,'dt':.005,'sample_every':20,'coupling_tolerance':1e-3},
              execution={'segment_steps':1000})
PY
# 只读分析：chain 走同一条统计路径
uv run --locked python -m experiments.inverse_cascade.analyze --chain runs/inverse_cascade_chains/CHAIN \
  --burn-in 40 --block 5 --K 8 --output reports/inverse_cascade/CHAIN_REPORT
# 证据清单：哪条记录可以用、为什么被排除
uv run --locked python -m experiments.inverse_cascade.inventory
# 计算矩阵与资源预算（默认 dry run，不连接任何服务器）
uv run --locked python -m experiments.inverse_cascade.plan --output reports/inverse_cascade/NEW_PLAN
```

chain schema 2：HDF5 先写 `.pending.h5` 再 rename，checkpoint JSON 的摘要绑定到 chain。
续算前逐段校验来源、配置、父 checkpoint、阶段形状、共同固定校正因子、全场连续性、
采样相位及状态。空采样段合法，failed 不可续算；max_wall_seconds 仅在段边界检查。
锁、孤儿文件或不一致提交须人工检查，绝不静默覆盖；不承诺断电事务恢复。
`chain.json` 记录段索引、步数、校验和与累积样本数；`load_chain` 拒绝缺段、重叠时间、重复采样、
校验和变化与未完成段；配置/来源身份不一致时**拒绝续算**。分段与连续运行的终态、
标量 r、诊断和抽样逐项相同（`tests/tasks/test_inverse_cascade_diagnostics.py` 断言到 1e-14）。
跨机器只允许复制和汇总，不冒充同一环境的精确续算。

`plan.py` 输出 `matrix.json`/`matrix.md`（加 `--write-configs` 还写出 `plan/configs/<id>.json`）：
资格、512² 稳态、临界区、kfL=16/32/64 各档的
配置、依赖、验收门槛与墙钟／内存／存储估算。成本基准：96²/192²/256²/512² 为实测
秒/步，1024² 以上为 `N²logN` 外推并明确标注 `extrapolated`；总量级（串行）约
2000 小时，另加 30% 审批余量；不是目标主机实测。内存仅占位下界，checkpoint、
初末全场副本及峰值 RSS 须另测。37 条只是部分候选，不是完整 Fig.1–4 研究规范。
高尺度已绑定各自 kf/Re-；旧 `reports/inverse_cascade/plan/configs` 不得执行。
执行必须 `--only ID --approval APPROVAL.json`：审批绑定有效配置、当前代码、资源证据和
全部声明依赖，高尺度另需自身网格/步长资格。门禁只核验具名人工审阅与文件完整性，
不自动证明物理资格。**这些作业尚未提交，本轮未授权生产计算或服务器连接。**

## 尚未建立、不得写进结论

- `mu_c` 的具体数值、`gamma_E`/`gamma_A` 指数、无穷 `Re-` 极限、谱指数；
- 原文 Fig. 3 主图的“临界附近相对涨落增大”；
- 原文 Fig. 4 的临界区谱斜率变化；
- 空间收敛（仅“192²/256²/512² 未分辨出差异”）、统计收敛（八点全部未过稳态筛查）、
  绝对波动发散或长时记忆。
