# 周期 NS 实验

目的：验证 Fourier–Leray 离散对四套已有格式的兼容性、周期速度收敛、辅助变量和自适应行为，并测量本实现的短时成本。计算、存储和分析入口独立于无滑移 MAC 工作流。

## 运行与分析

```sh
uv run --locked python -m experiments.periodic_ns.run --config experiments/periodic_ns/configs/smoke.json
uv run --locked python -m experiments.periodic_ns.run --config experiments/periodic_ns/configs/adaptive.json
uv run --locked python -m experiments.periodic_ns.run --config experiments/periodic_ns/configs/kolmogorov.json
uv run --locked python -m experiments.periodic_ns.analyze runs/periodic_ns/<run-id>
uv run --locked python -m experiments.periodic_ns.analyze runs/periodic_ns/<run-id> --plot
```

入口打印新建或复用的运行目录。`--root` 指定实验根目录；`--rerun` 创建新记录，不覆盖旧记录。分析只读显式目录，无计算启动副作用。默认输出 `analysis.json` 和 `history.csv`；`--plot` 输出速度、谱涡量、能量和 r 的 PNG/PDF，使用 XeLaTeX/amsmath 和 Times New Roman，需要本地 XeLaTeX、字体及 PDF 转 PNG 工具。

`configs/` 是可修改的输入配置，默认展开逻辑集中于 `run.DEFAULTS/effective_config`：

| 参数 | 语义 |
|---|---|
| `case` | `mms` 为连续解析非线性制造解；`kolmogorov` 为周期外力和确定性流函数扰动 |
| `scheme` | 原四格式，另有 `imex_euler/imex_euler_mrsav`（一阶，仅固定/给定步长） |
| `nx,ny,lx,ly` | 采样节点数及周期长度；实际保留严格 2/3 频带 |
| `nu,gamma,r0` | 黏性、SAV 参数、初始辅助变量；MMS 要求 r0=0 |
| `amplitude` | 制造解幅值或 Kolmogorov 外力幅值 |
| `epsilon,k_f` | Kolmogorov 初始流函数扰动幅值、完整周期整数外力波数；k_f<ny/3 |
| `mode,T` | `fixed/adaptive`；从 t=0 积分到 T |
| `dt` 或 `steps` | 固定步长或给定正步长序列，互斥；序列必须总和为 T |
| `initial_step,controller` | 自适应初步及 I/PI 参数；null max_step 表示无穷，估计阶按格式展开 |
| `scalar_atol` | 可选 SDIRK3-mrSAV 局部标量联合控制；不控制累计全局标量误差 |
| `snapshots` | 请求时刻列表；默认 [0,T]，保存最近接受节点及实际时刻，不截断自然步长 |
| `stokes_tolerance,log_every` | 完整线性残差门槛、接受步日志间隔 |

MMS 为 `psi=a(t)[sin X+.2 sin 2X]sin Y`，`p=a(t)cos X cos Y`，`X=2πx/lx,Y=2πy/ly`，`a=amplitude*exp(-t)*(1+.25sin t)`。外力由解析时间、空间导数及解析乘积构造，不调用求解器对流算子。nx>=15,ny>=9 可在严格频带内完整容纳非线性外力。参考速度在保存的实际时刻计算，r 的连续参考为零；压力输出只按阶段乘子解释。

Kolmogorov 外力为 `(-amplitude*sin(k_f Y),0)`；初值是上述单位幅值空间流函数的 epsilon 倍旋度。这是确定性周期扰动，与既有无滑移格点和流函数初值不同，不能混用旧参考。该案例仅报告诊断，没有标记未经验证的参考误差。

## 保存与复用

`runs/periodic_ns/<id>` 保存展开配置、包含代码/环境/线程身份的 manifest、实数 HDF5 场和日志。字段布局明确标记为 `collocated_yx`，读取器不做 MAC 面平均。存储复用通用 `save_result`，另外记录 Fourier 表示标签和自适应所有 attempts。结果先写 pending 文件，再原子改名，最后写完成状态及校验和。只复用校验通过、配置和来源相同的完整记录。失败试算不进入接受状态，保留失败前缀；中断记录不复用，无 checkpoint 续算。

HDF5 包含实际时间及 dt、能量/散度/H1/涡量能量/均值/功率/耗散/r、最终 u/v/r、最后接受步阶段压力、指定快照及其请求/实际映射、每个接受阶段的根候选和残差；自适应另存 error_u/error_r、接受拒绝、原因与步长建议等 trial 数据。u/v/压力均为实数网格场，不向旧 MAC reader 传递周期数据。

## 收敛与短时计时

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run --locked python -m experiments.periodic_ns.validate
uv run --locked python -m experiments.periodic_ns.analyze runs/periodic_ns_validation/<study-id> --validation
```

参数集中于 `validate.DEFAULTS`，可通过 `--config <json>` 覆盖。默认检查：24²、T=.4、10/20/40 步、四套格式、gamma=1/1000；无限模态 `psi=exp(.7cos X+.4cos Y)` 的 12/18/24/32² 定常 Stokes 空间误差；32/64/128/256²、四格式、10 步固定/交替给定序列、三次重复。两种计时序列具有相同总时间 .04。非线性、单/双 RHS Stokes、诊断分别预热后测量。

每个推进成员保存为完整子运行，空间验证保留解析与计算场；汇总分析重新读取场与校验和，检查共同来源和重复次数，输出误差、阶、计时中位数及范围。setup+integration CPU 不包含磁盘保存、导入与绘图。时间指标与标量误差分别记录；短时制造解的验证不能替代无滑移 PDE 理论、长期湍流稳定性或标量高阶证明。没有同周期物理问题的 MAC 基线，因此不报告 Fourier/MAC 加速比。

## Kolmogorov flow 动画

`configs/animation.json` 是独立展示配置：域(0,2π)²、128² Fourier点、严格2/3
频带、nu=0.02、f=(-sin(4y),0)、epsilon=0.1的原确定性流函数初值；普通SDIRK3、
dt=0.005、T=20，每0.1保存一次，共201个真实快照。原adaptive配置不变。

```sh
uv run python -m experiments.periodic_ns.run --config experiments/periodic_ns/configs/animation.json
uv run python -m experiments.periodic_ns.animate runs/periodic_ns/<运行编号>
```

动画入口只读指定完整记录，核验校验和、均匀实际快照间隔、频带及散度。
左侧为速度大小与示踪粒子，右侧为谱微分涡量及瞬时流函数等值线。
psi_hat=omega_hat/k²，零模取0，显示时加上平均速度对应的mean_u*y-mean_v*x。
粒子在保存场间线性时间插值、双周期双线性空间插值，用不超过0.005的RK2子步
推进；跨边界尾迹断开，避免绘制穿越整幅图的线段。它们不是新增染料PDE。

输出为独立 `reports/periodic_animation/<运行编号-导出编号>/` 中的MP4、GIF、
五个预览帧、轨迹、元数据、校验和和日志。默认20fps，播放约10秒；时钟标注
真实物理时间。速度色标全片固定，涡量以全片绝对值99百分位截色，色条端点
标示截色。静态文字以XeLaTeX/Times New Roman导出，时钟用同一系统字体，
pdffonts核验嵌入字体；需要已有ffmpeg、XeLaTeX及pdffonts。后处理不会启动
PDE计算或更改保存场。此动画不增加湍流、网格/时间收敛或参考解验收结论。
