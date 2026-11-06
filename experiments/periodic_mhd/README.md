# 双周期不可压 MHD · Orszag–Tang

本实验比较普通 SDIRK3 和每阶段两次固定标量校正的 SDIRK3-mr-ccSAV。
两种格式共用严格 $2/3$ 去混叠的 Fourier 网格、初值和固定时间步。目标是测量
速度与磁场各自的相对 $L^2$ 时间误差，并对照动能、磁能、总能量和总耗散。

## 方程、参数和观测量

域为 $(0,2\pi)^2$，无外力，$\nu=\eta=0.02$，终止时间 $T=2$。
Orszag–Tang (1979) 式 (7) 给出
$u_0=(-\sin y,\sin x)$、$A_0=\cos y+\frac12\cos(2x)$，故
$B_0=(A_{0,y},-A_{0,x})=(-\sin y,\sin(2x))$。
`amplitude` 同时乘初始 $u_0,B_0$，默认 1。幅值 2 用于检验图中初始幅值的线索。
SAV 标量初始为零，`gamma` 是线性阻尼系数，本次取 1。

`kinetic` 和 `magnetic` 是面积平均的动能与磁能，`total_energy` 为其和，
`dissipation` 为 $\nu\langle|\nabla u|^2\rangle+\eta\langle|\nabla B|^2\rangle$，
这里角括号表示面积平均。SAV 阶段配对沿用仓库的物理 $L^2$ 积分，记录的修正能量
将场和标量的修正能量一起除以面积。速度和磁场的散度分别记录，阶段记录保存两场散度
的最大值及完整投影方程残差。

[批次配置](configs/orszag_tang.json) 使用 $96^2$ 采样网格做时间阶，严格保留
$|m_x|,|m_y|<32$。两个格式都用步长 $2^{-6}$ 到 $2^{-9}$，在 $t=1,2$ 比较
与同网格普通 SDIRK3 的 $2^{-12}$ 参考解的相对 $L^2$ 误差。用 $2^{-11}$ 检查
参考敏感性。物理曲线在 $192^2$、步长 $2^{-9}$ 上比较两格式，普通 SDIRK3
另作 $96^2$ 同步长及 $192^2$ 双倍步长的敏感性检查。

## 论文图点的来源

[Orszag & Tang, JFM 90 (1979), 129–143](https://doi.org/10.1017/S002211207900210X)
图 1(c)（印刷页 132）和图 5（印刷页 134）的
[数字化点](reference/orszag_tang_1979.csv)来自附件扫描图。
使用 280 dpi 栅格、坐标轴刻度和曲线中心线读数。图 1 先去掉纵轴的 `×10` 标记，
两图随后都除以图 5 的初始总能量 4。模拟值各自除以本次运行的初始总能量。
图点精度受线宽、扫描倾斜和取样间距限制，CSV 小数位用于可复算，图中峰时取最靠近峰顶的采样点。
图 1 在靠近 $t=2$ 时多条曲线交汇，比较点取到 $t=1.8$。

按式 (7) 的初值，面积平均总能量为 1，初始耗散为 0.07。图 5 与图 1(c)
对应读数为 4 和约 0.28。以原文式 (5)–(6) 计算，式 (7) 振幅给出
$(R^v,R^m)\approx(92.1,45.1)$，两场振幅乘 2 给出 $(232.1,113.7)$，
与文中列出的 $(232,114)$ 一致。幅值 2 因此有独立于演化曲线的依据，
实验中作为单独的幅值敏感性对照。

## 运行和只读分析

```sh
uv run python -m experiments.periodic_mhd.run --study --config experiments/periodic_mhd/configs/orszag_tang.json
uv run python -m experiments.periodic_mhd.analyze --batch /absolute/path/to/batch.json
```

计算入口打印实际批次路径。每个成员在 `runs/periodic_mhd/<run-id>/` 保存有效配置、
结果、来源、校验和与 `run.log`。批次只包含显式运行路径。分析只读取完整记录，
在 `reports/periodic_mhd/<batch-id>/` 写入时间误差表、论文曲线对照表、PNG 和摘要。
HDF5 的磁场在 `final/magnetic` 和 `snapshots/magnetic`，分量顺序为 $B_x,B_y$。
`--config` 也可传入单次运行的参数 JSON，默认物理设置见 [run.py](run.py)。

时间阶限于此固定网格与参数的半离散问题。论文对照的精度受扫描图和初值幅值差异限制。

双倍振幅下的 [SAV 半步补算](configs/orszag_tang_amplitude_2_half_step.json) 使用相同的
$192^2$ 网格和 $2^{-10}$ 步长，可单独运行。已完成批次、结果及边界见
[验证记录](../../docs/validation.md)。

## Zhou 等（2019）的磁岛合并

在同一周期 MHD 模型上比较两种时间格式是否给出相同的磁能衰减、磁岛数量下降和谱峰迁移。
使用 [Zhou 等（2019）正式版](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.1.012004)
的周期方形、零初速度和背景磁通 $A=\cos(8x)\cos(8y)/8$。从磁通生成无散磁场，
取 $S_0=125$、$\nu=\eta=1/(8S_0)=0.001$。原文只描述为小幅空间随机扰动；
这里在磁通的 $|k_x|,|k_y|\le4$ 非零模中加入种子为 2019 的扰动，使扰动磁场
RMS 是背景场的 $10^{-4}$。保存两种格式的实际初始场并逐元素核对。

[首轮配置](configs/zhou_125.json) 用 $256^2$ 采样点、$\Delta t=1/32$ 和 $T=80$，
每单位计算时间保存一次磁场、零均值磁通 `snapshots/flux` 和电流 `snapshots/current`。
这档网格用于探索，论文的 $S_0=125$ 用了 $2048^2$。普通格式的半步长检查仍使用相同
初值、网格和输出时间。单独的 $384^2$ 尝试用于观察分辨率与可用步长的关系。

```sh
uv run python -m experiments.periodic_mhd.run --config experiments/periodic_mhd/configs/zhou_125.json --scheme sdirk3
uv run python -m experiments.periodic_mhd.run --config experiments/periodic_mhd/configs/zhou_125.json --scheme sdirk3_mrsav
uv run python -m experiments.periodic_mhd.analyze_zhou ORDINARY_RUN_ID SAV_RUN_ID --output reports/periodic_mhd/zhou_125
```

只读分析按磁能、磁通极值和磁能谱峰比较两种格式。谱以最近的整数 $|k|$ 为壳层，
按实 FFT 半谱权重求和，取非零壳层的峰。O 点数是周期网格上的局部极大值和极小值
个数，阈值为各帧 $\max|A|$ 的 10%。计数降到 128、64、32 等阈值并持续至少三帧时，
用第一次越过阈值的相邻快照作为合并阶段的时间区间。区间衡量的是全局计数变化，
具体岛的重联过程用磁通等值线和电流快照核对。

[图 2 蓝色曲线](reference/zhou_2019_fig2_s125.csv)取自正式版 PDF 第 3 页主图
的矢量折线。用主图的十进对数刻度校准横轴 $(t-t_0)/\tau_0$ 和磁能纵轴，
排除裁剪区外的顶点。数据精度受发表图的折线和坐标变换限制。
论文给出 $S_0=1786$ 的 $t_0=4$，正文未给出 $S_0=125$ 的偏移。
对照图保留论文的偏移时间，计算曲线用未平移的 $t/\tau_0$，其中
$\tau_0=\sqrt{S_0}\pi^2/32$。两者的横向差异包含启动时间和扰动的影响。
