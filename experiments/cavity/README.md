# 顶盖驱动方腔流：Re=100

目的：展示经典主涡/角涡结构，检查速度中心线的网格敏感性与终态残差。单位方腔，上壁切向速度 U=1，另外三壁静止，所有壁面法向速度为零；ν=0.01，Re=UL/ν=100。由静止启动，采用 SDIRK2-mr-ccSAV，dt=0.02，T=40。配置可编辑，Re由实际参数导出。

移动壁仍满足相对于壁面的无滑移。顶角速度不连续是标准方腔算例的特征；面自由度不包含顶角的切向速度。以 u_ghost=2U-u_inside 消去顶部虚拟点，只需在最上排自由 u 面添加 2νU/hy²。原 K 不变。当前守恒/反对称对流的墙面通量乘以法向速度零，故顶盖的切向虚拟值不改变这项通量。它是边界黏性载荷，不是物理体力。

这项扩展独立于原齐次壁面的理论证明。已有 h1_seminorm_squared 是齐次矩阵二次型；在移动壁算例中不能直接解释为带边界贡献的物理耗散。

```sh
uv run python experiments/cavity/run.py --config experiments/cavity/configs/grid_check.json
uv run python experiments/cavity/analyze.py --runs runs/cavity/<32网格运行编号> runs/cavity/<64网格运行编号>
```

计算和绘图分离。图包含速度大小/流函数等值线、内点涡量，以及两条中心线的网格比较。主涡位置从顶点流函数极小值估计。稳态指标是投影后的连续半离散 NS 右端 L2 范数；没有仅凭 T 或小散度宣称稳态。两网格差是敏感性检查，不是完整网格收敛证明。原始结果、边界配置和来源随运行记录保存。

- 正则化顶盖（角点无间断）由 `lid_viscous_load(..., profile=regularised_lid)` 提供，spectral 侧的同一轮廓用于 `LidLifting`；两侧对比见 `docs/spectral.md` 的 S3 一节。

经典文献：Ghia, Ghia & Shin (1982), High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method. https://doi.org/10.1016/0021-9991(82)90058-4 。本次不把两网格比较称为已通过 Ghia 数据验证。

## 启动过程动画

`configs/animation.json` 是独立展示配置：128² MAC、Re=100、SDIRK2-mr-ccSAV、
dt=0.01、t=0 到20，每隔0.1保存一个真实速度快照（201帧，含静止初态）。
不改变上面的原始Re=100配置和已有报告。

```sh
uv run python experiments/cavity/run.py --config experiments/cavity/configs/animation.json
uv run python -m experiments.cavity.animate runs/cavity/<运行编号>
```

动画入口只读校验完整记录及HDF5校验和，从保存的MAC面速度计算速度、瞬时
流函数等值线和内部顶点涡量。白色示踪粒子采用面速度双线性插值、保存时刻间
线性插值与RK2（子步不超过0.005）；它们不是染料方程或新增PDE结果。
移动顶壁与静止另外三壁参与粒子插值，异常越界裁剪次数单独保存。

输出位于独立的 `reports/cavity_animation/<运行编号-导出编号>/`，含MP4、GIF、
五个检查帧、粒子轨迹、元数据/校验和、日志和字体检查。默认20fps，播放约10秒；
时钟标注真实物理时间。颜色范围全程固定，内部涡量以全片绝对值98百分位截色。
静态标签通过XeLaTeX/Times New Roman导出，动态时钟采用同一系统字体，字体由
PDF检查确认。导出需要本机ffmpeg、XeLaTeX、pdffonts；不安装或替换现有依赖。
这是启动过程展示，未增加稳态、Ghia参照或网格/时间收敛验收结论。
