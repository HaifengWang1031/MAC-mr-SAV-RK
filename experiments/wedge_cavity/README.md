# 顶盖驱动楔腔

等腰三角形上边为向右运动的水平顶盖，两条斜壁静止且无滑移。`apex_angle` 是完整顶角
`2 alpha`，`lid_length=L`，深度 `H=L/(2 tan(alpha))`，`Re=U0 L/nu`。
空间离散为相同网格上的 P2 速度/P1 压力；保留一致质量、完整弱连续性和压力物理积分零均值检查。

当前 `scheme` 可选注册表中的六种共享格式，默认 `sdirk2_mrsav`；本实验入口仅支持
固定 `dt` 且要求其整除 `T`，尚未开放给定步长序列或 I/PI 自适应。共同 FEM 模型及
推进器的更广支持范围见[项目总览](../../README.md#当前算法与支持范围)。

## 计算与复用

```bash
uv run python -m experiments.wedge_cavity.run --config experiments/wedge_cavity/configs/anim_apex90.json
uv run python -m experiments.wedge_cavity.run --config experiments/wedge_cavity/configs/anim_apex25.json
```

两个动画配置均为 Re=100、顶盖96个单元、SDIRK2-mr-ccSAV、gamma=1、dt=0.005、T=3，
每步保存一次快照，共601张。初始自由速度系数为零，顶盖立即启动；`r0=0`。
顶盖与静止壁共享的两个端点取零，保证整条斜壁的 P2 迹均为零。
每个接受态的时间按固定步序列记录，末态及最后快照严格取 T；最后快照不依赖输出步幅是否整除步数。

移动顶盖使用完整速度 `U=w+g`。SAV 返回同一反对称求积的 `N(w)` 与 `b(U,U,g)`，
满足 `<N(w),w>+b(U,U,g)=0`；顶盖法向流量为零不能用于省略这个体积配对项。
这里沿用共同时间格式，不修改根选择、阶段数或完整残差门槛。

2026-11-01 之前的探索记录曾使用 r=1、两个顶盖端点随盖运动，并省略 lifting 配对。
旧记录文件保留；读取时用显式的 legacy 默认值恢复其实际约定，不将它们改称 r0=0 的计算。
正确性修正与性能对照分开：性能参考源码也包含上述修正。

## 从保存结果生成图像和动画

```bash
uv run python -m experiments.wedge_cavity.report /absolute/path/to/run
```

`report.render_frames` 保留原接口，内部复用静态版式、色条和三角化，更新数据与流线，
将帧直接送入 ffmpeg。默认同时保存 MP4、GIF、首/中/末帧和带输入摘要的 JSON。
分析只消费显式运行编号并校验记录；不调用时间推进或流动 Stokes 求解。
静态文字用 XeLaTeX/Times New Roman，并检查导出的字体。
需要 ffmpeg、XeLaTeX、pdffonts 能从 PATH 找到。

左图为全腔速度及 Poisson 恢复的流线，右图为涡量与流线。深楔腔右图放大顶盖附近，左图仍保留完整深度。
速度显示为 `|u|/U0`，涡量显示为 `omega H/U0`；色标固定，超界只作颜色饱和。
每帧使用实际保存时刻，不作时间插值，最后可短暂停留。流线是离散速度的 Poisson 恢复，
不是直接积分的粒子轨迹。这里展示启动过程，不能据动画断言已达到稳态或解析了所有角点涡旋。
阶段压力不在动画中作为物理端点压力展示。

本轮独立物理配对、壁面迹、公开保存/读取和无流动重算的动画检查见 `tests/test_wedge_cavity.py`；
性能与验证边界见 `docs/validation.md` 及 `runs/wedge_optimization/` 对应记录。

## 长时间与Re/网格对照

2026-11-01已按用户确认启动25°、T=30、Re100/400/1000与96/192顶盖单元的六组对照。
固定dt=.005、SAV格式和边界控制；完整冻结源、配置、短检、后台驱动和只读分析位于
`runs/wedge_longtime/20261101T103104-dd08c0e5/`。状态/日志见其中STATUS.md，分析产物在同编号reports/wedge_longtime下。
每个case6000步、601个保存时刻，数值库单线程，最多两个独立case并行。

`render_frames(..., detail='lower_cell')`保留全腔对数速度与正负分开的流线，并将右图用于下方正流函数区域；
该区域先作为候选，独立速度零点及梯度检验在对比报告中另行保存。两级网格仅用于敏感性检查，
不足以单独证明收敛；长时间或更大Re也不预设下涡更强。
