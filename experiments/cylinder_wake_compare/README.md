# 二维 SAV 圆柱尾流对比

目的：在相同几何、网格、入口及时间控制下比较 Re100 和 Re1000 的圆柱尾流。
高 Re 结果明确称为**二维非定常尾流**。不强制产生不规则图像，也不由二维涡量图认定三维湍流。

## 默认实验

| 参数 | 数值或定义 |
|---|---|
| 几何 | DFG2D-2：2.2 × 0.41，圆心 (0.2, 0.2)，直径 D=0.1 |
| 边界 | 抛物线入口，峰值1.5、均速1；壁面及圆柱无滑移；梯度形式自然出口 |
| Re / 黏性 | 100 / 0.001；1000 / 0.0001 |
| 空间 | P2/P1 FEM，resolution=1，96条圆周边，六阶求积 |
| 时间 | SDIRK3-mr-ccSAV，gamma=1000，r0=0；PI 自适应至 T=12 |
| 控制 | 速度 atol=1e-8、rtol=2e-4；标量 atol=1e-5；最大步长0.002 |
| 初值 | 各自黏性下的相容稳态 Stokes 解；固定入口，不额外加人为扰动 |
| 动画 | 请求 t=10…12、间隔0.01，共201帧、20fps |
| 显示 | 同一固定无量纲涡量色标 omega D/Umean ∈ [-5,5]；坐标等比例 |

参数集中于 `compute.DEFAULTS`。底层求解器、阶段公式及原实验入口不变。
控制器的局部容差不是长期轨迹误差界。

## 计算、检查和只读导出

1. `uv run python -m experiments.cylinder_wake_compare.compute` 准备并启动两组持久进程，
   打印完整批次路径。可用 `--config FILE.json` 覆盖集中参数。
   准备阶段复制数值源码与环境锁文件并保存摘要；工作器从该冻结副本导入，独立写入各自运行目录。
2. `uv run python -m experiments.cylinder_wake_compare.finish ABSOLUTE_CAMPAIGN_PATH`
   等待本批次工作器终止，核验来源，计算短窗检查，然后导出动画。
   本轮生产批次的此程序已作为独立进程启动；启动信息见 `completion-launch.json`。
3. 完成后，`uv run python -m experiments.cylinder_wake_compare.analyze ABSOLUTE_CAMPAIGN_PATH`
   仅重读具名结果和已保存的检查数组，重新分析/绘图，**不构造流动模型、不启动计算**。

已有单组结果可独立预览：
`uv run python -m experiments.cylinder_wake_compare.preview ABSOLUTE_CYLINDER_RUN_PATH --start 10`。
输出位于 `reports/cylinder_wake_preview/<run-id>/`；保留原生产来源、配置、参考门槛与输入摘要，
不将档案结果替换进新的两组受控比较。预览前后核对全部原始文件摘要不变。

用户需要无网格、较光滑的展示时，可在该命令加`--smooth`，输出单独保存至
`reports/cylinder_wake_smooth/<run-id>/`，保留原诊断预览。
先将保存P2速度的旋度作连续P1的L2恢复，再采样到规则栅格；默认使用物理长度
sigma=0.008的掩膜归一化高斯滤波（可用`--smoothing-sigma`修改）。圆柱内部保持遮罩，
连续色彩显示不画网格线，升力与实际时间不作滤波或插值。色标使用平滑涡量的波浪号，
动画明确标注仅供显示，全部显示参数和源摘要见analysis.json。
这种显示变换不消除原场振荡，也不提供新的空间精度资格。

批次位于 `runs/cylinder_wake_compare/<id>/`；原始场分别位于
`case_i/runs/cylinder_ns/<id>/`。`case_i/result.json` 记录成功或失败及原始结果摘要，
进度见原始 `run.log`。程序检查当前数值源码仍与冻结副本一致后才执行短窗检查，
不会把后续改动冒充原计算来源。中断计算不会自动续算。

分析输出位于 `reports/cylinder_wake_compare/<id>/`，包括并排 `cylinder_wakes.mp4`、
完整 GIF、较小的 `preview.gif`、首/中/末帧和 `analysis.json`。
`delivery.json` 的 `status=complete` 且对应分析摘要校验通过，才表示整套交付结束。
原始计算完成、检查完成与动画导出失败分别保留，不相互冒充。

快照使用各自实际接受节点的原场；两行时钟分别显示实际时间，不进行时间插值。
涡量为实际保存仿射 P2 场的单元平均旋度，先计算再显示；不以速度模长替代涡量。
静态图层用 XeLaTeX/amsmath 和 Times New Roman，并检查 PDF 的实际嵌入字体。

## 验证与限制

- 完整原始加载器核验实际网格、P2边自由度、配置/结果校验和、快照和观测序列。
  动画额外检查完整弱连续性、阶段残差和标量漂移；物理逐点散度另行记录。
- Re100 单独比较项目的 DFG2D-2 参考门槛；Re1000 不套用 Re100 的参考。
  这批 resolution=1 网格为动画探索用途，不继承既往细网格的验收资格。
- 从同一末态出发推进0.05：分别收紧时间控制、将 bulk resolution 增至1.25倍。
  圆周边数不变，细网格初态经插值及弱投影；保存转移误差和可独立重算的加权 L2 样本。
  预设短窗门槛为时间1%、空间5%，通过与否逐项显示；不冒充全轨迹空间收敛或湍流验证。
- Re1000 即使得到近周期涡脱落也保留真实结果。
  [DFG2D-2 官方基准](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html)
  给出 Re100 设定；[高 Re 二维基准研究](https://arxiv.org/abs/2507.09274)
  讨论高 Re 下充分分辨的周期解以及欠分辨带来的不稳定性。

## 本轮生产记录

2026-10-28 批次 `20261028T075552-62cb32db` 已在本机启动；实际状态以该批次文件为准。
两组原始计算和自动完成程序独立于聊天持续运行。此处不预先声称完整计算、参考门槛或短窗检查通过。

用户恢复 EIT-WSL 端口后，另取回档案 `20261027T204150-dd47fd10-9ea24b9f`，
四文件远端/本地 SHA256 一致。该记录为 Re100、resolution=1/N48、gamma=1，
与新批次 N96/gamma1000 不同。严格加载及弱约束/阶段残差检查通过；末三周期稳定，
但升力极值参照误差约10.28%–11.14%，参考门槛未通过。
已只读生成 t≈10…12、61帧的独立 MP4/GIF，作为低 Re 涡街预览，不声明精度验收。

## 用户指定的强制下限重跑

原 Re1000 批次以 `Step below min_step` 失败后，用户明确要求低于下限时强制使用下限。
圆柱运行配置可显式设置 `min_step_policy="force"`：普通建议截断到既有下限1e-7，
在此步长上允许有限的超容差试步被接受。NaN/Inf、阶段求解或诊断失败仍停止；
最后一个短于下限的终点对齐步仍检查原容差。
默认 `fail` 和未显式配置的旧实验不变。

实际误差不改写，`attempts/error_u`、`error_r`、`error`、`accepted` 和 `step` 保留每次决定；
`manifest.metrics.forced_min_step_acceptances` 统计超容差接受次数。
继续计算不等于维持原局部容差，也不自动取得精度或稳定性资格。

新批次 `runs/cylinder_floor_retry/20261028T091327-5ee92573/` 保存配置和冻结源码，
部署在 EIT-WSL `/home/hfwang/mac-mrsav-cylinder-floor-20261028T091327-5ee92573/`。
从 t=0 的原相容 Stokes 初值重算至12，不声称从失败 PI 历史精确续算。
物理、网格、SAV和误差控制参数不变；额外保存早期输出点便于诊断，日志间隔100步。
进度见远端 `runs/cylinder_ns/<run-id>/run.log`，最终以 `result.json` 及成员 manifest 为准。
本地记录不将未取回的数据冒充已下载结果；旧失败结果保留。

## 回流耗散出口的后续测试

用户要求进一步计算后，可选 `outlet_treatment="backflow_damped"` 已正式接入，
定义、改变的数学条件和验证限制见 [出口说明](../../docs/backflow-outlet.md)。
新旧边界不可混合成同条件比较，也不继承自然出口的基准资格。

`runs/cylinder_outlet_validation/20261028T113939-81d3c4c2/` 从同一 t≈1.40003 状态，
用2e-4/1e-4各推进至约1.60003，两组完成；速度物理L2相对差约0.1846%，
细步最大速度7.122、最大|r|≈6.845e-5。仅为同一粗网格的短窗时间敏感性检查。
通过后，新T12批次 `runs/cylinder_backflow_retry/20261028T114139-60ff3b11/` 已在EIT-WSL启动，
从原初值重算。完整结果与空间精度尚待验证，原失败数据保留。

粗网格随后完整到T12，但涡量有明显网格尺度振荡。用户要求加密后，新批次
`runs/cylinder_mesh_refine/20261028T234434-c3cb0832/`仅将resolution改为2，
采用相同冻结源码、96边几何、回流耗散出口和SAV控制，从各自相容Stokes初值重算。
5480→17321三角形、21638→68912自由速度自由度。短程T=.02流程/资源检查通过后，
北京时间2026-10-29 07:47:31已在EIT-WSL持久启动正式T12。批次`STATUS.md`记录日志、
基线及只读比较方法；完整结果和振荡是否减弱仍待核验，不能由启动或短程成功预先断言。

2026-10-30完成核验：该加密组于08:59:44严格到T12，52584接受/1拒绝、强制下限0次，
耗时25小时12分13秒。已取回并校验原始数据，完整只读比较见
`reports/cylinder_mesh_refine/20261028T234434-c3cb0832/comparison/RESULTS.md`。
晚期散度L2中位数下降75.33%、中央尾流涡量跳变指标下降52.68%；涡街轮廓更清楚，
但网格尺度振荡仍明显，局部物理散度峰值仍约1891，未取得空间精度资格。
加密诊断预览位于`reports/cylinder_wake_preview/20261028T234732-5cdb446e-6c98f104/`。
