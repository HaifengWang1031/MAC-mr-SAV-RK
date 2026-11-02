# Hele–Shaw翼型

固定翼型贯穿窄缝的MWR深度平均惯性Brinkman模型，复用现有P2/P1和六种时间格式。
首版为压差驱动、翼型/侧壁无滑移、自然压力应力端部，可选出口回流耗散。
经典Darcy作为独立标量压力参照，不使用SAV。方程、压力与模型力定义见[模型说明](../../docs/hele-shaw.md)。

安装可选网格工具（原有Notebook使用者保留对应group）：

```bash
uv sync --extra mesh --group dev --group notebook --inexact
```

先运行短程流程检查，再显式选择目标间隙启动算例：

```bash
uv run --extra mesh python -m experiments.hele_shaw_airfoil.run --config experiments/hele_shaw_airfoil/configs/smoke.json
uv run --extra mesh python -m experiments.hele_shaw_airfoil.run --config experiments/hele_shaw_airfoil/configs/startup.json
```

`startup.json`：c=1、alpha=10°、b/c=.02、nu=.01、rho=1、目标压差2700、爬升时间.02、T=.06、dt=.0005。
该压差对应无翼型、忽略侧壁的名义Darcy速度1；实际平均速度及Re从结果计算。
网格layer_cells=4、surface_points=120；外矩形[-3,6]×[-2,2]。
这是短启动基准，未预设出现涡街、物理失稳或达到严格稳态。

自适应配置设置mode=adaptive、dt=null、initial_step、controller（I/PI）及可选scalar_atol。
自适应只支持已有嵌入格式；原误差控制/拒步/终点对齐规则保持不变，不默认强制最小步长。

分析必须显式给完整运行目录和独立报告目录。需要本机XeLaTeX、Times New Roman；动画另需ffmpeg：

```bash
uv run python -m experiments.hele_shaw_airfoil.analyze /absolute/path/to/run --output /absolute/path/to/report --movie
```

分析读取保存的网格、速度与瞬时物理压力，不需要Gmsh，不推进流动；输出CSV、PNG、JSON和可选MP4。
图像细分每个原三角形为四个显示单元，保留原P2顶点和边中点数值；箭头直接由P2速度评估。
固定色标与真实保存时刻，无时间插值、无Gaussian平滑。

显式六组有界对照（程序写日志，不创建自动监控）：

```bash
uv run --extra mesh python -m experiments.hele_shaw_airfoil.campaign --quick
uv run --extra mesh python -m experiments.hele_shaw_airfoil.campaign
```

正式campaign依次计算alpha10的两档SAV步长、普通SDIRK2参考、细网格SAV和alpha0/-10对照。
粗/细layer_cells=2/4；轮廓点数固定，保证空间比较不混入轮廓改变。
所有原始运行独立保存，父任务目录`runs/hele_shaw_airfoil_campaign/<id>`记录成员与状态。
完整参考求积上的速度L2差用普通ndarray缓冲区和128点一批的精确P2评估，避免大型定位临时数组。
两级网格、单一普通格式参考仍保留明确验证限制。

高Re尾迹探索：`python -m experiments.hele_shaw_airfoil.wake_campaign prepare`
生成带源码清单的独立任务。所有线程环境固定为1后，从任务的`source/`目录执行同一模块的
`pilots --campaign <绝对路径>`。完成后显式保存`pilot_checks.json`，status为
`passed_startup_only`，再执行`production --campaign <绝对路径>`。
使用原虚拟环境的绝对Python路径；冻结源码需有同一线程/库环境。
生产串行计算两组并自动调用只读尾迹分析，日志及原始记录在任务cases目录，
动画、图片和诊断CSV在独立reports目录。没有checkpoint续算或自动重试。

入口角点恢复：`python -m experiments.hele_shaw_airfoil.inlet_recovery prepare`
冻结两个显式成员：原出口+新入口的T=1.1检查，以及新入口+已有回流耗散出口的T=20正式重跑。
在同一单线程/库环境，从任务source目录执行`inlet_recovery start --campaign <绝对路径>`，
要求任务release_checks.json的status为passed。两组均从零初值计算，前者通过后才运行后者。
progress.json由计算程序更新，记录r、节点峰值和入口角点速度；超限保存失败前缀。
`tools.probe_hele_shaw_inlet`为显式状态回放工具，不能当作checkpoint续算。
