# 公共只读后处理

本 module 只消费**显式保存的记录**。不导入实验配置、计算入口、任务工具或测试；
不构造动力模型、生成网格、求解 Stokes 或推进时间。`run_support` 负责记录完整性，
这里负责支持的保存场解释、渲染及报告；计算与绘图不放在同一 module。

## 接口与归属

| 入口 | 职责与限制 |
| --- | --- |
| `mac.read_run_fields(path)` | 经 `load_record(require_complete=True)` 校验后返回旧五元组：config、MACGrid、times、fields、history |
| `plotting.field_figure` / `history_figure` | 已给定字段/诊断的图；不读取运行、不求解流动；历史可以只有一个已识别面板 |
| `plotting.latex_binary` / `usetex_available` / `latex_style` | PATH 与标准 TeX Live/MacTeX 位置的工具查找、字体样式和上下文恢复；缺 TeX 默认回退 mathtext |
| `snapshots.report(run, root=..., name=..., fields=...)` | 完整 MAC 快照的图、统计 CSV 和 analysis.json；缺少快照拒绝报告，不补算 |
| `diagnostics.analyze_runs(inputs, root=...)` | 沿用通用固定步 MAC 诊断记录接口；支持显式失败接受前缀，CSV 保留输入状态，不把失败标为运行成功 |

快照 CLI（不需要 `tools/` 或 `experiments/`）：

```sh
uv run --locked python -m postprocess.snapshots --run /path/to/run \
  --root /path/to/output --name saved_fields --fields speed vorticity
```

原命令 `python tools/plot_speed_snapshots.py ...` 仍可用，它与模块命令共享同一个 `main`，
不维护另一份参数解析/报告逻辑。`experiments.plotting`、`experiments.analysis` 保留旧名
adapter；当前源码的新调用方直接导入这里。固定两种格式/斜率二的 `reference_error_figure`
仍属实验策略，不作为通用绘图入口。公共字段图、历史图和诊断报告覆盖完整创建/保存
生命周期：继承 usetex=True 时自动进入可用性检查/回退上下文，完成或异常均恢复调用者
样式与 PATH；原本未启用 TeX 时不因系统上检测到工具而自动启用它。

## 支持的 MAC 记录

- 读取已有配置 `nx/ny/lx/ly` 和快照数组，保留 `u=(ny,nx+1)`、`v=(ny+1,nx)`；
  标记缺省的历史 MAC 记录仍可读。显式 `discretization/field_layout` 仅接受
  `mac/staggered_mac`；边界仅为历史 `homogeneous_no_slip` 或
  `moving_top_lid_stationary_other_walls`。其他标签必须走对应家族的 reader。
- 校验快照数量、形状、有限时刻/字段和原有请求/实际时刻对齐规则，不静默截短 `zip`。
  没有快照的记录可用于读取历史，但不能生成快照报告。
- speed 是**格心平均速度的模**；vorticity 是**内部 MAC 顶点**上的
  `dx(v)-dy(u)`，不是另一个格心字段，也不补壁面旋度。
- Fourier、Channel、FEM、MHD、专属流式记录不是该 reader 的输入；不自动猜测或转换布局。
  阶段压力不被解释为物理端点压力。这里的完整性/布局检查也不是数值或物理资格认证。
- diagnostics 沿用原固定步 MAC 的配置/诊断字段（含 scheme、nx/ny、actual_steps 和
  kinetic/modified_energy/divergence_inf/r）；不承诺通吃所有专属或自适应记录。

## 来源、输出与兼容

报告写入 `root/reports/<name>/<id>`，不是输入运行目录。保持原图/表名称和已有报告字段；
快照报告仅复制已经记录的描述性参数，不要求研究专属的 force、initial_condition 或
experiment 键。输入的 manifest、有效参数、结果和历史来源不重写、不回填。

新报告的 `source` 为显式版本化清单：核心 solver/run_support、环境文件加 `postprocess/`。
因此记录了实际解析、渲染、参数解析与报告实现，不因无关本地研究文件变化而改变。
旧工具路径只是无额外行为的 adapter；分析来源与输入计算来源分别看待，不用新分析来源
重算旧运行身份。当前仍采用广泛来源的计算/快照入口另把 postprocess 附加进旧有序清单，
防止抽取后漏记真实依赖；没有该目录的真实旧源码树仍按原字节编码读取。

## 验证

```sh
uv run --locked pytest -q tests/core/test_postprocess.py tests/core/test_plotting.py
uv run --locked python -m tests.core_distribution
# 只有完整本地研究文件存在时，另验旧 experiments 导入：
uv run --locked pytest -q tests/tasks/test_postprocess_compatibility.py tests/tasks/test_record_compatibility.py
```

已知保存数组验证格心模与内部旋度；报告核对统计值、显式来源及输入不变；独立副本实际
执行 CLI 并禁止研究/数值计算模块导入。TeX 测试使用假的可执行路径，只验证发现和样式
恢复，不冒充真实 LaTeX 排版执行。完整范围及跳过项见 [validation](../docs/validation.md)。
