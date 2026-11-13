# 工具归属

不要把所有工具当成只读分析，也不要为了清理仓库把任务计算搬进公共后处理。
当前仅选定的公共命令完成去实验依赖并随 Git 检出提供；表中其他五个脚本仅本地保留
原路径，兼容显式源码选择和历史快照，不在干净检出中提供。

| 文件 | 类型 | 依赖/归属 |
| --- | --- | --- |
| `plot_speed_snapshots.py` | 公共只读命令 adapter | 委托 `postprocess.snapshots.main/report`；可独立于实验目录使用 |
| `worktree_check.py` | 公共开发侧只读身份预检 | 只依赖stdlib/Git；核对当前工作区，不导入solver/研究包，不执行后续写操作 |
| `benchmark_steps.py` | 会计算的单步计时 | 使用研究侧制造解；不是只读工具，本阶段不迁移参照或改计时语义 |
| `benchmark_fem_adaptive.py` | 会计算的 FEM 同状态回放/原生自适应计时 | 依赖指定实验 reader、配置与选定源码；保留旧源码兼容导入 |
| `probe_hele_shaw_inlet.py` | 会计算的翼型状态回放探针 | 专属 reader、入口处理与控制器，不是读图工具 |
| `rerun_low_high_stability_probe.py` | 会计算的低高频试算 | 历史基准、制造解、当前格式及源码捕获 |
| `search_low_high_stability_steps.py` | 会计算的步长搜索 | 复用低高频试算，保存每个尝试和批次，不自动放宽资格 |

公共命令的推荐入口不再需要这个目录：

```sh
uv run --locked python -m postprocess.snapshots --run /path/to/run --fields speed vorticity
```

原 `python tools/plot_speed_snapshots.py ...` 参数和 `report` 导入保留；实现、支持的保存
布局和来源契约见 [postprocess](../postprocess/README.md)。其他工具仍属任务/计算侧，
并不具备无实验目录运行能力；本地化只取消这五个脚本的跟踪，未删除文件或任务测试。
不忽略全部tools，公共wrapper、工作区预检与包初始化继续跟踪；默认mypy显式选择公共
Python文件，不把同机的本地计算脚本隐式纳入公共检查。备份范围见[本地化说明](../docs/core-localization-plan.md)。

## 工作区身份预检

在目标目录、干净状态下开始新的修改/分支操作/发布流程前运行。ROOT/PY使用显式绝对路径，
EXPECTED_HEAD填入本次已确认的完整提交SHA：

```sh
# 调用者先处于目标工作区；本命令不会自动cd或切换分支。
ROOT=/absolute/path/to/checkout
PY=/absolute/path/to/existing/venv/bin/python
EXPECTED_HEAD="${EXPECTED_HEAD:?请先设置本次已确认的完整提交SHA}"
"$PY" -B "$ROOT/tools/worktree_check.py" \
  --expect-root "$ROOT" --expect-branch main --expect-head "$EXPECTED_HEAD" \
  --profile core --expect-python "$PY" --expect-upstream origin/main \
  && git -C "$ROOT" status --short --branch
```

示例最后仅查询状态；真正写操作仍需用户授权，并使用成功连接`&&`及显式`git -C`目标。
CLI刻意拒绝dirty：它不是对已有待提交内容的通用pre-commit包装器。本轮已审阅修改的提交
仍须另核对目标身份与精确索引范围，不能用本工具声称批准任意待提交变更。
不要把已确认SHA替换为临时读取当前HEAD，否则会削弱识别错误版本的作用。也可从正确
目录以`python -B -m tools.worktree_check ...`调用；-B避免Python写入导入缓存。

- 必需：`--expect-root`（绝对路径）、`--expect-branch`、`--expect-head`（完整40/64位SHA）、
  `--profile core|full-local`。可选绑定`--expect-python`、`--expect-upstream`。
- 标准输出为JSON，包含actual cwd/root/branch/HEAD/upstream、解释器/环境前缀、布局、dirty
  和错误列表；匹配且干净返回0，不匹配/非仓库/dirty/detached返回2。参数语法错误使用argparse诊断。
- core要求没有已知本地研究目录/五个计算工具。full-local要求experiments含Python源码且
  tests/tasks含测试文件；只有部分目录、空目录或单独计算工具标为mixed并拒绝。这只是布局
  识别，不证明研究文件完整、依赖安装、测试收集或数据恢复资格。
- Git仅执行读取命令，禁用可选索引刷新与fsmonitor hook；拒绝GIT_DIR/GIT_WORK_TREE/
  GIT_INDEX_FILE/GIT_COMMON_DIR环境覆盖。解释器路径不跟随可执行文件符号链接，避免把两个
  使用同一基础Python的venv混为一谈。upstream只检查本地配置，不访问远端或确认同步状态。
- 它不是Git钩子、命令执行器或权限系统；不会拦截忽略退出码的人工操作，也不保证检查后
  工作区不再变化。后续写入仍需要Git自身的非强制/身份检查；本轮不安装hook或改全局配置。
