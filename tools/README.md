# 工具归属

不要把所有工具当成只读分析，也不要为了清理仓库把任务计算搬进公共后处理。
当前仅选定的公共命令完成去实验依赖；其他文件保留原路径，兼容显式源码选择和历史快照。

| 文件 | 类型 | 依赖/归属 |
| --- | --- | --- |
| `plot_speed_snapshots.py` | 公共只读命令 adapter | 委托 `postprocess.snapshots.main/report`；可独立于实验目录使用 |
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
并非本阶段已具备无实验目录运行能力；不会因未被核心调用而删除它们或移除已有任务测试。
本阶段不改 `.gitignore`、不取消跟踪，未来本地化时需按此清单逐项处理，而非忽略全部 tools。
