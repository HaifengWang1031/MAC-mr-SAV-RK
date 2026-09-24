# 带外源 NS：固定步长精度比较

目的：同一128² MAC网格上，对比 SDIRK2 和 SDIRK2-mrSAV 的时间误差，以小步长 SDIRK2 为共同参考，不构造精确解。

生产配置集中在 `configs/production.json`：单位方域、齐次无滑移、nu=0.01、gamma=1、r0=0、零初值（`ic_kind="zero"`）、定常外力 `[0, amplitude*sin(2*pi*m*x)]`，其中 `m=2` 即 `f=(0,sin(4*pi*x))`（`m` 为跨越单位区间的完整波数）。`amplitude` 只缩放外力；x是物理坐标，不含额外2π。步长为 `tau_base*2**(-k)`，k=1,...,8，观测时刻1、1.5、2。每次推进到T=2，只保留指定全场快照、末态和标量/阶段记录，不保存全部速度轨迹。smoke 配置在 16² 上镜像同一套物理参数。

初值有两种，由 `ic_kind` 选择，选择结果写进每次运行的 `config.json`：`zero`（恒零速度、r0=0）或 `isotropic_beams`（`experiments/forced_ns_convergence/isotropic.py` 的各向同性无滑移夹持梁流函数，参数 `ic_k_lo`/`ic_k_hi`/`ic_alpha`/`ic_seed`/`ic_target`/`ic_value`/`ic_symmetric`/`ic_project`，波数带受 `k_hi <= min(nx,ny)/8` 限制）。用各向同性初值时，**同一批次内所有步长级别与参考运行必须共用同一个 seed**，否则比较的是不同流场。`docs/validation.md` 记录了该初值的频带/寿命关系以及 `m=2` 零初值算例在 T≲1 就进入定常的实测结果。

smoke 配置用 16² 网格并镜像 `m=2`、零初值；若要用各向同性初值，16² 下波数带只能取 K∈[1,2]（8² 连最小的 (1,1) 模态都超过 `min(nx,ny)/8` 的分辨率上限）。`docs/validation.md` 记录了频带/寿命关系：`nu=0.01` 时 K∈[4,12] 的黏性寿命约 0.03–0.04，因此该初值只影响 T≲0.1 的瞬态，观测窗口 [1,2] 看到的是受迫状态。

参考 SDIRK2 从k=12开始，以k=13检查敏感性；两个范数、所有观测时刻的参考差值/最小有效被测误差均须≤5%。不满足则移动参考对并继续细化，最多额外两次；达上限仍失败标记 `reference_unresolved`，不伪称参考充分。这个指标不是严格参考误差界。同网格对照主要检验时间误差，不证明空间精度。

## 执行

```sh
uv run --locked --extra mpi python experiments/forced_ns_convergence/run.py --config experiments/forced_ns_convergence/configs/smoke.json
uv run --locked --extra mpi python experiments/forced_ns_convergence/run.py --config experiments/forced_ns_convergence/configs/production.json
```

首轮生产包括16个被测运行和2个参考运行，参考对共有245760步；参考继续细化时成本显著增加。运行保留现有来源/校验和复用机制，显式 `--rerun` 新建运行，不覆盖旧结果，不支持断点续算。失败保留接受前缀，其后观测时刻为NaN。

- 逐次运行：`runs/forced_ns/<run-id>/config.json, manifest.json, results.h5, run.log`。
- 批次：`runs/forced_ns_convergence/batches/<batch-id>/batch.json, run.log`，列明全部成员和参考路径。
- 每1000个接受步写进度，配置为 `base.log_every`；整体启动输出列明当前运行目录。
- 完成条件：批次最后一行 `FINAL status=complete`；有粗步长失败时为 `complete_with_failed_trials`；参考检查不通过为 `reference_unresolved`；计算/分析异常为 `failed`。失去进程时可能保持running，不能认为完成。

计算入口在显式参考成员保存后调用分析；独立分析入口仅加载指定批次，不触发计算：

```sh
uv run --locked --extra mpi python experiments/forced_ns_convergence/analyze.py --batch /absolute/path/to/batch.json
```

`reports/forced_ns_convergence/<analysis-id>/` 中包含analysis.json、analysis.log、误差CSV、L2.tex、H1.tex和误差/成本/r图。表格按T分组，每组两种格式各Error/Rate，k只取整数。Rate仅从相邻k的有限正误差计算，否则为--；缺失误差写NaN。H1半范数采用差值的离散黏性二次型，不等同于压力或涡量误差。成本含预热、分解和推进，不能直接当作纯时间推进加速比。LaTeX使用booktabs与graphicx。
