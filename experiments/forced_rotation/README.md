# A/B 定常外力：长期有界性、能量收支与大尺度旋转

目的：用两种定常外力比较**有净力矩**与**无净力矩**情形下的长期行为——是否长期有界、平均输入与耗散是否平衡、以及平均角动量是保持方向偏好（A）还是出现持续的符号反转（B）。

外力来自有界域扭矩研究（`f_A = f0(sin k y, -sin k x)`、`f_B = f0(cos k y, -cos k x)`，`k=5`，`f0=0.1`），在本机上按探索参数降本运行：

| 项目 | 本轮取值 | 论文 |
|---|---|---|
| Ω | `(0,2π)²` | 同 |
| 边界/初值 | 四周齐次无滑移；`u⁰=0`、`r⁰=0` | 同 |
| 外力 | `f_A=0.1(sin5y,-sin5x)`、`f_B=0.1(cos5y,-cos5x)` | 同 |
| 黏性 | **ν=1e-3** | ν=2×10⁻⁴ |
| 网格 | **256²**，512² 复核 | 4096² |
| 格式 | SDIRK2、SDIRK2-mrSAV，`γ=1` | — |
| 步长 | 候选 τ=0.01/0.005/0.0025 | — |

净力矩的闭式值（`2π` 方域、整数 `k`）：`M_A = 2f0/k = 0.04`、`M_B = 0`。离散侧由 `tests/test_forced_rotation.py` 对照该闭式值。

## 记录量

逐步（`results.h5/diagnostics/`，每个接受步一行）：`kinetic`（动能 E）、`modified_energy`、`h1_seminorm_squared`（`|u|_{H1}²`）、`dissipation`（`ν|∇u|²`）、`power`（`(f,u)`）、`angular_momentum`（`∫(r-r_c)×u dA`，域心）、`divergence_inf`、`r`。阶段记录保留 `roots/*`（含候选根、残差、选中 r）与 `stages/*`。全场快照按 `snapshot_every`（长期配置为 5）写入 `snapshots/u,v`。

角动量是**二阶精度的格心求积**（`solver/mac_ns.py::angular_momentum`），壁面速度按无滑移取零，因此对无滑移场是 O(h²) 而非精确值；跨网格比较统计量时这一点是系统性偏差，反转判据用相对阈值抵消。

## 配置与执行顺序

| 配置 | 内容 | 步数 |
|---|---|---|
| `configs/screen_A.json` / `screen_B.json` | 256²、τ=0.01、T=100、快照 25/50/75/100 | 10 000 |
| `configs/long_A.json` / `long_B.json` | 256²、τ=0.005、T=3000、`snapshot_every=5` | 600 000 |
| `configs/refine_tau_B.json` | 同 long_B 但 τ=0.0025 | 1 200 000 |
| `configs/scheme_B.json` | 同 long_B 但用 SDIRK2 | 600 000 |
| `configs/refine_grid_B.json` | 同 long_B 但 512² | 600 000 |

```sh
uv run python experiments/forced_rotation/run.py --config experiments/forced_rotation/configs/screen_A.json
uv run python experiments/forced_rotation/analyze.py --runs runs/rotation_ns/<A> runs/rotation_ns/<B> --start <过渡结束时刻>
```

成本（实测 256²、ν=1e-3 每步 ≈0.124 s 单进程 / 0.14 s 双进程；512² 实测 ≈**0.87 s/步**，约 7 倍，其中分解约 43 s）：短时筛选每个 256² 运行约 21 min（τ=0.01）；T=3000 的 256² 主实验约 **21–23 h/次**，τ=0.0025 约 45 h，512² 约 **6 天/次**。**没有断点续算**（与全项目一致），所以这些长算例必须一次跑完。首轮只跑短时筛选，长时间配置按需启动。

**线性求解器容差**：耦合稀疏直接解在 512² 上的相对残差实测约 1e-8（256² 为 1e-11 量级），因此 `DirectStokes` 默认的 1e-9 门槛在 512² 会直接判定失败（筛选时实测 `Stokes residual=1.164e-09`）。配置新增 `stokes_tolerance`（默认 1e-9，作为运行参数写入 `config.json`）；512² 复核用 1e-7 才有余量，代价是动量残差门槛放宽（散度仍单独检查，实测 ~1e-11）。要在 512² 上同时保持残差与精度，需要迭代/AMG 后端，那不在当前范围内。

## 分析

`experiments/forced_rotation/analyze.py` 只读已有 run，产出三组图：

1. `figures/energy.png`、`figures/angular_momentum.png`（含反转标记）、`figures/history_first_run.png`：能量与角动量时间曲线、功率/耗散/r/散度；
2. `figures/vorticity_<run>.png`：每个 run 的涡量快照（中心大涡的形成、瓦解与重建）；
3. `figures/statistics.png` 与 `tables/summary.csv`、`tables/reversals.csv`：跨步长/网格/格式的统计对比。

反转识别（`detect_reversals`）：符号改变必须使 `|L|` 离开阈值带（阈值 = `threshold_fraction` × 该窗口内 L 的 RMS，默认 0.1）并以新符号持续至少 `min_duration`（默认 1 个时间单位）才计数；从零增长时的首次符号确立不算反转。**报告的时刻是离开阈值带的时刻**，比真正的过零滞后约 `arcsin(阈值/幅值)` 的相位，所以它应读作符号改变时刻的上界。统计窗口用 `--start` 排除过渡段——过渡结束时刻必须由数据（能量与角动量的分段统计）确定，不能预先假定全部数据都是稳态样本。

## 限制

- 论文用 ν=2×10⁻⁴、4096²；本轮用 ν=1e-3、256² 只是探索，**不预先保证能观察到相同的反转现象**。
- 256²、ν=1e-3 下壁面涡量层只有几个网格（与 `docs/validation.md` 中 m=2 算例的实测一致），壁面量是网格敏感量；512² 复核是结论的前提。
- 候选步长必须通过 CFL、阶段残差与精度检查后才能进入长期生产；筛选阶段的结论记录在 `docs/validation.md`。
- “反转附近加密输出”尚未实现：快照按固定间隔记录，需要反转附近的更密快照时，只能先由角动量曲线定位反转，再单独重跑该时间窗。
