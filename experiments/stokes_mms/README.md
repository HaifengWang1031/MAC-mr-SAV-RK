# stokes_mms

目的：验证无滑移 Stokes 速度和零均值压力的空间收敛。

在项目根目录运行：
```sh
uv run python experiments/stokes_mms/run.py --config experiments/stokes_mms/configs/default.json
uv run python experiments/stokes_mms/run.py --config experiments/stokes_mms/configs/spatial_scan.json
uv run python experiments/stokes_mms/analyze.py --runs runs/stokes_mms/<运行编号>
```

配置中的 `nx,ny,lx,ly` 定义矩形网格，`nu` 是黏性，`amplitude` 是流函数幅值，`T` 是终止时间，`dt` 是固定步长（末步可缩短）。用 `steps` 正数列表替换 `dt` 即可给定步长序列，列表总和必须等于 `T`。`scheme` 为 `sdirk2` 或 `sdirk2_mrsav`，`gamma` 是标量阻尼，`cache_size` 为 LU 缓存上限。`snapshots` 请求最近接受节点，结果记录实际时刻。

重复运行自动复用通过校验的完整记录；显式加 `--rerun` 创建新尝试。失败记录不会被复用。不支持断点续算。

Stokes 实验忽略时间推进配置，直接求定常解。NS 制造解实验的误差是连续面速度的加权离散 L2 误差；阶段压力不能直接用来验证物理压力的时间阶。时间分析需 `--reference <参考运行目录> --refined-reference <更细参考目录>`，两者与被测解须同网格、同格式及同终止时间。
