# 三角涡量初值：精度与粗步长行为

[结果 Notebook](results.ipynb) · [中文论文描述](description.tex) · [收敛结果专题](../../../convergence-results.md)

## 固定输入

- 批次：`20260922T050733-f2dbf5c7`，位于 `runs/trig_ns_convergence/batches/`。
- 报告：`20260922T054144-5aee3c91`，位于 `reports/trig_ns_convergence/`。
- 数值表：该报告的 `tables/errors.csv`；18 个被测运行、2 个参考，共 54 行观测。
- 256²、(0,2*pi)² 齐次无滑移、nu=0.001、gamma=5、T=4；步长 `2**(-k)`，k=0...8。
- 保存配置的外力为 `[amplitude*cos(x), 0]`，amplitude=1；初值身份为 `weighted_trig_vorticity`。不是周期边界问题，也不是零初值受迫方腔。
- 当前初值工厂不用于反推或替换历史初态；历史求解来源由捕获批次和成员中的相同源码摘要标识。

## 执行

环境及执行方法与 [历史方腔展示](../forced_time_convergence/README.md) 相同，只把其中 Notebook 路径改成：

```text
experiments/forced_ns_convergence/studies/trig_time_convergence/results.ipynb
```

Notebook 中报告、误差表及全部输入均固定身份并校验；观测时刻为 1、2、4。原始记录缺失、损坏或参考检查未通过时拒绝展示，不启动求解器。

## 必须保留的现象

普通 SDIRK2 的 k=1...5 五个运行失败，对应 15 行观测无有效误差。失败位置画在坐标底部，不能把叉号当误差值；NaN 使折线断开。k=0 虽记录完成，但 T=4 的 L2 误差约 1.68e21，必须保留，不能因 `complete` 而当成准确结果。

SAV 粗步结果虽保持有限误差，也不能当作准确解。k=6...8 的局部放大展示两格式的近二阶趋势，不代替全区间；跨越粗细区时约 9.74 等异常大观察阶不是方法阶数。

该报告使用 k=11/12 的共同 SDIRK2 参考细化对；六项保存的敏感性检查通过，但不是严格误差界。更早的 `20260922T052514-1dacc8ed` 报告未通过参考检查，不能混用其表与此报告的验证状态。

中文片段由加载 `amsmath, amssymb` 的中文主文件以 `\input{.../description.tex}` 引入。没有重新计算场误差、模拟轨迹或性能结论。
