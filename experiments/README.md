# 实验成果导航

按**独立研究问题**查找，而不是把一个目录、一份配置或一次运行当成一项研究。
一个研究可包含多个格式、步长和重复运行；下表中的计算说明保留原来的执行入口。

**材料状态不等于数值验证状态。** “齐备”只说明展示或正文可读；计算完成、参考敏感性、
空间/时间收敛及长期稳定性的证据与限制，仍以对应研究记录为准。
`—` 表示本轮未制作该材料，不是失效链接。其余研究不会为凑齐导航而生成空文件。
本次只发布成果与所需说明；部分既有数值开发仍在本地未提交工作区。
“待整理”条目提供研究与家族入口，不保证该成果提交单独包含其最新计算配置或全部源码。

## 首批成果

| 实验研究（计算说明） | 问题简述 | 结果展示 | 论文实验描述 | 材料状态 |
| --- | --- | --- | --- | --- |
| [制造解时间收敛](manufactured_convergence/README.md) | 同一固定 MAC 网格上四种 SDIRK/mr-ccSAV 格式的时间误差；半离散已知解参照 | [Notebook](manufactured_convergence/studies/time_convergence/results.ipynb) | [中文 LaTeX](manufactured_convergence/studies/time_convergence/description.tex) | 两项齐备；已重绘并核对保存来源，非新的收敛证明 |
| [Kolmogorov 自然节点自适应/固定步对照](kolmogorov_adaptive/README.md) | 256²、联合速度/标量控制与同格式固定步的精度及成本比较 | 待补原始数据 | [中文 LaTeX](kolmogorov_adaptive/studies/adaptive_fixed/description.tex) | 正文齐备；生产参考未完全验证，暂不制作 Notebook |

[制造解展示的执行、输入及论文引入](manufactured_convergence/studies/time_convergence/README.md)；
[Kolmogorov 缺失数据及解锁条件](kolmogorov_adaptive/studies/adaptive_fixed/README.md)。
论文描述是中文设置与方法正文片段，不代填未核验结论。

## 物理算例

| 实验研究（计算说明） | 问题简述 | 结果展示 | 论文实验描述 | 材料状态 |
| --- | --- | --- | --- | --- |
| [顶盖驱动方腔](cavity/README.md) | 移动顶盖的稳态/启动过程与网格对比，含保存场动画 | — | — | 待整理 |
| [A/B 外力旋转](forced_rotation/README.md) | 净力矩与零净力矩的长期能量收支、角动量和旋转反转 | — | — | 待整理；历史原始证据有缺失限制 |
| [周期 Kolmogorov 流](periodic_ns/README.md) | 双周期 Fourier 模型的受迫流及保存场动画，不是无滑移 Kolmogorov 对照 | — | — | 待整理 |
| [Channel 压力驱动瞬态](channel_ns/README.md) | 离散 Poiseuille、从静止启动及受扰后的松弛，含保存场动画 | — | — | 待整理 |
| [Channel TS 失稳](channel_ns/README.md#tollmien--schlichting-instability) | 独立连续参照、线性增长资格检查及非线性可视化 | — | — | 待整理；动画不等于长期收敛认证 |
| [Molenaar 受迫方腔候选复现](molenaar_cavity/README.md) | 外力和弱旋转候选下的长期动力学；普通与 SAV 批次身份分开 | — | — | 待整理；归一化和动力学识别仍有限制 |

## 精度与控制器研究

首批制造解和自然节点对照见上表；以下是不同问题，不能仅因 runner 或图形相似就合并。

| 实验研究（计算说明） | 问题简述 | 结果展示 | 论文实验描述 | 材料状态 |
| --- | --- | --- | --- | --- |
| [受迫方腔参考细化](forced_ns_convergence/README.md) | 固定网格、给定观测窗口的时间比较，可按敏感性继续细化参考；注意窗口可能处于定常区 | — | — | 待整理 |
| [三角涡量初值对照](forced_ns_convergence/configs/trig_production.json) | trig_ns 初值与外力的参考解比较，区别于零初值受迫方腔 | — | — | 待整理；共用上述运行入口 |
| [无滑移格点初值与共同参考](no_slip_reference_convergence/README.md) | 指定格点初值、sin(x) 外力与普通 SDIRK3 双参考的时间比较 | — | — | 待整理 |
| [H-valued 外力对照](no_slip_reference_convergence/README.md) | 保持初值等条件，使用夹持流函数外力研究粗步失败和时间误差 | — | — | 待整理 |
| [Fourier 制造解验证](periodic_ns/README.md) | 双周期模型的时间/空间误差及保存读取，不替代无滑移验证 | — | — | 待整理 |
| [Channel 制造解验证](channel_ns/README.md#short-verification-study) | 周期流向/无滑移壁面模型的时间细化与连续 Stokes 空间细化 | — | — | 待整理 |
| [一阶格式验证](imex_euler/README.md) | IMEX Euler 与 IMEX Euler-mr-ccSAV 的六后端精度和约束检查 | — | — | 待整理 |
| [Kolmogorov 统一速度容差 I/PI](kolmogorov_adaptive/README.md#current-design) | 八组方法/控制器、严格输出对齐及同格式固定步阶梯 | — | — | 待整理 |
| [Kolmogorov 速度容差搜索](kolmogorov_adaptive/README.md#tolerance-search) | 短时重复计时下共同容差与分组容差的成本/精度取舍 | — | — | 待整理；旧扫描报告缺失 |
| [Kolmogorov 自然节点逐点参考](kolmogorov_adaptive/README.md#natural-adaptive-nodes-and-optional-reference-errors) | 每条自适应轨迹的接受节点参考误差及其计算开销 | — | — | 待整理 |
| [Kolmogorov 联合速度/标量容差搜索](kolmogorov_adaptive/README.md) | 严格输出节点下局部联合控制的 CPU、速度误差和标量取舍 | — | — | 待整理 |

## 性能与工程研究

计时边界及物理问题不同，下列研究不互相替代，也不从短时速度比较推断长期精度。

| 实验研究（计算说明） | 问题简述 | 结果展示 | 论文实验描述 | 材料状态 |
| --- | --- | --- | --- | --- |
| [Legendre–MAC 精度—成本](legendre_mac_benchmark/README.md) | 同一连续制造解的指定精度总成本，包含初始化 | — | — | 待整理 |
| [Legendre 优化前后](legendre_optimization/README.md) | 保存的前后源码等价性、总成本、剖析与内存 | — | — | 待整理 |
| [热启动每步成本](step_efficiency/README.md) | 多模型/格式的编译后每步成本；固定步排除已有分解成本 | — | — | 待整理 |
| [Channel direct/FFT](channel_ns/README.md#same-discretisation-timing-comparison) | 同一 MAC 离散的两种线性后端等价性与成本 | — | — | 待整理 |
| [Kolmogorov 固定/给定步后端](kolmogorov_adaptive/benchmark_stokes_backends.py) | 改变移位时 direct/tensor 的总成本，含后端初始化 | — | — | 待整理 |
| [Kolmogorov 自适应后端](kolmogorov_adaptive/benchmark_adaptive_backends.py) | 后端替换是否保留接受/拒绝、步长建议及最终速度 | — | — | 待整理 |
| [矩阵自由 Stokes](kolmogorov_adaptive/benchmark_matrix_free_stokes.py) | 同一标量 Schur 算法下的稀疏/模板算子路径 | — | — | 待整理 |
| [矩阵自由完整积分](kolmogorov_adaptive/benchmark_matrix_free_integration.py) | 算子路径的收益是否保留到完整时间推进 | — | — | 待整理 |
| [变换缓冲区复用](kolmogorov_adaptive/benchmark_transform_buffers.py) | 算子路径固定时可覆盖临时缓冲区的成本与等价性 | — | — | 待整理 |

这些脚本的问题、控制条件和计时边界见各自源码说明；[家族入口](kolmogorov_adaptive/README.md) 保留基础计算约定。后续归档须保留脚本依赖和来源快照。

## 历史研究与外部输入

| 实验研究（计算说明） | 问题简述 | 结果展示 | 论文实验描述 | 材料状态 |
| --- | --- | --- | --- | --- |
| [无滑移旧参数组](no_slip_reference_convergence/README.md) | 单位方域夹持梁初值、分数 k 与共同参考的旧时间比较 | — | — | 待整理；保留旧配置身份 |
| [Kolmogorov 分组 T=30 调参](kolmogorov_adaptive/README.md) | 与同格式固定步比较后选取已测参数，并保留混合来源说明 | — | — | 历史选择记录缺失 |
| [共同步长与增长上限](kolmogorov_adaptive/run_common_step_limits.py) | 固定上限与倍率范围下，分组参数的精度和成本对照 | — | — | 历史选择记录缺失 |
| [初始/优化源码对照](kolmogorov_adaptive/benchmark_initial_vs_optimized.py) | 跨源码快照的十步总成本及最终场差异 | — | — | 外部历史基线缺失 |

## 使用与保留规则

- 首次计算选 smoke 或短时验证；生产配置可能耗时数小时至数天。计算写入 `runs/`，分析只消费显式保存记录并写入 `reports/`，展示不会自动启动计算。
- 共享 `workflow.py`、`cli.py`、`analysis.py`、`plotting.py` 负责运行和分析；`problems.py` 仍供测试和性能工具使用，不是废弃的独立实验。
- [2026-09-24 记录盘点](../docs/experiment-inventory.md) 是历史证据索引，不是当前研究/材料状态清单。
- `workflow.provenance()` 将所有求解器、实验和工具 Python 文件的路径及内容纳入身份；项目配置和锁文件也在身份内。新增可选 Notebook 依赖不会修改数值公式，但会产生新的来源身份。
- 修改前保留精确源码和环境；旧 `runs/`、`reports/`、配置、校验和及失败前缀保持原样。旧批次扩展/回放仍需匹配原源码，不能用新快照冒充旧来源。
