# 收敛性结果专题

[返回全部实验导航](README.md)

本页按研究问题整理**已经保存的结果**。主文候选是固定空间网格上的时间误差，不把它们混同为空间收敛或 PDE 收敛证明。所有报告均显式选择，不按日期自动挑“最新”。

“输入校验通过”表示保存配置、HDF5、捕获批次和报告所列摘要一致；“参考检查通过”指保存报告的细化对敏感性检查。**本轮未独立从速度场重算误差，也未补算参考或失败轨迹。**

## 一、可直接整理的三项研究

| 独立研究 | 保存设置与比较 | 材料 | 数值观察与边界 |
| --- | --- | --- | --- |
| 制造解时间收敛 | 128²、nu=0.01、gamma=1、T=2；四种二/三阶 SDIRK 与 mr-ccSAV；半离散已知解 | [Notebook](manufactured_convergence/studies/time_convergence/results.ipynb) · [中文正文](manufactured_convergence/studies/time_convergence/description.tex) · [使用说明](manufactured_convergence/studies/time_convergence/README.md) | 40 份输入校验通过。二阶结果清楚；三阶 L2 在 k=6、7 约 2.96、2.98，k=8 约 1.63–1.66，误差约 1.2e-13。H1 最细层仍约 2.93–2.95，不把 L2 平台误读为整体降阶。 |
| 历史受迫方腔 | 128² 单位无滑移方域、nu=0.01、gamma=1、零初值；保存外力 **f=(0,sin x)**；两种二阶格式 | [Notebook](forced_ns_convergence/studies/forced_time_convergence/results.ipynb) · [中文正文](forced_ns_convergence/studies/forced_time_convergence/description.tex) · [使用说明](forced_ns_convergence/studies/forced_time_convergence/README.md) | 16 个被测＋2 个参考全部校验通过。T=2 的 L2 观察阶接近 2，H1 逐渐趋近 2；六项保存的参考检查通过。不是当前 m=2 生产外力 f=(0,sin 4*pi*x) 的结果。 |
| 三角涡量初值 | 256²、(0,2*pi)² 无滑移、nu=0.001、gamma=5、T=4；f=(cos x,0)、历史初值身份 weighted_trig_vorticity | [Notebook](forced_ns_convergence/studies/trig_time_convergence/results.ipynb) · [中文正文](forced_ns_convergence/studies/trig_time_convergence/description.tex) · [使用说明](forced_ns_convergence/studies/trig_time_convergence/README.md) | 18 个被测＋2 个参考均校验，其中 5 个被测运行失败。保留全区间及 k>=6 放大图；细区间约二阶，不把粗步有限解等同于准确解。 |

### 固定证据编号

| 研究 | 批次编号 | 采用的报告编号 | 数值表 |
| --- | --- | --- | --- |
| 制造解 | `20260926T094814-5097bbbb` | `20260926T100547-ac5f1fac` | `reports/manufactured_ns_convergence/<报告编号>/tables/errors.csv`，40 行 |
| 历史方腔 | `20260922T023439-35a9711f` | `20260922T032418-3cd92bd4` | `reports/forced_ns_convergence/<报告编号>/tables/errors.csv`，48 行 |
| 三角涡量 | `20260922T050733-f2dbf5c7` | `20260922T054144-5aee3c91` | `reports/trig_ns_convergence/<报告编号>/tables/errors.csv`，54 行 |

三角涡量使用通过检查的 k=11/12 参考对；同一批次较早的报告 `20260922T052514-1dacc8ed` 未通过参考检查，不将其表与后一个报告的状态拼接。128²、仅一个被测步长层的 trig pilot 也不用于估计收敛阶。

### 统一展示约定

各研究独立出图，不跨方程、外力、初值或网格叠加成一条阶数曲线。每个参考解 Notebook 保存全步长图、固定 k>=6 放大图、完整 L2/H1 半范数误差及保存观察阶、参考敏感性信息。图表由 CSV 重绘，不嵌入旧 PNG 代替执行。

失败行及 NaN 保留；失败位置在坐标底部画叉，叉号不是误差值，折线不跨失败层连接。三角涡量普通格式 k=1...5 的失败对应 15 行观测；k=0 虽完成，T=4 的 L2 误差约 1.68e21，也必须显示。SAV 跨越粗细区的异常大观察阶不能当成方法阶数。

原始误差表不改写。Notebook 校验捕获的 `input_batch.json`，而非后来可能变动的批次文件；显式固定报告及表格 SHA256。历史输入只按实验名和运行编号迁移定位，绝不以当前源码身份复用或用其他运行补位。执行方式见各研究说明。

## 二、待核验：256² 无滑移共同参考结果

以下两项都用格点流函数初值，但外力不同，**不共享参考轨迹**。目前只保留报告和误差 CSV 的证据身份；不制作空 Notebook，也不把 CSV 的趋势写成已核验结论。

| 研究身份 | 显式报告 | 保存内容 | 本轮确认的缺口 |
| --- | --- | --- | --- |
| 旧外力标识 `sin_x`，256²、nu=0.001、gamma=1000 | `20261004T054907-1c8a66ff` | 304 行观测，四格式及粗步加密；参考 k=10/11 | 报告引用的 78 份原始记录三件套本地不齐；保存的 5% 参考检查在 t=2、4 未全部通过。T=4 的 H1 比值约 17.53%。 |
| 新外力标识 `clamped_streamfunction`（H 值外力），相同网格与阻尼 | `20261005T142413-cec57910-k11-reference` | 360 行观测；使用 k=11 为误差参考，与 k=10 比较 | 引用的 92 份 config/manifest/HDF5 本地均缺失；没有更细参考，状态为 `finer_reference_unverified`。保存的 T=4 H1 比值约 11.98%，且各项 passed 为空，不得当成通过。 |

上述报告均在 `reports/no_slip_reference_convergence/<报告编号>/`。报告文件的 `status=complete` 只表示分析流程写完，不推翻 `reference_passed=false`。新外力较早的 k=10 provisional 报告 `20261005T102403-30343770-k10-provisional` 也不是参考已验证材料。

解锁条件是补齐这些显式成员和参考记录、验证原来源/配置/摘要及观测节点，并解决参考充分性：需要更细参考或独立可审查的误差证据。本轮不启动该计算。若原始数据随后提供，先核对身份，不把当前配置生成的轨迹冒充历史结果。

## 三、算法验证附录候选

以下与上述物理算例分列；本轮盘点到保存分析文件，但未通过本轮新展示入口重新审核其全部子记录或 NPZ。

| 验证研究 | 保存报告编号 | 应回答的问题 |
| --- | --- | --- |
| [周期 Fourier 验证](periodic_ns/README.md) | `20261004T074942-8a577b4b`，位于 `reports/periodic_ns_validation/` | 周期模型制造解时间误差及空间细化；不是无滑移结果。 |
| [Channel 验证](channel_ns/README.md#short-verification-study) | `20261004T092910-39d279d1`、`20261004T094521-94681a55`，位于 `reports/channel_ns_validation/` | 周期流向/无滑移壁面下的时间误差与连续 Stokes 空间细化。两份报告保留独立来源，不按日期自动替换。 |
| [一阶格式验证](imex_euler/README.md) | `20261004T135624-eb4fabff`、`20261004T140449-1fa7c628`，位于 `reports/imex_euler_validation/` | IMEX Euler 与对应 mr-ccSAV 在六后端的一阶精度和约束；不混入 SDIRK 二/三阶表。 |

32² 短时 smoke/pilot、单位方域夹持梁初值的旧 128² 无滑移批次及非单调诊断仍保留在原位置，不升级为主文的独立已验证收敛证据。尤其旧 smoke 的敏感性阈值可为 50%，不能套用主研究的 5% 说明。

## 建议的论文组织

先用制造解说明半离散时间精度及最细误差限制；随后呈现历史方腔的参考解时间比较；再以三角涡量展示粗步失效、有界不等于准确及细区间二阶。256² 共同参考的两项作为待核验材料，暂不用于支持已验证结论。周期、Channel 与一阶研究留作独立算法验证附录，待各自原始子记录审核后再完成展示。
