## 2026-10-23 推进格式更正

用户要求改为 SDIRK3-mr-ccSAV 后，MAC MINI 的普通SDIRK3批次 v1 已停止，数值文件原样保留，元数据标记 interrupted。新批次 `20261023-mrsav-scan-v2` 从零独立启动，使用 `configs/long_scan_mrsav.json`，scheme=sdirk3_mrsav、gamma=1（实现默认值）、r0=0。24个物理/网格/步长成员与v1相同；旧 long_scan.json 仍明确表示普通格式，不改写旧参数身份。

新入口按配置选择格式并逐步传递r。HDF5 schema_version=2 保存每步四阶段的r、全部候选根/根残差、标量/动量/全散度残差；空候选槽为NaN。诊断保存r及采样区间最大标量/根残差；快照保存r。最终速度和r均持久化。该更改只在实验入口，不改动SDIRK3-mr-ccSAV阶段公式。

# Molenaar 受迫无滑移方腔候选复现

目的：检查给定候选源是否产生稳态、周期、双/三频准周期、锁频和混沌；不以参数名或宽带谱单独宣称复现。普通 SDIRK3、MAC、零初速度、静止无滑移边长2方腔；x,y∈(0,2)，X=x−1,Y=y−1。

源为 q=0.1[cos(6πX)+cos(6πY)]+Qs。体力由 model.py 独立构造，其旋度与该源相符。实际角波数6π有文献图像支持；源幅值/Fourier归一化以及Qs=0.01或0.02仍未从作者代码确认。两个正旋转候选均保留；未加底摩擦、随机扰动或强制模态对称投影。

`configs/long_scan.json`：nu_prime=A0/nu∈{20,30,40,57.5,70,77.5,80,85}，各取Qs∈{0.01,0.02}，A0=0.05、128²、dt=0.002、T=1000；然后在nu_prime=40/80的两种Qs下分别进行dt=0.001时间对照和192²空间对照。共24成员，均从静止独立启动。样本每0.02保存，完整速度每5保存；日志每30秒刷新。两个工作进程，每个进程BLAS/OMP等限制为1线程。

源来自博士论文字面式(2.11)结合图像推断；不得把这些候选标记为文献精确归一化已验证。128²只有约21格/同号周期，边界层和近临界分岔需通过加密后判断。时间步也不同于原文谱方法，后列对照只检查两个代表参数；它们不能验证所有准周期临界值。短时检查不证明长期稳定或混沌复现。

运行：`uv run python -m experiments.molenaar_cavity.run --config experiments/molenaar_cavity/configs/long_scan.json --output /absolute/new/campaign --workers 2`。

每个新目录有config.json、manifest.json、run.log和流式results.h5。数据集diagnostics带列名，包含(原域1/2,0)涡量探针、动能、enstrophy=离散H1半范数平方/2、角动量、功率、耗散、每个采样区间最大阶段残差/全散度与采样时CFL。enstrophy使用无滑移无散速度的离散梯度能量代理，与只对内部顶点涡量求积不同。快照包括实际物理时间，未保存每步全场。double精度存储。保留失败前缀，成功后才写complete并生成结果校验和。无断点续算入口；快照不是已实现的续算承诺。重新计算使用新目录。每个manifest保留实际配置、代码/锁文件哈希及环境版本。

MAC MINI 批次使用单独source快照，避免覆盖远端工作区。source.tar.gz为最初传输包；正式启动前如有修正会生成source-final.tar.gz及SHA256，后者才是实际启动代码。进程在tmux内运行并用caffeinate防止运行期间睡眠。批次完成与失败以campaign.json和各成员manifest.json为准。日志正常不代表现象已复现。

分析独立消费显式run ID：`uv run python -m experiments.molenaar_cavity.analyze /absolute/case-00 --discard 500 --output /absolute/new/report`。只读取校验过的完整记录，不创建模型、不推进。输出谱与峰值仅为描述性证据，仍需比较去瞬态窗口、延迟相图、Poincare映射与空间/时间敏感性后识别动力学。波数和弱旋转审计见项目reports/literature/molenaar_forcing_20261023/。
