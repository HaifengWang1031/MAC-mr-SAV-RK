# 制造解时间收敛成果

- [结果 Notebook](results.ipynb)：从保存的数值表重绘，含已执行图表输出。
- [中文论文实验描述](description.tex)：实验设置与比较方法，不是新的结果结论。
- [计算说明](../../README.md)。

## 执行与输入

在项目根目录安装可选工具：`uv sync --locked --group notebook`。
可用编辑器的 Jupyter 支持选择项目 `.venv` 内核；命令行重新执行并保存输出：

```sh
uv run --locked --group notebook jupyter execute --inplace experiments/manufactured_convergence/studies/time_convergence/results.ipynb
```

默认明确选定报告 `reports/manufactured_ns_convergence/20260926T100547-ac5f1fac/`，
以及它捕获的批次 `20260926T094814-5097bbbb` 和 40 个成员。
先在标记为 `parameters` 的单元核实项目根目录、报告相对路径和两个固定摘要。
更换数据时必须显式选择兼容报告并独立核实摘要，不自动查找最新运行。

Notebook 校验原报告、原报告捕获的批次副本、误差表及每个成员的配置/结果校验和、
源码版本、物理设置、格式、步长和完成状态；原始绝对成员路径仅按明确的运行编号
重新定位到当前项目的数据目录。分析期间捕获的批次状态可能是 running，
因为 runner 在分析完成后才更新原批次；不要求该副本等于后来修改的 batch.json。

CSV 中 L2/H1 误差及观察阶来自原分析器，并非 Notebook 新算的场误差。
表的 SHA256 固定在参数单元；校验这些输入不等于对历史误差计算作独立数学验证。
本地数据不随 git 提交；没有数据时仍可阅读保存输出，重新执行将明确报错，
不会自动补算、下载或改用另一个批次。执行只产生 Notebook 输出，不改写 runs/reports。

## 论文引入

使用支持中文的主文档（例如 `ctexart` + XeLaTeX），加载 `amsmath,amssymb`，
再通过 `\input{experiments/manufactured_convergence/studies/time_convergence/description.tex}` 引入。
片段没有独立前导区，也不固定论文模板；如果从其他目录编译，应相应调整引入路径。
