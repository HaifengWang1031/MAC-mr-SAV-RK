# 运行记录与来源

`records.py` 负责单次计算的开始/完成/失败、配置与来源身份、公共 HDF5、校验和及复用。
`source.py` 负责明确源码范围、生成来源信息；两者都不导入实验计算入口。
模型与阶段推进仍属于 `solver/`，实际网格和表示的专属校验仍由对应 reader 负责。

## 新调用方：核心＋显式任务来源

```python
from run_support.records import begin_run, save_result, finish_run, load_record
from run_support.source import provenance

source = provenance(task_paths=[
    'local/my_study/run.py',
    'local/my_study/problem.py',
])
```

默认核心范围为 `solver/`、`run_support/` 下全部 Python 文件以及根目录的
`pyproject.toml`、`uv.lock`、`.python-version`。`task_paths` 相对源码根目录解析；
目录收集其 Python 文件，单个文件也可指定参数之外的必要任务数据。
缺文件、空目录、越出根目录的路径均报错。显式来源不接受源码符号链接（即使目标在根内，
包括嵌套目录或文件路径中的链接），避免遍历漏项及快照丢失导入名称；源码根目录本身允许规范化。
路径排序并去重，不受源码根目录绝对位置影响。`source_files(root=..., task_paths=...)` 返回同一清单供源码捕获使用。

结果包含 `source_schema=2`、相对文件名到 SHA256 的 `source_files`、`code_sha256`，
以及 Python/平台、依赖版本和线程环境。必需依赖中的 scikit-fem/pymetis 也记录版本；
可选 PETSc/PARDISO 等依赖仍由选择这些后端的计算入口补充。相同代码哈希不代表完整
来源相同，更不代表科学验证通过。`run_identity` 使用有效配置和**完整来源**。

调用方负责列全实际任务依赖，包括相关包初始化；不自动分析导入闭包，也不自动发现
其他本地研究目录。当前仅支持同一源码根目录中的任务，不是跨仓库发布/打包系统。
`periodic_mhd/run.py` 是首个采用新方式的真实入口：计算、初值和 reader 在清单中，
只读分析与无关研究不影响其计算身份。

## 过渡期兼容

`experiments.records` 继续导出原记录函数，记录实现只有一份。其 `provenance` 有意保留
旧的广泛扫描策略：按旧顺序扫描 solver/experiments/tools，再加入 run_support/postprocess。
`legacy_source_files` 给旧式快照和校验提供同一清单，依次保留原 solver/experiments/tools，
再附加 run_support/postprocess 的真实实现；仍支持缺少这些新目录的真实旧源码树。
公共后处理的新报告显式选择 postprocess，不把它加入默认计算核心来源；见
[postprocess](../postprocess/README.md)。旧 manifest 与冻结源码不回写。
因此**尚未迁移的入口仍受本地无关 Python 文件影响**。研究文件现仅本地保留在原路径，
Git 忽略不会从来源扫描中移除它们；不把本地化伪装成所有任务已迁移显式来源。
pyproject 配置变化会真实改变新来源哈希，不回写旧运行或放宽 chain/retarget。

新来源策略不会重算或改写旧 manifest；旧结果按其记录的来源与校验和读取。新共享实现
改变了真实源码，新运行的身份应如实变化，不应与旧运行伪装同源。HDF5 schema、历史根槽、
冻结因子标量校正和专属压力/网格语义均未改变。已有冻结源码和生产任务不在此分支重写。

## 验证

```sh
uv run --locked pytest -q tests/core/test_run_support.py tests/core/test_record_compatibility.py
uv run --locked mypy
# 只有完整本地研究文件存在时：
uv run --locked pytest -q tests/tasks/test_record_compatibility.py
```

核心测试在只有 solver/run_support/环境文件的临时副本中，独立进程阻止 experiments/tools
导入，实际保存并读取记录。兼容测试另验旧来源编码、旧导入、冻结 worker 和 MHD 复用。
该验证不代表所有实验已迁移，也不替代可选后端或长期生产恢复验证。
