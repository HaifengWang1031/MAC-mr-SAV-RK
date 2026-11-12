# 核心仓库保留与本地研究备份方案

**状态：用户已于 2026-11-12 确认清单，并指定 ~/GameStation 外部盘。** 备份与独立
恢复门槛已通过；已按此清单完成 index-only 本地化及公共配置/文档修订，最终验证见
[validation](validation.md)。不推送或合并，不操作远端生产作业。

**后续独立授权（2026-11-12）：** 用户要求先修复主分支审查三项问题。其中纯求解器的
bounded-history/联合PI测试从本地tests/tasks/test_torque_production.py移入公共core，
保持函数体/阈值并只保留一份；该本地文件编辑前另经外盘tar备份和独立恢复。故下文
412文件全字节未变、171跟踪文件及全部Python未改均指83a043d本地化快照，不是永久
不许修改声明；后续修复另记验证与备份，原完整备份保持封存。

## 1. 基线与推荐结论

- 优化 worktree：`/Users/wanghaifeng/Desktop/MAC-mr-SAV-RK-records-core`，分支
  `refactor/run-records-core`，基线 `b0c684e9a82cb5b363443ee865a483662c08b2fb`。
- 原 worktree：`/Users/wanghaifeng/Desktop/MAC-mr-SAV-RK`，仍为 `744a695`，本轮不切换其分支。
- 保持**同一源码根、原来的本地路径**：研究文件保留在磁盘，仅取消 Git 跟踪。
  用户已批准此范围；不拆成第二个研究仓库、不移动目录、不引入跨仓库导入或统一 runner。
- 基线 582 个跟踪文件：建议远端 HEAD 保留 **170** 个，本地专用 **412** 个。
  Git blob 内容合计分别为 1,327,807 / 2,826,893 字节，不含缓存、Git 历史或运行数据。
  本计划等新增公共文档另计；这些不是未来最终提交的固定文件总数。
- 这里只精简后续提交的文件树，**不重写 Git 历史**。旧提交仍包含研究源码，仓库包体积
  不保证立刻缩小。原先已忽略的约 12 GB runs / 110 MB reports 不是此次移出 Git 的内容。

完整逐文件清单在本机 `.scratch/core-localization/inventory.json`（已被现有规则忽略）。
它记录基线、每个路径的 keep/local_only 建议、Git blob、模式、字节数和建议 mypy 目标，
标记为 `proposal_not_approved_not_a_backup`。本次清单 SHA256：
`5ca6ed66bbaa3790a21e779436abb9a06ecd17718f2714781e6bf53673a868d4`。
原始盘点状态作为历史证据不改；获批副本和状态 `approved_by_user_2026-11-12` 在外盘
`metadata/approved-localization-inventory.json`。清单不是备份，正式完成标记另见第 5 节。
实施前发现基线漂移、新增跟踪文件或本地修改，必须重新对照并纳入备份，不能盲用旧清单。

## 2. 获批远端保留

| 范围 | 基线文件数 | 原因 |
| --- | ---: | --- |
| `solver/` 全部 | 63 | 五类空间模型、六格式、自适应、可选求解器及其数学 interface；不按当前研究是否使用裁剪 |
| `run_support/` 全部 | 4 | 公共记录、校验、来源及历史兼容 |
| `postprocess/` 全部 | 6 | 公共只读 reader、图、报告、独立 CLI |
| `tests/core/`、`tests/references/`，加 `tests/__init__.py`、`tests/README.md`、`tests/core_distribution.py` | 55 | 保留全部核心机制、独立数学参照和隔离分发 seam；不删可选后端测试 |
| `tools/__init__.py`、`tools/plot_speed_snapshots.py`、`tools/README.md` | 3 | 旧只读 CLI adapter 与工具归属说明；不保留任务计算实现 |
| 下述公共/数学/资格文档 | 29 | 当前契约与可追溯证据，而非运行入口 |
| `archive/README.md` | 1 | 只保留历史来源与独立装配说明，不复制原桌面程序 |
| 根目录九个文件 | 9 | `.gitignore`、`.python-version`、`AGENTS.md`、`CONTEXT.md`、`GLOSSARY.md`、`README.md`、`pyproject.toml`、`uv.lock`、`skills-lock.json` |

公共文档采用保守保留：除下一节列出的五份纯研究资料外，保留现有 docs 文件。
其中包括 ADR/agent 约定、spec/architecture、MAC/Channel/Fourier/Legendre/FEM/Hele–Shaw
数学说明、优化实现与已否决后端的资格限制、validation 及其历史 JSON 证据。
保留逆级联审计撤回记录，避免本地化被误读为恢复已撤回的科学资格；它不是远端计算入口。
历史模型迁移、评审、测量和旧求根描述仍按日期解释，由当前 ADR/契约覆盖，不提升为当前事实。

不因为文件名带研究名或出现实验路径就删除数学说明；也不把实验 builder 搬进公共 module
来维持旧首页命令。公共 solver/records/postprocess 的既有 interface 不因此次 Git 操作改变。

## 3. 获批仅本地保留

| 范围 | 基线文件数 | 本地化方式 |
| --- | ---: | --- |
| `experiments/` 全部 | 341 | 配置、初值/制造解、runner、专属 reader/分析、campaign、notebook、兼容 adapter 均保持原路径 |
| `tests/tasks/` 全部 | 60 | 全部任务回归保留；存在时默认 pytest 仍收集，不设置静默排除 |
| 五个会计算的 tools 脚本 | 5 | 保持原路径与 `--source-root`/冻结 worker 兼容 |
| 五份纯研究资料 | 5 | 保持 docs 下原路径，本机继续使用 |
| `.scratch/implementation.md` | 1 | 历史实施便笺回归本地；以后 `.scratch/` 不再留跟踪例外 |

五个计算脚本：

```text
tools/benchmark_steps.py
tools/benchmark_fem_adaptive.py
tools/probe_hele_shaw_inlet.py
tools/rerun_low_high_stability_probe.py
tools/search_low_high_stability_steps.py
```

五份研究资料：

```text
docs/experiment-inventory.md
docs/fluid-model-extension-survey.md
docs/forced-2d-mhd-literature.md
docs/forced-2d-mhd.bib
docs/mhd-long-time-discussion.md
```

`runs/`、`reports/` 继续保持本地且分开；被忽略的新研究输入、脚本及 `.scratch/` 内容也要备份，
不能仅以这 412 个当前跟踪路径作为备份范围。新研究工具/资料也必须采用明确的本地约定，
不能以此次忽略规则自动识别未来所有研究文件。

## 4. 获批配套修改与持续约束

1. **mypy 目标**：当前配置仍指定 experiments 和整个 tools。建议默认目标改为
   `solver, run_support, postprocess, tools/__init__.py, tools/plot_speed_snapshots.py,
   tests/references, tests/core_distribution.py`。
   不用整个 tools 目录，否则同机存在的本地计算工具会再次扩大检查范围。
   完整本地检查显式指定原目标（含 experiments/tools），不能把它的通过声明改称核心通过。
   不调整依赖/可选后端，不安装新环境，`uv.lock` 保持原样。
2. **导航和公共用法**：README 首个示例改为直接构造公共 solver 或运行只读 CLI；规范和
   核心说明链接到公共 owner。优先修订 README、CONTEXT、architecture/spec、tests/README、
   solver/adaptivity/README、run_support/postprocess/tools 说明。保留的历史文档中，本地研究
   和报告路径标注为可选本地材料，不把缺失文件做成公共必需步骤；不删资格限制或篡改历史。
3. **忽略规则**：以后只新增明确列出的本地范围。experiments、tests/tasks 可以整目录忽略；
   tools 只忽略上述五个脚本，不忽略整个目录；docs 只忽略上述五份研究资料；撤销 scratch
   历史单文件的跟踪例外。新规则必须用 `git check-ignore --no-index` 验证本地路径被忽略、
   公共 wrapper/核心测试/数学文档不被忽略。现已跟踪路径还须单独取消跟踪。
4. **来源不按 Git 状态过滤**：本地研究仍从原路径导入。legacy 扫描继续记录磁盘上的真实
   研究依赖，显式 task_paths 仍包括被忽略的任务文件；不为了“远端干净”漏记它们。
   单纯取消跟踪不应改变来源；但修改 pyproject.toml 的 mypy 配置会真实改变其文件哈希，
   所以新运行身份可能改变。旧 manifest、冻结源码、链与 retarget 校验一律不改、不放宽。
5. **操作限制**：只能针对获批的精确清单执行 index-only 取消跟踪，先预览，再核对文件仍
   存在且字节不变。不得执行普通删除、`git clean -fdx`、强制 rm、reset/clean 或历史重写。
   忽略不等于保护：切换分支、reset 和 clean 可能覆盖/删除本地文件。保留专用 worktree，
   不把唯一副本留给 Git 的“未跟踪/已忽略”状态。

这些配套随本地化实施；数值公式、依赖锁、任务默认值及保存记录保持不变。

## 5. 备份范围、位置与一致性

实际外盘为 USB ExFAT `/Volumes/GameStation`（~/GameStation 指向它，独立于本机设备）。
备份目录：`~/GameStation/MAC-mr-SAV-RK-backups/20261112T044006Z-pre-localize/`，标记
`COMPLETE-local-backup.json`。以两份 tar 保存完整文件树，避免 ExFAT 丢失 POSIX 模式。
两工作区共 10,508 个文件 / 12,888,020,622 字节；all-refs Git bundle 独立克隆并 fsck。
文件内容/模式/精确 mtime 恢复校验完成，恢复树留在本机 APFS
`/private/tmp/mac-localization-restore-20261112T044006Z/`；临时恢复树可能被系统清理，
不能替代外盘正式归档。SHA256SUMS、排除清单、原始属性、恢复步骤与 6 项实际 reader
证据在外盘 metadata/RESTORE.md。以后备份仍须重新测算空间和一致性，不沿用此次观察。

备份应包括：

- **Git 独立历史**：创建 `repository.bundle`（所有 refs），另记各分支/远端引用、两 worktree
  的 HEAD、状态和差异。原 worktree 的 `.git` 是真实 Git 目录，优化 worktree 的 `.git`
  是指向它的文件，单独复制这个指针不是备份。bundle 不自动包含 reflog-only/dangling
  对象；存在需要保留的游离工作须另行显式保护。bundle 不上传远端。
- **两 worktree 的实际文件**：分别保存两份源码/配置/文档/本地研究内容，不能用原分支
  文件覆盖优化后的 adapter。包含全部未跟踪/被忽略研究输入和 scratch，不仅 Git 文件。
- **运行与报告**：原 worktree 已有约 12 GB runs、110 MB reports；优化 worktree 中若有
  任何运行/报告也独立保存。不重生成数据，不修改 manifest，不整理或合并运行目录。
  特别保留运行下的冻结 source、checkpoint、chain、batch、配置和报告自带脚本。
- **环境证据**：Python、平台、锁文件、已安装依赖版本、显式数值线程设置及启动说明。
  不复制虚拟环境/编译缓存来冒充可迁移环境，不收集无关密钥或完整敏感环境变量。
  可选 PETSc/PARDISO/Gmsh 和真实 TeX 的重建能力单独记录，不由锁文件自动授予资格。
- **外部依赖附表**：记录符号链接目标和项目外输入/源码引用；不静默跟随链接去归档整个
  桌面、不修改原桌面材料。未覆盖的必要外部输入必须补备份或声明缺口，不能声称可完整恢复。

排除项限于 `.git` 本体（由 bundle 替代）、虚拟环境及明确可再生缓存，并写入排除清单。
备份文件清单记录命名空间、路径、类型、字节数、模式、SHA256；链接记录链接本身及目标，
不伪装为普通文件。保留两 worktree 的归属，避免同名文件覆盖。

本次没有源符号链接或特殊文件。macOS 的 provenance/TCC 属性会在新位置重发，原值完整
留在清单，不恢复为安全授权；本批扩展属性仅有这两类。数据字节、模式/mtime 均严格
核对，不宣称完整 ACL/系统身份克隆。恢复 reader 只从恢复的优化源码导入，实际通过旧
MAC、周期 MHD、逆级联谱、两条 schema-2 checkpoint 链和力矩 MAC NPZ 的只读检查；
没有本地真实 FEM 保存记录可验。11 个旧 running 标记原样保留，未发现本地活动 writer。
历史远端 seed 的四个现存本地副本已覆盖；服务器生产只有本地计划副本，远端结果和环境
不在本备份内，不据此授权生产续算或声称新机器完全复现。

**有活动写入就不能直接宣称完整一致备份。** 不停止、挂起或改动生产作业；先识别 writer，
使用已完成不可变记录/已验证应用 checkpoint 或经确认的一致快照。普通跨文件复制、
仅比较前后哈希，均不能自动证明活动 chain/HDF5 的应用一致性。没有可靠一致点时等待或
报告阻塞，不能忽略 tar/copy 警告继续取消跟踪。临时备份目录只在下节通过后标记 complete。

## 6. 恢复演练与放行顺序

1. 在独立位置完成备份，校验每个源文件/备份文件的字节摘要及必要元数据；确认源记录
   在备份期满足一致性要求。bundle 校验后实际克隆到新目录，不只检查 bundle 文件存在。
2. 在**另一个空目录**重建文件树；分别恢复原始/优化源码和本地数据，不覆盖两个活跃
   worktree。验证全部备份摘要；bundle 不能依赖原 `.git` 才可读取。
3. 先复用已知环境作只读恢复验证：从恢复目录导入 reader，显式读取恢复的数据，检查
   模块来源，防止退回原项目“看似成功”。按实际记录家族选择其专属 reader；至少覆盖
   已存在的关键 MAC/FEM/MHD/专属 checkpoint 或谱记录，并验证链接/链关系。
   不以一个通用 MAC reader 证明所有家族恢复，不启动计算。绝对外部路径不能擅自回填，
   有未解决引用或加载失败即不放行。没有新环境安装证据时不得宣称可在新机器完全复现。
4. 在无本地研究的候选源码副本运行默认 pytest、mypy、核心隔离入口和公共 CLI；另外
   在保留本地研究的工作区运行完整 pytest/mypy。两种 profile 的通过数与 skips 分开记录。
5. 上述条件全部满足且用户批准后，才在优化分支实施第 4 节修改和 index-only 取消跟踪；
   比对精确索引差异、确认本地 412 个原文件仍在、内容未变且规则正确。取消跟踪后本机
   默认 pytest 应继续发现任务测试，不能通过改 testpaths/exclude 隐藏它们。
6. 完整本地回归、纯核心副本验收及文档检查再次通过，审查后只作本地提交。原分支不变，
   推送/合并仍另行批准。原始备份与验收日志保留到后续确认，不自动删除。

回滚优先在新目录从 bundle/文件备份恢复，而不是对活跃 worktree reset --hard。
若 index-only 操作尚未提交，只按精确路径恢复 index（不覆盖 working tree）；若已提交，
先比较、另建恢复 worktree，不能让恢复跟踪的旧文件静默覆盖本地研究新修改。

## 7. 实施与方案演练记录

实施时逐项匹配基线 Git blob、备份完成状态、清单审批和忽略规则，再仅取消 412 个
路径的 index 条目。立即复核其磁盘文件及 SHA256 全部不变；所有基线 Python/uv.lock
字节也不变，原工作区仍干净位于 744a695。基线保留 170 个文件，加本说明共 171 个。
公共配置默认 mypy 78 个文件已通过；两种形态的最终回归与审查结果在 validation。
逐路径实施证据留在本地 scratch；外盘保留批准清单及原始完整备份，不依赖 .git 指针。

以下是审批前的历史方案演练（复用原 Python 环境，不安装依赖），不是实施后的状态：

- 完整清单覆盖全部 582 个基线路径，keep/local_only 无重叠；导出的 170 个保留文件逐个
  核对尺寸及 Git blob，未混入本地修改或研究目录。保留 Python 的 AST 无研究 import。
- 只有保留文件的副本，禁止实验/任务导入，**默认 pytest 392 passed / 17 skipped**
  （25.09 秒）。没有修改 testpaths 来制造核心通过；该套件包含实际隔离快照模块 CLI。
- 副本原样运行默认 mypy，按预期因 experiments 缺失报错，证明尚有配套工作；用建议
  公共目标显式覆盖后 **78 文件通过**。实际完整工作区 mypy 仍 **246 文件通过**。
- 公共旧工具 wrapper 在副本中以 `-I` 执行 `--help` 成功；这项只证明入口启动，真实
  报告生成来自核心测试里的模块 CLI，不夸大旧 wrapper 的独立完整报告演练范围。
- 拟议忽略规则只在临时 Git sandbox 验证：**恰好 412 个本地路径被忽略，170 个公共路径
  全部不被忽略**。真实 .gitignore、pyproject、index 和实验文件均未改变。
- 明细在本机 `.scratch/core-localization/dry-run.json`；草拟规则在同目录
  `proposed.gitignore`，不是实际启用的规则。临时源码/sandbox 已自动清理。

第三阶段完整基线仍为 728 passed / 36 skipped；本轮未重跑任务套件，没有改生产 Python。
以上不是正式备份、数据恢复/活动写入一致性、最小环境安装或最终发布验收的证据。
候选副本的公共导航仍是基线版本，第 4 节修改尚未实施。

用户确认和外盘备份/恢复已完成；配套修改、精确 index-only 操作及两种 profile 的最终
回归/审查见 validation。此清单不授权远端公开访问、历史清理、新环境安装、服务器迁移、
生产重启或推送/合并。今后新增本地研究文件仍须另行备份，Git 不再替这些内容保存新版本。
