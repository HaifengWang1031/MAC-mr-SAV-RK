# 实验与测试记录盘点（2026-09-24）

本次仅盘点现存文件；manifest 中的 complete 表示当次程序完成，不等于当前版本已复核其物理结论。running 表示记录未收尾，不能据此判断进程仍在运行。历史输入不移动、不覆盖，以维持报告引用。

## 现存计算记录

| 实验 | 记录数 | 状态 | 定位 |
|---|---:|---|---|
| cavity | 7 | complete: 5, failed: 1, running: 1 | 现存实验结果 |
| decay | 20 | complete: 19, running: 1 | 已停用实验，历史证据 |
| forced_ns | 18 | complete: 18 | 现存实验结果 |
| mac_parallel | 2 | complete: 2 | 已停用实验，历史证据 |
| ns_mms | 54 | complete: 54 | 已停用实验，历史证据 |
| stokes_mms | 12 | complete: 12 | 已停用实验，历史证据 |
| trig_ns | 25 | complete: 20, failed: 5 | 现存实验结果 |

旋转实验代码和配置存在，但当前没有 `runs/rotation_ns/`。`docs/validation.md` 中相关历史数字不能由当前仓库内原始记录复核。

## 代表记录导航

按实验、格式和完成/失败状态各列一个代表，优先选较大网格、较长终止时间；这只是导航规则，不意味着较大运行更可信。完整清单见下表。

- cavity / sdirk2_mrsav / complete：[记录](../runs/cavity/20260922T073245-2cba41a7-cb35ec3c/manifest.json)
- cavity / sdirk2_mrsav / failed：[记录](../runs/cavity/20260920T114145-f859dd51-2bcca509/manifest.json)
- cavity / sdirk2_mrsav / running：[记录](../runs/cavity/20260920T114329-7cb856df-6b6a2558/manifest.json)
- decay / sdirk2 / complete：[记录](../runs/decay/20260920T033855-2ad313ac-d7a4c4a5/manifest.json)
- decay / sdirk2 / running：[记录](../runs/decay/20260921T033037-e5cdbd9c-8e5b788b/manifest.json)
- decay / sdirk2_mrsav / complete：[记录](../runs/decay/20260920T033855-00531f44-d83c2cf2/manifest.json)
- forced_ns / sdirk2 / complete：[记录](../runs/forced_ns/20260922T023439-85ea93c5-d47e7349/manifest.json)
- forced_ns / sdirk2_mrsav / complete：[记录](../runs/forced_ns/20260922T023440-cc139ded-4d9cfbc7/manifest.json)
- mac_parallel / sdirk2 / complete：[记录](../runs/mac_parallel/20260921T053326-294793bf-60383047/manifest.json)
- mac_parallel / sdirk2_mrsav / complete：[记录](../runs/mac_parallel/20260921T053257-55808f91-e1b30707/manifest.json)
- ns_mms / sdirk2 / complete：[记录](../runs/ns_mms/20260920T033847-d2c255aa-d117cbe9/manifest.json)
- ns_mms / sdirk2_mrsav / complete：[记录](../runs/ns_mms/20260920T033850-6b3252ae-2c05d600/manifest.json)
- stokes_mms / sdirk2 / complete：[记录](../runs/stokes_mms/20260920T033846-e60ce7a6-3aafe192/manifest.json)
- trig_ns / sdirk2 / complete：[记录](../runs/trig_ns/20260922T050733-3f4ed806-1c7ebcb8/manifest.json)
- trig_ns / sdirk2 / failed：[记录](../runs/trig_ns/20260922T050739-0aca4f24-41c55fb1/manifest.json)
- trig_ns / sdirk2_mrsav / complete：[记录](../runs/trig_ns/20260922T050736-3c2b08cc-a32c7544/manifest.json)

## 全部运行索引

| 运行目录 | 状态 | 格式 | 网格 | T | dt |
|---|---|---|---|---|---|
| [20260920T081647-d970189a-d1346b11](../runs/cavity/20260920T081647-d970189a-d1346b11/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 40.0 | 0.02 |
| [20260920T081649-0fb5257b-3f12761c](../runs/cavity/20260920T081649-0fb5257b-3f12761c/manifest.json) | complete | sdirk2_mrsav | 64 × 64 | 40.0 | 0.02 |
| [20260920T114145-f859dd51-2bcca509](../runs/cavity/20260920T114145-f859dd51-2bcca509/manifest.json) | failed | sdirk2_mrsav | 128 × 128 | 300.0 | 0.001 |
| [20260920T114329-7cb856df-6b6a2558](../runs/cavity/20260920T114329-7cb856df-6b6a2558/manifest.json) | running | sdirk2_mrsav | 128 × 128 | 300.0 | 0.001 |
| [20260920T122407-5c32332f-0f957068](../runs/cavity/20260920T122407-5c32332f-0f957068/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 1.0 | 0.001 |
| [20260920T122619-f7209c02-c66af965](../runs/cavity/20260920T122619-f7209c02-c66af965/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 10.0 | 0.001 |
| [20260922T073245-2cba41a7-cb35ec3c](../runs/cavity/20260922T073245-2cba41a7-cb35ec3c/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 50.0 | 0.001 |
| [20260920T033855-00531f44-d83c2cf2](../runs/decay/20260920T033855-00531f44-d83c2cf2/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.2 |
| [20260920T033855-2ad313ac-d7a4c4a5](../runs/decay/20260920T033855-2ad313ac-d7a4c4a5/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.05 |
| [20260920T033855-57612fc7-7bcd16ec](../runs/decay/20260920T033855-57612fc7-7bcd16ec/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.05 |
| [20260920T033855-7da5ba50-5d1f7256](../runs/decay/20260920T033855-7da5ba50-5d1f7256/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.2 |
| [20260920T033855-a4aea1bc-0390c7df](../runs/decay/20260920T033855-a4aea1bc-0390c7df/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.01 |
| [20260920T033855-afe093ad-fc776783](../runs/decay/20260920T033855-afe093ad-fc776783/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.01 |
| [20260920T080010-083e8547-f7b71964](../runs/decay/20260920T080010-083e8547-f7b71964/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.05 |
| [20260920T080010-5228f676-0d294512](../runs/decay/20260920T080010-5228f676-0d294512/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.2 |
| [20260920T080010-57540484-e3871f2e](../runs/decay/20260920T080010-57540484-e3871f2e/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.2 |
| [20260920T080010-7a67d71c-58026d1d](../runs/decay/20260920T080010-7a67d71c-58026d1d/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.05 |
| [20260920T080010-caa6f2fa-699d3276](../runs/decay/20260920T080010-caa6f2fa-699d3276/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.01 |
| [20260920T080010-d5b58c4c-ff5f2611](../runs/decay/20260920T080010-d5b58c4c-ff5f2611/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.01 |
| [20260920T080308-56c3e865-1ef1de37](../runs/decay/20260920T080308-56c3e865-1ef1de37/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.2 |
| [20260920T080308-632bc7f4-190891e8](../runs/decay/20260920T080308-632bc7f4-190891e8/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.05 |
| [20260920T080308-80367633-9d6c1bc0](../runs/decay/20260920T080308-80367633-9d6c1bc0/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.2 |
| [20260920T080308-9606ee5e-3ec5b5ad](../runs/decay/20260920T080308-9606ee5e-3ec5b5ad/manifest.json) | complete | sdirk2 | 32 × 32 | 1.0 | 0.01 |
| [20260920T080308-ba078555-b20a7258](../runs/decay/20260920T080308-ba078555-b20a7258/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.01 |
| [20260920T080308-cebc925b-d90c3a9b](../runs/decay/20260920T080308-cebc925b-d90c3a9b/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 1.0 | 0.05 |
| [20260920T080557-0bb6157d-9e4c4042](../runs/decay/20260920T080557-0bb6157d-9e4c4042/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.001 |
| [20260921T033037-e5cdbd9c-8e5b788b](../runs/decay/20260921T033037-e5cdbd9c-8e5b788b/manifest.json) | running | sdirk2 | 1024 × 1024 | 0.02 | 0.001 |
| [20260922T023439-85ea93c5-d47e7349](../runs/forced_ns/20260922T023439-85ea93c5-d47e7349/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 0.05 |
| [20260922T023440-cc139ded-4d9cfbc7](../runs/forced_ns/20260922T023440-cc139ded-4d9cfbc7/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 2.0 | 0.05 |
| [20260922T023441-d2425b1c-b54c4bf9](../runs/forced_ns/20260922T023441-d2425b1c-b54c4bf9/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 0.025 |
| [20260922T023442-f0acd9a0-75afe613](../runs/forced_ns/20260922T023442-f0acd9a0-75afe613/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 2.0 | 0.025 |
| [20260922T023444-595f3005-44ab9ac3](../runs/forced_ns/20260922T023444-595f3005-44ab9ac3/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 0.0125 |
| [20260922T023446-f84db035-9ddb64e2](../runs/forced_ns/20260922T023446-f84db035-9ddb64e2/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 2.0 | 0.0125 |
| [20260922T023449-6f443c87-8a3ec06a](../runs/forced_ns/20260922T023449-6f443c87-8a3ec06a/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 0.00625 |
| [20260922T023453-49d3f95b-51c5fa7d](../runs/forced_ns/20260922T023453-49d3f95b-51c5fa7d/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 2.0 | 0.00625 |
| [20260922T023459-3c9ad6ac-de46694e](../runs/forced_ns/20260922T023459-3c9ad6ac-de46694e/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 0.003125 |
| [20260922T023506-fa20a291-15163bd7](../runs/forced_ns/20260922T023506-fa20a291-15163bd7/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 2.0 | 0.003125 |
| [20260922T023518-fd50cceb-f50f0adf](../runs/forced_ns/20260922T023518-fd50cceb-f50f0adf/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 0.0015625 |
| [20260922T023532-824fb9d8-56e666c4](../runs/forced_ns/20260922T023532-824fb9d8-56e666c4/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 2.0 | 0.0015625 |
| [20260922T023555-a1ed289b-013a07e4](../runs/forced_ns/20260922T023555-a1ed289b-013a07e4/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 0.00078125 |
| [20260922T023622-ae131528-463c246c](../runs/forced_ns/20260922T023622-ae131528-463c246c/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 2.0 | 0.00078125 |
| [20260922T023706-1e19aa6e-f97d4254](../runs/forced_ns/20260922T023706-1e19aa6e-f97d4254/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 0.000390625 |
| [20260922T023759-4763e383-7a5514e2](../runs/forced_ns/20260922T023759-4763e383-7a5514e2/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 2.0 | 0.000390625 |
| [20260922T023927-aac7ffa3-28969e18](../runs/forced_ns/20260922T023927-aac7ffa3-28969e18/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 2.44140625e-05 |
| [20260922T025353-74ead0a4-3c1b191e](../runs/forced_ns/20260922T025353-74ead0a4-3c1b191e/manifest.json) | complete | sdirk2 | 128 × 128 | 2.0 | 1.220703125e-05 |
| [20260921T053257-55808f91-e1b30707](../runs/mac_parallel/20260921T053257-55808f91-e1b30707/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.05 | 0.01 |
| [20260921T053326-294793bf-60383047](../runs/mac_parallel/20260921T053326-294793bf-60383047/manifest.json) | complete | sdirk2 | 64 × 64 | 0.01 | 0.001 |
| [20260920T033847-34fefaad-aba46c69](../runs/ns_mms/20260920T033847-34fefaad-aba46c69/manifest.json) | complete | sdirk2 | 64 × 64 | 0.02 | 0.0005 |
| [20260920T033847-414e33b7-e0c61e0a](../runs/ns_mms/20260920T033847-414e33b7-e0c61e0a/manifest.json) | complete | sdirk2 | 32 × 32 | 0.02 | 0.0005 |
| [20260920T033847-d2c255aa-d117cbe9](../runs/ns_mms/20260920T033847-d2c255aa-d117cbe9/manifest.json) | complete | sdirk2 | 128 × 128 | 0.02 | 0.0005 |
| [20260920T033849-556b6553-800be582](../runs/ns_mms/20260920T033849-556b6553-800be582/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.02 | 0.0005 |
| [20260920T033849-7696a8fd-5dfe6ec3](../runs/ns_mms/20260920T033849-7696a8fd-5dfe6ec3/manifest.json) | complete | sdirk2 | 48 × 32 | 0.02 | 0.0005 |
| [20260920T033849-a1c5c7f9-aa605809](../runs/ns_mms/20260920T033849-a1c5c7f9-aa605809/manifest.json) | complete | sdirk2_mrsav | 64 × 64 | 0.02 | 0.0005 |
| [20260920T033850-6b3252ae-2c05d600](../runs/ns_mms/20260920T033850-6b3252ae-2c05d600/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 0.02 | 0.0005 |
| [20260920T033852-d0723d7f-2436018b](../runs/ns_mms/20260920T033852-d0723d7f-2436018b/manifest.json) | complete | sdirk2_mrsav | 48 × 32 | 0.02 | 0.0005 |
| [20260920T033853-09bc4d56-b6161474](../runs/ns_mms/20260920T033853-09bc4d56-b6161474/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.0003125 |
| [20260920T033853-758e23c0-bbaeda29](../runs/ns_mms/20260920T033853-758e23c0-bbaeda29/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.01 |
| [20260920T033853-9ad92352-a6dec135](../runs/ns_mms/20260920T033853-9ad92352-a6dec135/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.005 |
| [20260920T033853-d179da9d-8b527f86](../runs/ns_mms/20260920T033853-d179da9d-8b527f86/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.000625 |
| [20260920T033853-ec7bf701-e061b23c](../runs/ns_mms/20260920T033853-ec7bf701-e061b23c/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.0025 |
| [20260920T033854-098e7993-ff063714](../runs/ns_mms/20260920T033854-098e7993-ff063714/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.0003125 |
| [20260920T033854-1640ee42-88a03402](../runs/ns_mms/20260920T033854-1640ee42-88a03402/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.000625 |
| [20260920T033854-3710e080-50b00139](../runs/ns_mms/20260920T033854-3710e080-50b00139/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.005 |
| [20260920T033854-a1ac46a5-12d24c6f](../runs/ns_mms/20260920T033854-a1ac46a5-12d24c6f/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.01 |
| [20260920T033854-e99860ba-a45f51b5](../runs/ns_mms/20260920T033854-e99860ba-a45f51b5/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.0025 |
| [20260920T080004-9172fb77-ddedfb47](../runs/ns_mms/20260920T080004-9172fb77-ddedfb47/manifest.json) | complete | sdirk2 | 32 × 32 | 0.02 | 0.0005 |
| [20260920T080005-5b8422cd-3aabe1fe](../runs/ns_mms/20260920T080005-5b8422cd-3aabe1fe/manifest.json) | complete | sdirk2 | 64 × 64 | 0.02 | 0.0005 |
| [20260920T080005-7f6dd3e1-bd11da7f](../runs/ns_mms/20260920T080005-7f6dd3e1-bd11da7f/manifest.json) | complete | sdirk2 | 128 × 128 | 0.02 | 0.0005 |
| [20260920T080006-2f75a574-997f6fc7](../runs/ns_mms/20260920T080006-2f75a574-997f6fc7/manifest.json) | complete | sdirk2 | 48 × 32 | 0.02 | 0.0005 |
| [20260920T080006-3ee53c2d-e5215ddd](../runs/ns_mms/20260920T080006-3ee53c2d-e5215ddd/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 0.02 | 0.0005 |
| [20260920T080006-bbf9ca07-d1e36c5b](../runs/ns_mms/20260920T080006-bbf9ca07-d1e36c5b/manifest.json) | complete | sdirk2_mrsav | 64 × 64 | 0.02 | 0.0005 |
| [20260920T080006-c677b0b6-3fd778cc](../runs/ns_mms/20260920T080006-c677b0b6-3fd778cc/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.02 | 0.0005 |
| [20260920T080007-ab580266-200053d8](../runs/ns_mms/20260920T080007-ab580266-200053d8/manifest.json) | complete | sdirk2_mrsav | 48 × 32 | 0.02 | 0.0005 |
| [20260920T080008-02d99219-994765d6](../runs/ns_mms/20260920T080008-02d99219-994765d6/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.0025 |
| [20260920T080008-94869c4e-7cc25b38](../runs/ns_mms/20260920T080008-94869c4e-7cc25b38/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.0003125 |
| [20260920T080008-9b27e747-a1427b89](../runs/ns_mms/20260920T080008-9b27e747-a1427b89/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.01 |
| [20260920T080008-9bde8ad7-5dc45e99](../runs/ns_mms/20260920T080008-9bde8ad7-5dc45e99/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.000625 |
| [20260920T080008-e14cb145-2522a717](../runs/ns_mms/20260920T080008-e14cb145-2522a717/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.005 |
| [20260920T080009-09e811bd-55461417](../runs/ns_mms/20260920T080009-09e811bd-55461417/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.0025 |
| [20260920T080009-66697e3b-faf928af](../runs/ns_mms/20260920T080009-66697e3b-faf928af/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.0003125 |
| [20260920T080009-d989b89d-f9d941e2](../runs/ns_mms/20260920T080009-d989b89d-f9d941e2/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.000625 |
| [20260920T080009-eeec1d76-f62d3e98](../runs/ns_mms/20260920T080009-eeec1d76-f62d3e98/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.01 |
| [20260920T080009-ff06185a-09d779c8](../runs/ns_mms/20260920T080009-ff06185a-09d779c8/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.005 |
| [20260920T080302-3b500e1f-3571d1f4](../runs/ns_mms/20260920T080302-3b500e1f-3571d1f4/manifest.json) | complete | sdirk2 | 32 × 32 | 0.02 | 0.0005 |
| [20260920T080302-595a3d89-8a2b62c7](../runs/ns_mms/20260920T080302-595a3d89-8a2b62c7/manifest.json) | complete | sdirk2 | 128 × 128 | 0.02 | 0.0005 |
| [20260920T080302-8f3e4c8b-b7adc11b](../runs/ns_mms/20260920T080302-8f3e4c8b-b7adc11b/manifest.json) | complete | sdirk2 | 64 × 64 | 0.02 | 0.0005 |
| [20260920T080303-b3fe1656-6d3bd4d4](../runs/ns_mms/20260920T080303-b3fe1656-6d3bd4d4/manifest.json) | complete | sdirk2 | 48 × 32 | 0.02 | 0.0005 |
| [20260920T080303-c0315060-3abd9ae6](../runs/ns_mms/20260920T080303-c0315060-3abd9ae6/manifest.json) | complete | sdirk2_mrsav | 64 × 64 | 0.02 | 0.0005 |
| [20260920T080303-ff752d2c-4ace8f85](../runs/ns_mms/20260920T080303-ff752d2c-4ace8f85/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.02 | 0.0005 |
| [20260920T080304-0cc4c818-489a7bbb](../runs/ns_mms/20260920T080304-0cc4c818-489a7bbb/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 0.02 | 0.0005 |
| [20260920T080305-9f51950f-2bfaf092](../runs/ns_mms/20260920T080305-9f51950f-2bfaf092/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.01 |
| [20260920T080305-e6904738-d95ddb4c](../runs/ns_mms/20260920T080305-e6904738-d95ddb4c/manifest.json) | complete | sdirk2_mrsav | 48 × 32 | 0.02 | 0.0005 |
| [20260920T080306-3446c2a4-5487f0ae](../runs/ns_mms/20260920T080306-3446c2a4-5487f0ae/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.0025 |
| [20260920T080306-8abeafa1-3c3e1de3](../runs/ns_mms/20260920T080306-8abeafa1-3c3e1de3/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.005 |
| [20260920T080306-ac17e5e8-c7486271](../runs/ns_mms/20260920T080306-ac17e5e8-c7486271/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.0003125 |
| [20260920T080306-b9788e08-6857559b](../runs/ns_mms/20260920T080306-b9788e08-6857559b/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.000625 |
| [20260920T080306-d76af349-2a12b053](../runs/ns_mms/20260920T080306-d76af349-2a12b053/manifest.json) | complete | sdirk2 | 32 × 32 | 0.1 | 0.005 |
| [20260920T080306-fd511bac-53357582](../runs/ns_mms/20260920T080306-fd511bac-53357582/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.01 |
| [20260920T080307-6417cf8a-fb8b0a58](../runs/ns_mms/20260920T080307-6417cf8a-fb8b0a58/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.0003125 |
| [20260920T080307-cfd49d17-9385debe](../runs/ns_mms/20260920T080307-cfd49d17-9385debe/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.000625 |
| [20260920T080307-f797fe94-384b32b7](../runs/ns_mms/20260920T080307-f797fe94-384b32b7/manifest.json) | complete | sdirk2_mrsav | 32 × 32 | 0.1 | 0.0025 |
| [20260920T033846-06a2d0dc-61fe1a54](../runs/stokes_mms/20260920T033846-06a2d0dc-61fe1a54/manifest.json) | complete | sdirk2 | 32 × 32 | 0.02 | 0.0005 |
| [20260920T033846-87b54ed3-6d209300](../runs/stokes_mms/20260920T033846-87b54ed3-6d209300/manifest.json) | complete | sdirk2 | 64 × 64 | 0.02 | 0.0005 |
| [20260920T033846-e60ce7a6-3aafe192](../runs/stokes_mms/20260920T033846-e60ce7a6-3aafe192/manifest.json) | complete | sdirk2 | 128 × 128 | 0.02 | 0.0005 |
| [20260920T033847-7fa247f9-b1756508](../runs/stokes_mms/20260920T033847-7fa247f9-b1756508/manifest.json) | complete | sdirk2 | 48 × 32 | 0.02 | 0.0005 |
| [20260920T080004-31a18d62-3e8e3500](../runs/stokes_mms/20260920T080004-31a18d62-3e8e3500/manifest.json) | complete | sdirk2 | 48 × 32 | 0.02 | 0.0005 |
| [20260920T080004-5bb668e9-700e2b53](../runs/stokes_mms/20260920T080004-5bb668e9-700e2b53/manifest.json) | complete | sdirk2 | 128 × 128 | 0.02 | 0.0005 |
| [20260920T080004-66596cbe-7abd8baf](../runs/stokes_mms/20260920T080004-66596cbe-7abd8baf/manifest.json) | complete | sdirk2 | 64 × 64 | 0.02 | 0.0005 |
| [20260920T080004-9ac682ac-6df11d5a](../runs/stokes_mms/20260920T080004-9ac682ac-6df11d5a/manifest.json) | complete | sdirk2 | 32 × 32 | 0.02 | 0.0005 |
| [20260920T080301-076acf17-913d1139](../runs/stokes_mms/20260920T080301-076acf17-913d1139/manifest.json) | complete | sdirk2 | 32 × 32 | 0.02 | 0.0005 |
| [20260920T080301-587e8733-fbcdc39a](../runs/stokes_mms/20260920T080301-587e8733-fbcdc39a/manifest.json) | complete | sdirk2 | 64 × 64 | 0.02 | 0.0005 |
| [20260920T080301-8607b90c-0fbd0f48](../runs/stokes_mms/20260920T080301-8607b90c-0fbd0f48/manifest.json) | complete | sdirk2 | 128 × 128 | 0.02 | 0.0005 |
| [20260920T080302-0a09c019-b198fc1e](../runs/stokes_mms/20260920T080302-0a09c019-b198fc1e/manifest.json) | complete | sdirk2 | 48 × 32 | 0.02 | 0.0005 |
| [20260922T044558-3c9b89a7-be08794c](../runs/trig_ns/20260922T044558-3c9b89a7-be08794c/manifest.json) | complete | sdirk2 | 128 × 128 | 1.0 | 0.1 |
| [20260922T044558-7970b3a7-d6df8472](../runs/trig_ns/20260922T044558-7970b3a7-d6df8472/manifest.json) | complete | sdirk2_mrsav | 128 × 128 | 1.0 | 0.1 |
| [20260922T044559-9d2a48ac-13515dc0](../runs/trig_ns/20260922T044559-9d2a48ac-13515dc0/manifest.json) | complete | sdirk2 | 128 × 128 | 1.0 | 0.025 |
| [20260922T044600-7dbbc997-2aaeb761](../runs/trig_ns/20260922T044600-7dbbc997-2aaeb761/manifest.json) | complete | sdirk2 | 128 × 128 | 1.0 | 0.0125 |
| [20260922T050733-3f4ed806-1c7ebcb8](../runs/trig_ns/20260922T050733-3f4ed806-1c7ebcb8/manifest.json) | complete | sdirk2 | 256 × 256 | 4.0 | 1.0 |
| [20260922T050736-3c2b08cc-a32c7544](../runs/trig_ns/20260922T050736-3c2b08cc-a32c7544/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 1.0 |
| [20260922T050739-0aca4f24-41c55fb1](../runs/trig_ns/20260922T050739-0aca4f24-41c55fb1/manifest.json) | failed | sdirk2 | 256 × 256 | 4.0 | 0.5 |
| [20260922T050743-9c968ca3-9f596a25](../runs/trig_ns/20260922T050743-9c968ca3-9f596a25/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 0.5 |
| [20260922T050746-f0d3c178-8654cc8d](../runs/trig_ns/20260922T050746-f0d3c178-8654cc8d/manifest.json) | failed | sdirk2 | 256 × 256 | 4.0 | 0.25 |
| [20260922T050750-36f53128-5e3c99d5](../runs/trig_ns/20260922T050750-36f53128-5e3c99d5/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 0.25 |
| [20260922T050754-509f1cde-e897fac7](../runs/trig_ns/20260922T050754-509f1cde-e897fac7/manifest.json) | failed | sdirk2 | 256 × 256 | 4.0 | 0.125 |
| [20260922T050758-6a8c894a-fc467521](../runs/trig_ns/20260922T050758-6a8c894a-fc467521/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 0.125 |
| [20260922T050803-8d1598d4-a27d670d](../runs/trig_ns/20260922T050803-8d1598d4-a27d670d/manifest.json) | failed | sdirk2 | 256 × 256 | 4.0 | 0.0625 |
| [20260922T050808-68ef48ca-fe930eec](../runs/trig_ns/20260922T050808-68ef48ca-fe930eec/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 0.0625 |
| [20260922T050816-3d564e41-590e1a36](../runs/trig_ns/20260922T050816-3d564e41-590e1a36/manifest.json) | failed | sdirk2 | 256 × 256 | 4.0 | 0.03125 |
| [20260922T050825-45f61c01-404cfdea](../runs/trig_ns/20260922T050825-45f61c01-404cfdea/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 0.03125 |
| [20260922T050839-b94879e4-f620a653](../runs/trig_ns/20260922T050839-b94879e4-f620a653/manifest.json) | complete | sdirk2 | 256 × 256 | 4.0 | 0.015625 |
| [20260922T050856-ead8be09-199ac7d5](../runs/trig_ns/20260922T050856-ead8be09-199ac7d5/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 0.015625 |
| [20260922T050921-8ddfc3fb-882a4522](../runs/trig_ns/20260922T050921-8ddfc3fb-882a4522/manifest.json) | complete | sdirk2 | 256 × 256 | 4.0 | 0.0078125 |
| [20260922T050952-5b610954-85ddb795](../runs/trig_ns/20260922T050952-5b610954-85ddb795/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 0.0078125 |
| [20260922T051040-7aa160a3-f6f0264f](../runs/trig_ns/20260922T051040-7aa160a3-f6f0264f/manifest.json) | complete | sdirk2 | 256 × 256 | 4.0 | 0.00390625 |
| [20260922T051138-48c10af3-8a573845](../runs/trig_ns/20260922T051138-48c10af3-8a573845/manifest.json) | complete | sdirk2_mrsav | 256 × 256 | 4.0 | 0.00390625 |
| [20260922T051312-573063b5-169e6a82](../runs/trig_ns/20260922T051312-573063b5-169e6a82/manifest.json) | complete | sdirk2 | 256 × 256 | 4.0 | 0.0009765625 |
| [20260922T051656-41f91c7f-8a5a436f](../runs/trig_ns/20260922T051656-41f91c7f-8a5a436f/manifest.json) | complete | sdirk2 | 256 × 256 | 4.0 | 0.00048828125 |
| [20260922T052515-5cccf8d8-8af9031d](../runs/trig_ns/20260922T052515-5cccf8d8-8af9031d/manifest.json) | complete | sdirk2 | 256 × 256 | 4.0 | 0.000244140625 |

## 分析记录

以下只检查 analysis.json 存在与记录状态，不重新启动计算。

- [reports/cavity/20260920T081722-d70424f1](../reports/cavity/20260920T081722-d70424f1/analysis.json)：complete
- [reports/cavity/20260920T081815-2597891e](../reports/cavity/20260920T081815-2597891e/analysis.json)：complete
- [reports/cavity/20260920T122436-b329640b](../reports/cavity/20260920T122436-b329640b/analysis.json)：complete
- [reports/cavity/20260920T122923-a6ba2f34](../reports/cavity/20260920T122923-a6ba2f34/analysis.json)：complete
- [reports/cavity/20260922T074918-2d55800f](../reports/cavity/20260922T074918-2d55800f/analysis.json)：complete
- [reports/diagnostics/20260920T033853-ef223c25](../reports/diagnostics/20260920T033853-ef223c25/analysis.json)：complete
- [reports/diagnostics/20260920T033854-acf2a51f](../reports/diagnostics/20260920T033854-acf2a51f/analysis.json)：complete
- [reports/diagnostics/20260920T033855-98ab037d](../reports/diagnostics/20260920T033855-98ab037d/analysis.json)：complete
- [reports/diagnostics/20260920T033855-f2ed8e7c](../reports/diagnostics/20260920T033855-f2ed8e7c/analysis.json)：complete
- [reports/diagnostics/20260920T080008-30c01b93](../reports/diagnostics/20260920T080008-30c01b93/analysis.json)：complete
- [reports/diagnostics/20260920T080009-ad6fc966](../reports/diagnostics/20260920T080009-ad6fc966/analysis.json)：complete
- [reports/diagnostics/20260920T080010-9069591b](../reports/diagnostics/20260920T080010-9069591b/analysis.json)：complete
- [reports/diagnostics/20260920T080010-e63166ca](../reports/diagnostics/20260920T080010-e63166ca/analysis.json)：complete
- [reports/diagnostics/20260920T080305-40deeafb](../reports/diagnostics/20260920T080305-40deeafb/analysis.json)：complete
- [reports/diagnostics/20260920T080306-e6dec9b4](../reports/diagnostics/20260920T080306-e6dec9b4/analysis.json)：complete
- [reports/diagnostics/20260920T080307-65be408f](../reports/diagnostics/20260920T080307-65be408f/analysis.json)：complete
- [reports/diagnostics/20260920T080308-6e0abaf8](../reports/diagnostics/20260920T080308-6e0abaf8/analysis.json)：complete
- [reports/diagnostics/20260920T080603-041a3227](../reports/diagnostics/20260920T080603-041a3227/analysis.json)：complete
- [reports/forced_ns_convergence/20260922T032418-3cd92bd4](../reports/forced_ns_convergence/20260922T032418-3cd92bd4/analysis.json)：complete
- [reports/trig_ns_convergence/20260922T044601-2356ffb1](../reports/trig_ns_convergence/20260922T044601-2356ffb1/analysis.json)：complete
- [reports/trig_ns_convergence/20260922T044601-b3c4e81c](../reports/trig_ns_convergence/20260922T044601-b3c4e81c/analysis.json)：complete
- [reports/trig_ns_convergence/20260922T052514-1dacc8ed](../reports/trig_ns_convergence/20260922T052514-1dacc8ed/analysis.json)：complete
- [reports/trig_ns_convergence/20260922T054144-5aee3c91](../reports/trig_ns_convergence/20260922T054144-5aee3c91/analysis.json)：complete
