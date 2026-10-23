# HiSparse 本轮代码审阅（2026-10-23）

这是一对 fork 草稿，用来审阅实际实验快照。公共 vLLM 与 Ascend 插件需要成对使用；它们不是已经完成验证的上游功能发布。

## 先看哪些文件

| 模块 | 主要文件 | 本轮变化 |
| --- | --- | --- |
| 公共 runtime | `vllm/v1/hisparse/runtime.py`、`backend.py` | 把 MTP 多 token Decode 的组预取条件与单 token resident 写入/Host mirror 条件分开；resident gather 交给设备 backend。 |
| Ascend 接口与 KV 格式 | `distributed/kv_transfer/hisparse/{backend,binding}.py`、`attention/indexer.py`、`core/kv_cache_interface.py` | attention 和 Indexer 的 FP8 E4M3FN + FP32 scales 布局、视图和 opaque payload 绑定；按实际页步长搬运。 |
| Decode 搬运与算子 | `ops/hisparse.py`、`ops/triton/hisparse.py`、`ops/triton/hisparse_resident_gather.py` | 减少中间变换，调整 gather/映射与 LRU/搬运路径；保留实际步长、当前 token 与图 owner 的约束。包含 Triton 路径，未证明达到最优性能。 |
| attention C8 | `attention/sfa_hisparse.py`、`ops/hisparse_c8*.py`、`ops/hisparse_direct.py` | 量化 attention KV 的 staging、物理位置映射及稀疏 attention 接口；C8 不是并发 8，也不只是 Indexer 量化。 |
| P/D 控制与传输 | `distributed/kv_transfer/kv_p2p/hisparse_pd/` | SOURCE + INDEXER 的显式布局、ticket/lease、地址计划、分段 HBM stage、完成后 ACK、取消和关闭流程。 |
| MTP 与图生命周期 | 同目录 `execution.py`、`owners.py`、`import_session.py`；`worker/v2/model_runner.py` | 从实际 loaded layer 名字绑定 target/draft 所有权；只交接已完成前缀，D 端重建 proposals；等公共 draft mirror 完成才发布；校验图 owner、地址与注册生命周期。 |

## P/D 的完整路径

1. 初始化时公共 HiSparse 仍负责逻辑 block、Host 池和完成通知。插件绑定 SOURCE/INDEXER 与实际 target/draft 层，不改变共享 cache group 的规划标志。
2. P 完成前缀后等待公共 draft mirror fence，发布带模型身份、格式、已完成范围、rank 与 lease 的 ticket。暂态 speculative lookahead 不作为已完成状态交接。
3. D 沿用公共 Host import/分配路径，按 P/D 各自 block 表生成真实字节地址计划。两侧物理 block ID 不要求相同。
4. MemFabric 只注册真实 ACL HBM stage；SOURCE 经有界 staging 传输。D 确认 D2H 完成之后才 ACK，避免提前释放 P 的前缀。
5. D 导入 SOURCE/INDEXER 后重建本地 Resident/Hot 状态及 MTP proposals，再进入实际 Decode 图回放。
6. 超时、读失败与关闭时保留仍可能被访问的 owner。先排空传输并关闭 reader/stage，再解除公共 Host 池的 pin。

`FULL_DECODE_ONLY` 只表示 Decode 图；Prefill 仍按实际路径执行。当前只开放代码中明确校验的同 checkpoint MTP3 合同，PP/CP/多 DP、其他 draft/图模式仍有 guard。

## 已验证与未验证

| 项目 | 实际结果 |
| --- | --- |
| 本次完整旧 P/D CPU 套件 | 118 项：79 通过、39 失败。旧夹具/旧 wire/layout 合同存在不兼容，未逐一排除真实实现问题。失败测试名称和原因在 JSON 清单中。 |
| 新增 graph/MTP 合同检查 | 8/8 通过；覆盖已完成前缀、lookahead 拒绝、不支持 draft 模式、loaded ownership、mirror fence 和绑定失效。 |
| 减层真实权重 P/D | 7 target + 原 MTP 层、同机 TP4+TP4、双 FP8、MTP3；model398 的 15/15、model404 的 15/15、model414 的 9/9 fixture 请求完成。计数包含 baseline/P-only/D-import 请求，不表示全部都是独立完整 P/D 请求。 |
| 减层输出和生命周期 | 对照实际 draft/输出；model414 的 8 个 worker/reader/stage 均在 Host unpin 前关闭。P API 仍出现关闭阶段 EngineDeadError，日志不干净；未宣称修复。诊断 observer 引入同步，不能用作正式性能结果。 |
| 完整 78+1 双机部署 | D 的 8 个 rank 完成图捕获；P 加载到 37/99 时 supervisor 被 SIGKILL。发送者未知，未见相应 OOM 证据。模型验证请求 0，统一入口与 131072+2048 请求未验证。自己的活动模型和 lease 已清理，保留僵尸进程未被容器 init 回收的记录。 |
| 1M、最大并发、共享 KV 池、多 DP | 本轮未验收。不能据此宣称支持。 |
| 精度与性能 | 广泛精度按用户要求暂缓；没有独立 publication-head 的完整模型性能/精度结果，不宣称 TPOT 退化已解决。 |

CPU 检查使用真实 CPU torch 和实际生产定义；配套 fixture 用 AST 隔离不相关的 vLLM 导入。它不是标准全量 pytest 或硬件验证。复现（准备好成对源码和依赖后，在 Ascend 仓库）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON" -m pytest -q -p no:cacheprovider \
  --confcutdir=tests/ut/distributed/kv_transfer/hisparse_pd \
  tests/ut/distributed/kv_transfer/hisparse_pd
```

## 审阅边界

- 更新现有 fork 草稿，保留之前的提交；本次增量 commit 可以单独打开查看。两个面向上游的窄接口 PR 保持独立。
- 源码取自冻结 candidate395；除 LF 换行外保留运行过的源码内容。新文档/测试单独标记。自动生成的 `_build_info.py` 和 `_version.py` 不提交。
- 不发布 `.so`、模型、SSH 凭据或机器配置。实验用 native 扩展 SHA256 写入清单；未重新编译，因此不能把这次发布描述为独立构建验证通过。
- 保留旧失败、延迟退化和关闭问题。Ruff/格式检查如有失败，写入 publication JSON；不为格式化修改已验证的运行路径。
- 本草稿由 OpenAI Codex 协助整理；仍需人工逐行审阅、修复失败测试、完成标准 CI 与最终硬件验证。
