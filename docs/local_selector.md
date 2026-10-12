# 本地条件边际收益 selector（未训练实验版）

## 改动边界

保留现有 memory 生成、MEG 视图选择、候选融合、reranker 和可选 rollback。
新 selector 接在这些阶段之后，对 child/parent 候选联合选取，然后交给原有 reader。
不新增 query plan 字段，不改持久化 memory 结构；`token_cost` 只在本次选择的运行时数据和诊断中计算。
没有新增 `count_evidence_tokens(payload)` 之类的包装函数。
不按 category 决定答案类型，也不强制 category 2 只输出时间。

旧的 `llm` selector 保留作为默认对照；新机制必须显式启用 `local`。
旧路径的备用候选行为不变；**local 路径不传 Top-12 backup、不透传未选 parent**。
reader JSON 解析失败时只用同一份选中证据重试，不恢复被删除的候选。
低收益可以返回空集；模型加载、推理异常会让该问题明确报错，不冒充 selector 成功后回退 TopK。

## 主循环与边界

设原问题为 q，当前已选集合为 S，候选为 m，证据预算为 B：

```text
S = []
while 尚有候选:
    F = 加入后完整序列化证据包不超过 B 的候选
    batch_score(q, S, F) -> 每个候选的 conditional yes/no 分数 g
    r(m,S) = max(0, (max_cosine(m,S) - tau) / (1 - tau))
    u(m|q,S) = g(q,S,m) - lambda_r * r(m,S) - lambda_t * token_cost(m) / B
    若没有可评分候选，或最高 u <= min_gain：停止
    把最高 u 的一条加入 S，从剩余候选中移除
    下一轮重新基于完整的新 S 评分
```

相似度使用已经缓存的 retrieval embedding；没有向量、零向量或维度不匹配时不施加该项惩罚，不发起新 embedding 调用。
相似度仅为软惩罚；高相似不等于没有新增事实，因此不会硬删除。
可用 `--selector_redundancy_weight 0` 关闭显式重复惩罚。
平分按上游候选顺序打破，跨 child/parent ID 用类型区分。

预算直接用 selector tokenizer 统计 reader 可见 evidence package 和 parent memories 的 JSON，包含容器、ID、时间和原有分数字段。
每次尝试加入都核验整个证据包，而不只累加独立 memory 的 token 数。
这是 **reader token 成本的代理**，不是远端 reader tokenizer 的精确计费；不包括问题、query plan、system prompt 和最终回答。
`--selector_max_length` 则限制评分器的完整输入，与上述预算不同。
原问题、已选证据、候选和 yes/no 后缀不会被静默截断；超长候选跳过并记录 `scorer_input_length`。

评分器沿用 [Qwen3-Reranker 官方 Transformers 示例](https://huggingface.co/Qwen/Qwen3-Reranker-4B#using-transformers)
的 causal LM 与 yes/no token 打分方式，改为判断候选相对于已选集合是否提供新的答题信息。
它不生成答案、也不新增远端 selector API 调用；但仍使用本地语言模型的前向推理。
批次左侧 padding，并同步 tokenizer/model 的 pad token ID；权重进程内惰性共享，加载与 GPU 推理分别加锁。

**g 只是未训练的条件增益代理，不是正确率增量，也不保证能识别完整证据链。**
更不能把停止条件解释成“已经证明可完整回答”。互补链条可能需要多个单独低分的节点，本版逐条贪心仍可能漏选它们。
所有阈值为实验初值，需要实际消融，不能用加载测试或规则测试证明准确率提升。

## AutoDL 使用

本机没有足够空间和预算下载模型，因此没有运行真实权重加载或 GPU 推理。
本地验证仅覆盖 mock scorer、mock tokenizer/model 接口和已有回归测试；实际兼容性、显存和质量需在 AutoDL 验证。

在项目已有环境中安装可选依赖，复用现有 CUDA torch：

```bash
pip install -r requirements-selector.txt
```

模型必须已下载到本地目录；所有 `from_pretrained` 调用都设置 `local_files_only=True`。
可通过 `.env` 的 `SELECTOR_MODEL_PATH` 或命令行指定目录；命令行优先。
以下命令在包含 `run.py` 的项目目录运行，把模型路径、sample、模型别名换成现有实验的值。

先做条件打分能力探测，不调用远端 reader：

```bash
python -m eval.evaluate_selector \
  --cases eval/selector_probe_cases.jsonl \
  --selector_model_path /root/autodl-tmp/models/Qwen3-Reranker-4B \
  --selector_batch_size 2
```

探测输出包括候选分数、输入长度、排序检查、加入重复信息后分数是否下降。
示例是少量人工构造病例，不是 benchmark；通过也不代表真实错例有效。
可以替换为真实候选，JSONL 每行格式为：

```json
{"case_id":"example","question":"What pets does Ana own?","selected":[],"candidates":[{"memory_id":"a","rewrite_content":"Ana owns a cat."},{"memory_id":"b","rewrite_content":"Bob likes tennis."}],"expanded_selected":[{"memory_id":"seen","rewrite_content":"Ana has a pet cat."}],"expected_order":[["a","b"]],"expected_decrease":["a"]}
```

`expected_*` 只用于事后检查，不进入模型。候选 ID 在探测文件每个病例内需唯一。
`expanded_selected` 是完整的扩展后 S，不是自动追加列表。

再用已有流水线做少量端到端测试（这一步仍会调用原有检索/reader API）：

```bash
python run.py --data locomo --model gemini --eaes --semantic_hierarchy \
  --sample 26 --max_questions 5 --workers 1 --file selector_smoke \
  --evidence_selector_backend local \
  --selector_model_path /root/autodl-tmp/models/Qwen3-Reranker-4B \
  --selector_batch_size 2 --selector_token_budget 2048
```

先使用 `--workers 1`，确认显存和延迟再调并发。若显存不足可减小 batch 到 1；这不能解决单个长输入本身的显存需求。
不自动量化、切换 CPU 或下载其他模型。
local 结果文件增加 `_local_selector` 后缀，不改变 rewrite/embedding 缓存路径。
改变权重、预算或阈值时使用不同 `--file`，避免断点续跑混入旧配置。

## 诊断与对照

QA 结果中的 `selector` 保存每轮所有候选的 gain、redundancy、token cost、utility、跳过原因、选中节点、停止原因、调用次数和耗时。
`elapsed_seconds` 包括首次加载/锁等待；不是纯 GPU 推理耗时。
`selected_nodes` 是贪心入选顺序；reader 保持现有 child-package / parent-list 分组，`prediction_context` 则按实际 reader 分组内顺序记录，不再默认截到 20 个。
显式报告 Hit@20 的评估仍可取前 20 个；完整 provenance 不等于语义回答充分性。

加 `--retrieval_only` 可跳过最终 reader，但 **local selector 仍执行**。
原 `final_child`、`selected_parent`、`final_combined` 指标和对应候选保留为上游检索指标；
新增 `pre_selector`、`post_selector_child`、`post_selector_parent`、`post_selector` 指标及实际选中候选。
比较 reader 实际收到的证据时请看 `post_selector`，不要把上游 `final_combined` 当成新 selector 的输出。
聚合时给 `eval/evaluate_retrieval.py` 传入与 run 一致的
`--evidence_selector_backend local`、`--max_questions`、`--exclude_categories` 等文件后缀参数。

现有 QA 评估入口 `evaluate_reasoning.py` 则要求 `--file` 传入完整后缀。
例如上面的 smoke 测试对应：

```bash
python eval/evaluate_reasoning.py --data locomo --model gemini \
  --file selector_smoke_q5_eaes_hierarchy_local_selector --allfile
```

这会调用现有远端 judge，产生相应费用；本机没有执行该命令。

建议固定同一候选来源和 reader，比较：

- `--disable_evidence_selector`：无 selector，不加载本地模型。
- `--evidence_selector_backend llm`：旧 selector。
- `--evidence_selector_backend local`：本次机制。
- local + `--selector_redundancy_weight 0` / `--selector_cost_weight 0`：分离惩罚项影响（硬预算仍在）。

同时看真实问答正确率、完整证据保留、证据 token 数、scorer 前向批次和端到端耗时。
减少 reader token 不一定减少总延迟或总算力开销；本版最坏有多轮批量重评分。

## 本地无权重测试

```bash
python -m unittest discover -s tests
```

测试不下载模型、不调用真实模型或远端 API；缺失可选依赖或已标记停用的已有测试会 skip。
