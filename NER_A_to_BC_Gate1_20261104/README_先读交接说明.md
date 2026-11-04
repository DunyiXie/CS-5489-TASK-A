# A → B、C：NER 第一阶段交接

交接日期：2026-11-04。步骤 0、1、2、3、4、5a 已验收，Gate 1 通过。
这是组内开发交接包，不是课程最终提交包。B、C 各保存一份相同的包。

## 包中内容

- `project/src/`：九个已验收的 Python 模块。`data.py`、`metrics.py`、`prediction_io.py`、`experiment_protocol.py` 是共同接口；其余文件是它们的依赖、子词对齐、EDA、记忆基线与备份工具。不要只复制两个入口文件。
- `project/config/ner-experiment-v1/`：冻结协议、训练集 5 折与备用 3 折的完整索引。协议 SHA-256：`f70ce61993c32c158900623a495c61efd0a70c95fadbdc2cfb572cf9d01cec01`。
- `project/config/label_mapping.json`、`dataset_latest.json`、`step5a_accepted_data_reference.json`：标签顺序、数据来源/固定 commit、完整性记录与恢复依据。
- `project/config/ner-memory-baseline-v1/`：已验收的小型词典模型与最终配置。
- `project/results/`：步骤 2 EDA、步骤 3 对齐记录、步骤 5a 五张表、完整报告及开发/测试预测 JSONL；`project/figures/` 为步骤 2 的 PNG/PDF。
- `reference/step3/`：已验收的预处理配置与标签记录，用于核对 C 的分词设置。
- `reference/Development_Documentation_NER.docx`：原开发文档。存储方式以本交接说明为准，原文的共享 Drive 安排已改为运行时本地文件与手动 ZIP 交接。
- `notebooks/`：01 数据加载、02 EDA、03 预处理、04 评估、05a RuntimeFix。均为供参考/恢复的未执行本地版本。
- `setup/`：固定依赖清单及修正后的 Python 兼容性/依赖安装单元格。
- `evidence/`：本地验收摘要和来源 ZIP manifest；`HANDOFF_MANIFEST.json` 校验包内每个文件。

## 运行与恢复

1. 使用 Colab Enterprise 的 **Python 3.12** 运行时。5a 已验收环境为 Python 3.12.13。NumPy 2.0.2 / pandas 2.2.2 不用于 Python 3.13。
2. 将交接包上传运行时后解压。通过 Files 面板或 Python 标准库解压均可；不要挂载 Drive，不用 GCS bucket、OAuth/API key 或凭据文件。
3. `project/` 是代码/配置根目录。添加 `project/src` 到 `sys.path` 后导入模块；将实际根路径传给公共函数，不照抄历史 `/root/...` 日志路径。
4. 先运行 `setup/step5a_dependencies.py` 的内容检查共同评估依赖；若安装成功提示重启，按提示重启后重新检查。`requirements-project.txt` 还列出后续模型/绘图所需的固定版本，按自己的阶段安装；PyTorch/CUDA 版本使用实际运行时并另行记录。
5. 原始数据与 Arrow 分词缓存不在包中。B 可调用 `data.load_verified_dataset(project_root, accepted_reference, run_id)`，其中 `accepted_reference` 读取 `project/config/step5a_accepted_data_reference.json`。该函数会下载同一固定公开数据版本并核对来源哈希、标签和句序；run_id 使用新的唯一值。
6. C 可在 Python 3.12 上运行 03 notebook 恢复固定 tokenizer 和对齐数据。Notebook 默认根目录为 `Path.home()/cs5489_ner_local`；若改用解压后的 `project/`，将 `PROJECT_ROOT` 改为该实际目录，导出目录放在项目目录之外。
7. 04 和 05a 的结果已验收，无须为交接重复运行。已有模型的测试预测直接复用，尤其不要把 05a 全本运行当作环境初始化。每人运行后仍下载并保留自己的 ZIP。

## B 使用与回传

- 主责 5b–5d：LR、线性 SVM、CRF 与特征比较；之后步骤 7。
- 直接使用既定训练折；所有可学习变换必须在各折训练部分拟合。使用实体级 F1 选参，不能用 token accuracy 代替。
- F3 标准 POS/chunk 特征仅作研究对照，可部署版本要能处理原始文本。
- 回传候选参数/选择规则、CV 每折分数及均值/样本标准差、开发集整体与各类型 P/R/F1、预测 JSONL 和 sidecar、最终配置、源码、环境/运行记录及必要检查点恢复说明。

## C 使用与回传

- 主责步骤 6：BERT 与第二编码器、三个种子；另负责 F7、步骤 9 抽取器与演示。
- 使用已冻结的 tokenizer commit、RoBERTa `add_prefix_space=True`、max_length=256；仅首子词监督，其余/特殊/padding 为 -100。先重组为原始整句再评分，禁止逐 chunk 独立评分。
- 开发集选好设置后固定，并运行种子 42、52、62；报告每个种子及均值/样本标准差。不得按测试分数挑选最好种子。
- 回传超参数、每种子开发集指标与预测 JSONL/sidecar、验证集最佳检查点、tokenizer/config、训练曲线、环境/运行记录；为 B 的步骤 7 提供检查点，为 A 提供完整对齐的预测文件。

## 共同接口与测试纪律

- `metrics.evaluate_ner(gold, predictions)` 和 `metrics.entity_f1(...)` 接收原始句子的词级标签列表。严格实体边界+类型；固定 orphan-I 解码，保留 raw prediction，gold 不修正。
- 使用 `prediction_io.write_predictions(...)` 输出 JSONL 和 `.meta.json`；system 名称包含模型/特征/种子等唯一标识，传入同一冻结协议哈希。
- JSONL 字段：`sentence_id`、`tokens`、`gold_tags`、`pred_tags`、`raw_pred_tags`。保留原句序、原词及每词一个标签；放在 `project/results/predictions/`，不要自行改 schema。
- B、C 首先交付开发集结果。错误分析、参数选择与模型选择仅用开发集。最终配置/检查点冻结后，各最终模型测试一次；测试结果和包内基线测试文件仅供最终汇总，不用于调参。
- 不改公共 v1 接口、协议或折文件。如确有必要变更，先记录理由并与 A 协调新的版本，保留原版。

## 已验收基线，供最终报告使用

| 划分 | P (%) | R (%) | F1 (%) | F1 95% CI (%) |
|---|---:|---:|---:|---|
| Validation | 79.3536 | 60.7371 | 68.8084 | 67.3341–70.2871 |
| Test | 72.0994 | 46.7599 | 56.7286 | 55.2028–58.2321 |

Bootstrap 为原句重采样 2000 次，95% 百分位区间，seed 2026。文件中的分数为 0–1，上表为百分制。
测试 F1 比论文 59.61 低 2.8814 个百分点；流水线已验证，历史分数未完全复现。差异原因尚未确定，不能只归因于 IOB2，也不能据此调整测试匹配规则。

A 后续负责步骤 8、统一结果表与配对比较、报告统稿及最终打包。此包已准备在本机，尚未向任何组员发送。
