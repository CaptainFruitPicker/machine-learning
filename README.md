# Heart Disease Risk Stratification

> 基于结构化临床指标的心脏病风险识别原型。项目完成了数据清洗、逻辑回归建模、效果评估与关键风险因素解释，并沉淀为可复现的脚本化流程。

## 项目成果

| 维度 | 结果 |
| --- | --- |
| 数据规模 | 272 条样本，13 个结构化特征 |
| 建模方案 | 基于 `NumPy + pandas` 自实现逻辑回归训练流程 |
| 验证集 Accuracy | 87.04% |
| 验证集 Precision | 86.96% |
| 验证集 Recall | 83.33% |
| 验证集 F1 Score | 85.11% |

## 项目价值

- 面向心脏病风险分层场景，构建从数据准备到结果解释的完整机器学习流程。
- 在保证结果可复现的前提下，将模型训练、评估和解释拆分为独立脚本，便于后续扩展与复用。
- 输出风险贡献度分析结果，帮助非算法角色快速理解影响预测结果的关键因素。

## 数据与任务定义

- 数据来源：公开心脏病数据集整理版，目标字段为 `Heart Disease`
- 任务类型：二分类预测，识别样本是否具有心脏病风险
- 特征类型：年龄、血压、胆固醇、最大心率、运动诱发心绞痛、ST 段相关指标等结构化变量
- 数据处理：统一字段命名、数值化转换、缺失值填补、分层抽样划分训练集与验证集

## 方法流程

```text
Raw Data
  -> Column Standardization
  -> Numeric Conversion / Missing Value Handling
  -> Stratified Train-Test Split
  -> Logistic Regression Training
  -> Metrics Evaluation
  -> Feature Importance Explanation
```

## 关键结果解读

- 模型在验证集上实现了较稳定的分类表现，兼顾准确率与召回率，适合用作风险筛查原型。
- 影响风险升高最明显的因素包括：荧光透视下主要血管数量、Thallium 检查结果、胸痛类型、ST 段压低程度。
- 与风险下降相关的因素主要包括：最大心率表现相对更好，以及部分基础指标处于更低风险区间。
- 项目输出同时保留了结构化指标文件与可读文字说明，便于用于汇报、展示或后续二次开发。

## 项目结构

```text
Heart_Disease_Prediction/
├─ data/
│  └─ raw/
├─ models/
├─ outputs/
├─ src/
│  ├─ data.py
│  ├─ model.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ explain.py
└─ requirements.txt
```

## 交付物

- `models/logistic_model.pkl`：训练完成的模型文件
- `outputs/train_metrics.json`：训练阶段验证指标
- `outputs/evaluation.json`：评估结果汇总
- `outputs/feature_importance.csv`：特征重要性排序
- `outputs/explanation.txt`：面向阅读的模型解释文本

## 快速运行

```powershell
py src\train.py
py src\evaluate.py
py src\explain.py
```

## 技术栈

- Python
- pandas
- NumPy
- 自实现 Logistic Regression

## 项目说明

该项目定位为风险识别与建模流程原型，用于展示数据处理、模型训练、效果评估和结果解释能力，不替代临床诊断。
