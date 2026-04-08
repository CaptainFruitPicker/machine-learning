# Heart Disease Prediction

这个项目已经放在本地目录 `C:\Users\钱程远\Desktop\学习\项目1\Heart_Disease_Prediction` 中，不依赖 GitHub，也没有初始化 Git 仓库。

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

## 你的数据集

项目默认直接读取你桌面的原始 Excel 文件：

```text
C:\Users\钱程远\Desktop\Heart_Disease_Prediction.xlsx
```

也就是说，你现在不需要上传 GitHub，也不需要先把数据手动复制进仓库才能运行。

## 运行方式

在 PowerShell 中进入项目目录：

```powershell
cd C:\Users\钱程远\Desktop\学习\项目1\Heart_Disease_Prediction
```

训练模型：

```powershell
py src\train.py
```

评估模型：

```powershell
py src\evaluate.py
```

查看特征解释：

```powershell
py src\explain.py
```

## 输出结果

- `models/logistic_model.pkl`：训练好的模型
- `outputs/train_metrics.json`：训练阶段验证集指标
- `outputs/evaluation.json`：评估指标
- `outputs/feature_importance.csv`：特征重要性
- `outputs/explanation.txt`：文字解释结果

## 说明

- 这个版本使用 `pandas + numpy` 实现二分类逻辑回归，不依赖 `scikit-learn`
- Excel、CSV、XLSX 文件都可以读取
- 目标列默认识别为 `Heart Disease`
