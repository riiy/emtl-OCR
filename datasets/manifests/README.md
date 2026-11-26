# 数据集元数据清单

本目录只存放不含图片字节和人工标签的元数据清单。

- dataset-audit.json：源归档哈希、样本计数和校验状态。
- train-validation-manifest.json.gz：固定 880/220 分层划分；按 source 保存 sample_id 成员列表，不含标签。
- blind-test-candidates.json.gz：200 张首选候选及 87 张备用候选，记录归档路径和 SHA-256，不含人工标签。

原始归档、提取图片、标注进度、冻结标签和冻结图片只保存在本地 .local-data/，不得提交到 Git，直至完成数据及第三方字体权利审查。

清单由 tools/prepare_dataset.py --gzip 根据 captcha_train_1100.tgz 确定性生成。随机种子和源归档 SHA-256 写在各 JSON 文件中。
