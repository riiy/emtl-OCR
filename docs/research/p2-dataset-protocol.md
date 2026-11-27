# P2 数据集建设与盲测标注协议

**状态：** 清单生成和标注工具已实现；200 张人工标注尚未完成。  
**源归档 SHA-256：** 90cb341f02c7db2658d5a25f3a21da431eb7108b3e56fb24da731ac06cb97494

## 训练/验证划分

固定种子 20261127，按来源分层，对各来源按文件名排序后独立洗牌，再按 80/20 划分：

| 来源 | 训练 | 验证 | 合计 |
|---|---:|---:|---:|
| ACCEPTED | 800 | 200 | 1,000 |
| REJECTED | 24 | 6 | 30 |
| SKIPPED_INVALID_FORMAT | 56 | 14 | 70 |
| **合计** | **880** | **220** | **1,100** |

## 新盲测抽样

287 张未被训练清单引用的图片作为候选集。固定种子 20261127，候选路径排序后使用 Python random.Random(seed).shuffle()；前 200 张为首选盲测集，剩余 87 张按清单顺序备用。记录源归档及每张图片 SHA-256。

## 生成清单

```bash
python tools/prepare_dataset.py \
  --archive /path/to/captcha_train_1100.tgz \
  --blind-output datasets/private/blind-test-v1.json \
  --split-output datasets/private/train-validation-v1.json \
  --seed 20261127
```

工具校验 1,100 条标签、1,387 张图片、287 张候选图片、图片 SHA-256、101×36 RGB JPEG 格式及归档 SHA-256。

## 人工标注

依赖 Python 3.10+ 和 Pillow；标注界面使用本地 HTTP 服务与浏览器，不依赖 Tkinter。运行：

```bash
python tools/label_blind_test.py \
  --archive /path/to/captcha_train_1100.tgz \
  --manifest datasets/private/blind-test-v1.json \
  --progress datasets/private/blind-test-progress.jsonl
```

维护者亲自逐张标注；服务仅绑定 127.0.0.1:8765，浏览器界面只显示原图，不显示或请求任何 OCR 预测。仅接受恰好四位 ASCII 数字字符串，保留前导零（例如 0042）。支持上一张、下一张、提交、跳过、修改已有标签及 skipped 重新标注。浏览不写入事件；只有提交/跳过才追加 JSONL，并执行 flush/fsync。JSONL 是追加式审计日志，同一图片可有多条事件，最终状态以最后一条事件为准。达到 200 条最终 labeled 状态后停止新增标注，但仍允许修订；数量低于 200 时恢复补标。启动时自动定位第一张从未产生事件的图片，跳过已处理（包括 skipped）的图片；可手动返回修订。关闭服务后可续标。不得根据模型预测修改标签。

## 标签冻结

完成 200 条有效标签后运行：

```bash
python tools/finalize_blind_test.py \
  --manifest datasets/private/blind-test-v1.json \
  --progress datasets/private/blind-test-progress.jsonl \
  --labels datasets/private/blind-test-labels.csv \
  --freeze datasets/private/blind-test-freeze.json
```

冻结工具按每张图片最后一次事件归约最终状态，允许修订历史；验证最终有效标签恰好 200 条、图片 ID 唯一、路径和图片哈希、四位字符串格式，并生成 manifest、完整事件日志和标签 CSV 的 SHA-256。冻结成功前不得运行 Benchmark。

## 发布边界

本 PR 不上传图片、人工标签、标注进度或模型权重，不运行 OCR，不产生 Benchmark 结果。87 张备用图片不参与训练/验证。最初 20 张远程来源图片继续排除。字体权利审查仍是数据和模型公开前的门槛。
