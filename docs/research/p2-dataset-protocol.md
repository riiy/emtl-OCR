# P2 数据集建设与盲测标注协议

**状态：** 数据清单生成和本地 Web 标注工具已实现；200 张人工标注尚未完成。  
**源归档 SHA-256：** **90cb341f02c7db2658d5a25f3a21da431eb7108b3e56fb24da731ac06cb97494**

## 训练/验证划分

固定种子 20261127，按来源分层，对各来源按文件名排序后独立洗牌，再按 80/20 划分：

| 来源 | 训练 | 验证 | 合计 |
|---|---:|---:|---:|
| ACCEPTED | 800 | 200 | 1,000 |
| REJECTED | 24 | 6 | 30 |
| SKIPPED_INVALID_FORMAT | 56 | 14 | 70 |
| **合计** | **880** | **220** | **1,100** |

## 新盲测抽样

287 张未被训练清单引用的图片作为候选集。固定种子 20261127，候选路径排序后使用 Python **random.Random(seed).shuffle()**；前 200 张为首选盲测集，剩余 87 张按清单顺序备用。记录源归档及每张图片 SHA-256。

## 生成清单

~~~bash
python tools/prepare_dataset.py \
  --archive /path/to/captcha_train_1100.tgz \
  --blind-output datasets/private/blind-test-v1.json \
  --split-output datasets/private/train-validation-v1.json \
  --seed 20261127
~~~

工具校验 1,100 条标签、1,387 张图片、287 张候选图片、图片 SHA-256、101×36 RGB JPEG 格式及归档 SHA-256。

## 本地 Web 人工标注

标注工具使用 Python 标准库 HTTP 服务和浏览器，不依赖 Tkinter、Flask 或 FastAPI。Pillow 仍用于数据准备阶段；标注服务本身直接提供原始 JPEG 字节，不调用图像模型。

~~~bash
python tools/label_blind_test.py \
  --archive /path/to/captcha_train_1100.tgz \
  --manifest datasets/private/blind-test-v1.json \
  --progress datasets/private/blind-test-progress.jsonl
~~~

服务只绑定 **127.0.0.1:8765**，默认打开 **http://127.0.0.1:8765/**。页面、脚本、样式和图片均由本地服务提供；不使用 CDN、外部字体、第三方分析或外部服务。原始 TGZ 只读，标注事件保存在本地 JSONL。

页面仅显示图片，不显示或请求任何 OCR 预测。输入必须是恰好四位 ASCII 数字字符串，保留前导零（例如 **0042**）。提交、跳过、上一张、下一张分别处理；浏览不会写入事件。离开含未提交输入的图片时会要求确认。

每次提交或跳过均追加一条事件并 **fsync**。修改标签、将已标注图片改为跳过、或将跳过图片重新标注，均追加新事件，不覆盖历史。每张图片以最后一次事件决定当前状态，冻结时每个图片 ID 只输出一行。达到 200 条有效标签后，不能新增第 201 条；仍可浏览和修改已有标签。释放名额后恢复补充标注。

断点续标优先定位到从未产生事件的第一张图片；已跳过图片视为已处理，不自动重复出现，但可手动返回重新标注。进度文件不完整、JSON 损坏或图片哈希不匹配时，工具会停止并报告错误，不会静默忽略。

## 标签冻结

完成恰好 200 条唯一图片的最终有效标签后运行：

~~~bash
python tools/finalize_blind_test.py \
  --manifest datasets/private/blind-test-v1.json \
  --progress datasets/private/blind-test-progress.jsonl \
  --labels datasets/private/blind-test-labels.csv \
  --freeze datasets/private/blind-test-freeze.json
~~~

冻结工具验证事件历史、最终状态、标签数量、唯一图片 ID、路径和图片哈希、四位字符串格式，并生成 manifest、进度文件和标签 CSV 的 SHA-256。冻结成功前不得运行 Benchmark。

## 发布边界

本 PR 不上传图片、人工标签、标注进度或模型权重，不运行 OCR，不产生 Benchmark 结果。87 张备用图片不参与训练/验证。最初 20 张远程来源图片继续排除。字体权利审查仍是数据和模型公开前的门槛。
