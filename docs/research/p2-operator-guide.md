# P2 操作指南

标注工具使用本地 Web 服务与浏览器，不依赖 Tkinter。服务只绑定 `127.0.0.1`，不向局域网开放，不加载 CDN、外部脚本或 OCR 模型。图片从原始 TGZ 读取，归档保持只读。

## 环境

需要 Python 3.10+ 和 Pillow：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install Pillow
```

确认归档 SHA-256：

`90cb341f02c7db2658d5a25f3a21da431eb7108b3e56fb24da731ac06cb97494`

## 启动

```bash
python tools/label_blind_test.py \
  --archive /secure/path/captcha_train_1100.tgz \
  --manifest datasets/private/blind-test-v1.json \
  --progress datasets/private/blind-test-progress.jsonl
```

浏览器自动打开 `http://127.0.0.1:8765/`。如果没有自动打开，可复制终端显示的本地地址。停止服务按 Ctrl+C。

## 标注规则

- 页面只显示原始图片，不运行、不显示 ddddOCR、Tiny-CNN 或其他 OCR 预测。
- 输入恰好四位 ASCII 数字，保留前导零，例如 `0042`。
- Enter 或“提交标签”保存 labeled 事件；“跳过”保存 skipped 事件。
- 上一张/下一张只浏览，不会自动提交。离开有未提交输入的图片时，页面要求确认。
- 已标注图片会显示当前有效标签，允许修改；已跳过图片可以重新标注。
- 所有提交/跳过/修订都追加到 JSONL，不覆盖历史。最终状态由每张图片最后一条事件决定。
- 有效标注达到 200 后停止新增标注，但仍允许浏览和修订。若修订使有效数量降到 199，自动恢复补标。
- 断点续标启动时定位第一张从未产生事件的图片；已跳过图片视为已处理，但可手动返回重新标注。

## 冻结

完成恰好 200 条最终有效标签后：

```bash
python tools/finalize_blind_test.py \
  --manifest datasets/private/blind-test-v1.json \
  --progress datasets/private/blind-test-progress.jsonl \
  --labels datasets/private/blind-test-labels.csv \
  --freeze datasets/private/blind-test-freeze.json
```

冻结工具会归约完整事件历史，以最后一次事件为准，输出唯一图片 ID 的 CSV，并记录 manifest、JSONL 和 CSV 的 SHA-256。冻结成功前不得运行 Benchmark。

不要将原始图片、人工标签、标注进度或冻结产物提交到公开仓库。
