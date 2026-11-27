# P2 操作指南

离线运行，不访问网络，也不调用 OCR。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install Pillow
python -m tkinter
```

部分 Linux 发行版需要安装 python3-tk。

确认归档 SHA-256 为：

90cb341f02c7db2658d5a25f3a21da431eb7108b3e56fb24da731ac06cb97494

启动标注：

```bash
python tools/label_blind_test.py \
  --archive /secure/path/captcha_train_1100.tgz \
  --manifest datasets/private/blind-test-v1.json \
  --progress datasets/private/blind-test-progress.jsonl
```

输入四位数字后按 Enter；按 S 或点击跳过按钮记录无法辨认的图片。退出后进度保留，重新运行相同命令即可续标。界面不显示模型预测。

完成 200 个有效标签后，按 p2-dataset-protocol.md 执行冻结。不要手工修改已冻结 CSV。
