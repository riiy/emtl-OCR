# P2 操作指南

标注工具运行于本机浏览器，只绑定 **127.0.0.1**，不访问外网、不调用 OCR，也不上传图片。无需 Tkinter。标注服务仅使用 Python 标准库。

确认归档 SHA-256 为：

**90cb341f02c7db2658d5a25f3a21da431eb7108b3e56fb24da731ac06cb97494**

启动标注：

~~~bash
python tools/label_blind_test.py \
  --archive /secure/path/captcha_train_1100.tgz \
  --manifest datasets/private/blind-test-v1.json \
  --progress datasets/private/blind-test-progress.jsonl
~~~

默认浏览器地址：**http://127.0.0.1:8765/**。如需在终端/无桌面环境启动服务，可增加 **--no-browser**；端口可通过 **--port** 修改。

输入恰好四位 ASCII 数字后按 Enter；前导零必须保留。点击“跳过”记录无法辨认的图片。上一张、下一张只浏览，不自动提交；如果输入尚未提交，离开当前图片前会提示确认。已标注图片可修改，已跳过图片可重新标注。所有操作追加到 JSONL 并立即 **fsync**，不会覆盖旧事件。

重启后自动定位到第一张从未产生标注事件的图片。已跳过图片视为已处理，不会自动重复出现，但可通过导航手动返回。达到 200 条有效标签后仍可浏览和修改已有标签；如果将已有标签改为跳过，数量下降后会重新开放补充标注。

完成 200 个唯一图片的最终有效标签后，按 **p2-dataset-protocol.md** 执行冻结。不要手工修改已冻结 CSV。原始 TGZ 和公开仓库均不会保存私人标注进度。
