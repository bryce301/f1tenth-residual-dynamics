# v3 训练说明

- 训练代码提交：`6bd4d14 Training scheme v3: measured servo gain, 15 recordings incl. sustained drifts, drift held out`
- 最终校验：**PASS**，16 项全部通过。
- 默认设置完整运行：是。使用发布的 15 份数据，所有训练脚本和配置保持原样，运行 `python run_all.py`。
- 硬件：AMD Ryzen 9 8945HX，训练在 CPU 上运行。Python 3.12.16，PyTorch 2.9.1+cu128，复用了已有的 SuperMimic Python 环境。
- 总耗时：约 27 分 20 秒，2026-11-25 23:41:51 至 2026-11-26 00:09:11（美国东部时间，以训练日志创建和最后写入时间估算）。
- 正式运行的日志中未见警告或报错。64×64 残差网络触发了默认的连续 60 轮无改善早停。开跑前校验仅因尚未生成拟合、评估和导出文件报告缺失，正式运行结束后全部通过。

训练开始时的数据切分输出原文：

```text
train: 165 runs, 727 s, of which sideslip > 10 deg: 120.1 s
held out: 57 runs, 271 s, of which sideslip > 10 deg: 83.1 s
```
