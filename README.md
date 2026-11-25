# f1tenth-residual-dynamics

Residual vehicle dynamics model (Fiala bicycle model + small MLP, motor-current control) for an F1TENTH MPPI controller.

> ## ★★★★★ 开始之前必须完整阅读 [训练说明.md](训练说明.md)
>
> 当前为 v2：模型的计算方式已修改，旧版本的训练结果作废。
> 有一组设置不能改；训练完成后必须运行 `python verify.py` 并交回反馈文件。全部细节在 [训练说明.md](训练说明.md)。

```bash
python run_all.py      # fit, train, evaluate, export, verify; then send back the out/ folder
```

## v2 训练结果（2026-11-25）

训练基于提交 `e966687`，使用仓库发布的 9 份数据及默认配置，在 CPU 上完成。留出集指标为 0.5 秒开环预测的归一化误差，越低越好。

| 模型 | 实际轮数 | 最佳轮次 | 留出集误差 | 相对纯经典模型 | 最佳权重 |
|---|---:|---:|---:|---:|---|
| Fiala 经典模型 | 300 | — | 0.1003 | 基准 | — |
| 经典模型 + 32×32 残差 | 600 | 575 | **0.0477** | 降低约 52% | [32×32 权重](results/2026-11-25/residual_32x32_epoch_0575.pt) |
| 经典模型 + 64×64 残差 | 373，早停 | 313 | 0.0488 | 降低约 51% | [64×64 权重](results/2026-11-25/residual_64x64_epoch_0313.pt) |

轮次从 1 开始计数，对应日志中的 `epoch 574` 和 `epoch 312`。两种网络在 0.5、1、2 秒的滑移起步样本中，横摆角速度、车速、侧偏角和轮速的 RMSE 均低于纯经典模型。32×32 的留出集归一化误差最低，本次控制器 JSON 导出使用该网络。

拟合模型的积分稳定步长上限为 4.70 ms，实际使用 2.0 ms。导出数值自检最大差值为 1.14 × 10⁻⁶，通过 1 × 10⁻⁴ 阈值。官方校验总体为 **FAIL**，唯一失败项是脚本哈希。为每 100 轮保存 checkpoint，本次在 `fit_classical.py` 和 `train_residual.py` 增加了保存逻辑，并通过 `run_all.py` 传递间隔参数。训练数据和受保护配置均通过校验。
