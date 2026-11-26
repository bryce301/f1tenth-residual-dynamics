# f1tenth-residual-dynamics

Residual vehicle dynamics model (Fiala bicycle model + small MLP, motor-current control) for an F1TENTH MPPI controller.

> ## ★★★★★ 开始之前必须完整阅读 [训练说明.md](训练说明.md)
>
> 当前为 v3：舵机标定改为实测值、训练数据改为 15 个文件（含持续漂移）。v2 的结果作废，必须用全新副本重新训练。
> 数据、脚本和受保护设置都不能改；训练完成后必须按说明第 7 节交回结果。全部细节在 [训练说明.md](训练说明.md)。

```bash
python run_all.py      # fit, train, evaluate, export, verify; then push out/ to results/v3 on a branch
```
