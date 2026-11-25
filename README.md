# f1tenth-residual-dynamics

Residual vehicle dynamics model (Fiala bicycle model + small MLP, motor-current control) for an F1TENTH MPPI controller.

> ## ★★★★★ 开始之前必须完整阅读 [训练说明.md](训练说明.md)
>
> 当前为 v2：模型的计算方式已修改，旧版本的训练结果作废。
> 有一组设置不能改；训练完成后必须运行 `python verify.py` 并交回反馈文件。全部细节在 [训练说明.md](训练说明.md)。

```bash
python run_all.py      # fit, train, evaluate, export, verify; then send back the out/ folder
```
