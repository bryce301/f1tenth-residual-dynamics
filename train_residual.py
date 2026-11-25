#!/usr/bin/env python3
"""Step 2: train the residual network on top of the fitted classical model.

    python train_residual.py           # needs out/fitted.json, writes out/residual.pt
"""
import argparse
import copy
import json
import torch
from core import (HERE, Fiala, FullModel, Residual, describe, load_config, load_sessions, rmse, rollout_error,
                  split_runs, to_model_controls, windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--data', nargs='*')
    parser.add_argument('--fitted', default=str(HERE / 'out' / 'fitted.json'))
    args = parser.parse_args()
    config = load_config(args.config); c = config['residual']
    torch.manual_seed(0)
    runs, dt = load_sessions(config, args.data)
    train, val = split_runs(runs, config)
    describe(train, dt, 'train'); describe(val, dt, 'held out')
    horizon, stride = round(config['rollout']['horizon_s'] / dt), round(config['rollout']['stride_s'] / dt)
    train_batch, val_batch = windows(train, horizon, stride), windows(val, horizon, stride)

    classical = Fiala(config, json.loads(open(args.fitted).read())).requires_grad_(False)
    residual = Residual(config)
    x0, u, x = train_batch
    with torch.no_grad():
        residual.set_normalization(torch.cat([x.reshape(-1, 4), to_model_controls(classical, u).reshape(-1, 2)], -1))
        baseline = rollout_error(FullModel(classical), classical, val_batch, dt, config)
    model = FullModel(classical, residual)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    best, best_state, stale = float('inf'), None, 0
    print(f'held-out error of the classical model alone: {baseline[0].item():.4f}')
    for epoch in range(c['epochs']):
        order = torch.randperm(len(x0))
        total = 0.0
        for i in range(0, len(order), c['batch_size']):
            index = order[i:i + c['batch_size']]
            optimizer.zero_grad()
            loss, _ = rollout_error(model, classical, (x0[index], u[index], x[index]), dt, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0)
            optimizer.step(); total += loss.item() * len(index)
        with torch.no_grad():
            val_loss = rollout_error(model, classical, val_batch, dt, config)[0].item()
        if val_loss < best - 1e-5:
            best, best_state, stale = val_loss, copy.deepcopy(residual.state_dict()), 0
        else:
            stale += 1
        if epoch % 10 == 0 or stale == 0:
            print(f'epoch {epoch:4d}  train {total / len(order):.4f}  held out {val_loss:.4f}{"  *" if stale == 0 else ""}', flush=True)
        if stale >= c['patience']:
            print(f'stopped: no held-out improvement for {c["patience"]} epochs')
            break
    residual.load_state_dict(best_state)
    with torch.no_grad():
        final = rollout_error(model, classical, val_batch, dt, config)
    print(f'\nheld-out {config["rollout"]["horizon_s"]} s rollout RMSE   yaw rate   speed  sideslip  wheel speed')
    print('  classical model             ' + '  '.join(f'{v:7.3f}' for v in rmse(baseline[1])))
    print('  classical + residual        ' + '  '.join(f'{v:7.3f}' for v in rmse(final[1])))
    change = 100 * (final[0].item() / baseline[0].item() - 1)
    print(f'  scaled error {baseline[0].item():.4f} -> {final[0].item():.4f} ({change:+.0f} %)')
    if change > -5:
        print('  The residual does not clearly help on held-out data: keep the classical model, or record more data.')
    torch.save(residual.state_dict(), HERE / 'out' / 'residual.pt')
    print('wrote out/residual.pt')


if __name__ == '__main__':
    main()
