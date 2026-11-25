#!/usr/bin/env python3
"""Step 1: identify the physical parameters of the classical model from recorded driving.

    python fit_classical.py            # reads data/*.npz and config.yaml, writes out/fitted.json
"""
import argparse
import torch
from core import (HERE, Fiala, FullModel, describe, load_config, load_sessions, rmse, rollout_error, save_json,
                  split_runs, windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--data', nargs='*')
    args = parser.parse_args()
    config = load_config(args.config)
    runs, dt = load_sessions(config, args.data)
    train, val = split_runs(runs, config)
    describe(train, dt, 'train'); describe(val, dt, 'held out')
    horizon, stride = round(config['rollout']['horizon_s'] / dt), round(config['rollout']['stride_s'] / dt)
    train_batch, val_batch = windows(train, horizon, stride), windows(val, horizon, stride)

    classical = Fiala(config); model = FullModel(classical)
    initial = {k: float(v) for k, v in classical.parameters_dict().items()}
    with torch.no_grad():
        before = rmse(rollout_error(model, classical, val_batch, dt, config)[1])
    optimizer = torch.optim.Adam(classical.parameters(), lr=config['classical']['learning_rate'])
    for epoch in range(config['classical']['epochs']):
        optimizer.zero_grad()
        loss, _ = rollout_error(model, classical, train_batch, dt, config)
        loss.backward(); optimizer.step()
        if epoch % 50 == 0 or epoch == config['classical']['epochs'] - 1:
            with torch.no_grad():
                val_loss, _ = rollout_error(model, classical, val_batch, dt, config)
            print(f'epoch {epoch:4d}  train {loss.item():.4f}  held out {val_loss.item():.4f}', flush=True)

    fitted = {k: float(v) for k, v in classical.parameters_dict().items()}
    with torch.no_grad():
        after = rmse(rollout_error(model, classical, val_batch, dt, config)[1])
    print(f'\n{"parameter":34s} {"initial":>9s} {"fitted":>9s}   bounds')
    for name, (_, low, high) in config['identified'].items():
        edge = '  <-- at a bound: not determined by the data' if min(fitted[name] - low, high - fitted[name]) < 0.02 * (high - low) else ''
        print(f'{name:34s} {initial[name]:9.4f} {fitted[name]:9.4f}   [{low}, {high}]{edge}')
    print(f'\nheld-out {config["rollout"]["horizon_s"]} s rollout RMSE   yaw rate   speed  sideslip  wheel speed')
    print('  initial parameters          ' + '  '.join(f'{v:7.3f}' for v in before))
    print('  fitted parameters           ' + '  '.join(f'{v:7.3f}' for v in after))
    (HERE / 'out').mkdir(exist_ok=True)
    save_json(HERE / 'out' / 'fitted.json', fitted)
    print('wrote out/fitted.json')


if __name__ == '__main__':
    main()
