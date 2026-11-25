#!/usr/bin/env python3
"""Step 3: compare the classical model with every trained residual on held-out runs.

    python evaluate.py                 # prints tables, writes out/evaluation.json and out/evaluation.png
"""
import argparse
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from core import (HERE, STATE_NAMES, Fiala, Residual, horizon_samples, load_config, load_sessions, rollout,
                  save_json, split_runs, to_model_controls, windows)


def load_models(config):
    """Classical model and {name: residual or None} for every out/residual_*.pt."""
    classical = Fiala(config, json.loads((HERE / 'out' / 'fitted.json').read_text()))
    models = {'classical': None}
    for path in sorted((HERE / 'out').glob('residual_*.pt')):
        hidden = [int(n) for n in path.stem.split('_')[1].split('x')]
        residual = Residual(config, hidden); residual.load_state_dict(torch.load(path))
        models[f'classical + residual {path.stem.split("_")[1]}'] = residual
    return classical, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--data', nargs='*')
    args = parser.parse_args()
    config = load_config(args.config)
    runs, dt = load_sessions(config, args.data)
    _, val = split_runs(runs, config)
    classical, models = load_models(config)
    result = {}
    print('held-out rollout RMSE (grip: |sideslip| <= 10 deg at the start of the rollout; slide: above)')
    with torch.no_grad():
        for seconds in (0.5, 1.0, 2.0):
            x0, u, x = windows(val, horizon_samples(config, dt, seconds), round(0.1 / dt))
            if not len(x0):
                print(f'  {seconds} s: no held-out run is long enough'); continue
            slide = x0[:, 2].abs() > math.radians(10)
            print(f'  horizon {seconds} s ({len(x0)} rollouts, {int(slide.sum())} start while sliding)')
            print(f'    {"":28s} {"":6s}' + ''.join(f'{n:>13s}' for n in STATE_NAMES))
            for name, residual in models.items():
                error = rollout(classical, residual, x0, to_model_controls(classical, u), dt, config) - x
                for label, mask in (('all', torch.ones_like(slide)), ('grip', ~slide), ('slide', slide)):
                    if mask.any():
                        values = error[mask].pow(2).mean((0, 1)).sqrt().tolist()
                        result[f'{seconds}s/{name}/{label}'] = dict(rmse=values, rollouts=int(mask.sum()))
                        print(f'    {name:28s} {label:6s}' + ''.join(f'{v:13.3f}' for v in values))
        # Open-loop prediction along the longest held-out run, restarted every 2 s.
        run = max(val, key=lambda r: len(r['state']))
        state, control = torch.from_numpy(run['state']), torch.from_numpy(run['control'])
        step = horizon_samples(config, dt, 2.0)
        hold = round(config['rollout']['residual_hold_s'] / dt)
        figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        time = torch.arange(len(state)) * dt
        for k, axis in enumerate(axes):
            axis.plot(time, state[:, k], 'k', lw=2, label='recorded'); axis.set_ylabel(STATE_NAMES[k]); axis.grid()
        for (name, residual), colour in zip(models.items(), ('tab:blue', 'tab:red', 'tab:green', 'tab:orange')):
            for start in range(0, len(state) - hold, step):
                stop = start + (min(step, len(state) - 1 - start) // hold) * hold
                if stop == start:
                    break
                pred = rollout(classical, residual, state[None, start],
                               to_model_controls(classical, control[None, start:stop]), dt, config)[0]
                for k, axis in enumerate(axes):
                    axis.plot(time[start + 1:stop + 1], pred[:, k], colour, lw=1, label=name if start == 0 else None)
        axes[0].legend(); axes[0].set_title(f'{run["file"]}: open-loop prediction restarted every 2 s'); axes[-1].set_xlabel('s')
        figure.tight_layout(); figure.savefig(HERE / 'out' / 'evaluation.png', dpi=80)
    save_json(HERE / 'out' / 'evaluation.json', result)
    print('wrote out/evaluation.png and out/evaluation.json')


if __name__ == '__main__':
    main()
