#!/usr/bin/env python3
"""Step 3: compare the classical model with and without the residual on held-out runs.

    python evaluate.py                 # prints tables, writes out/evaluation.png and out/evaluation.json
"""
import argparse
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from core import (HERE, STATE_NAMES, Fiala, FullModel, Residual, load_config, load_sessions, rollout, save_json,
                  split_runs, to_model_controls, windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--data', nargs='*')
    args = parser.parse_args()
    config = load_config(args.config)
    runs, dt = load_sessions(config, args.data)
    _, val = split_runs(runs, config)
    classical = Fiala(config, json.loads((HERE / 'out' / 'fitted.json').read_text()))
    models = {'classical': FullModel(classical)}
    if (HERE / 'out' / 'residual.pt').exists():
        residual = Residual(config); residual.load_state_dict(torch.load(HERE / 'out' / 'residual.pt'))
        models['classical + residual'] = FullModel(classical, residual)
    substeps, result = int(config['rollout']['substeps']), {}
    print('held-out rollout RMSE (grip: |sideslip| <= 10 deg at the start of the rollout; slide: above)')
    with torch.no_grad():
        for seconds in (0.5, 1.0, 2.0):
            x0, u, x = windows(val, round(seconds / dt), round(0.1 / dt))
            if not len(x0):
                print(f'  {seconds} s: no held-out run is long enough'); continue
            slide = x0[:, 2].abs() > math.radians(10)
            print(f'  horizon {seconds} s ({len(x0)} rollouts, {int(slide.sum())} start while sliding)')
            print(f'    {"":22s} {"":6s}' + ''.join(f'{n:>13s}' for n in STATE_NAMES))
            for name, model in models.items():
                error = rollout(model, x0, to_model_controls(classical, u), dt, substeps) - x
                for label, mask in (('all', torch.ones_like(slide)), ('grip', ~slide), ('slide', slide)):
                    if mask.any():
                        values = error[mask].pow(2).mean((0, 1)).sqrt().tolist()
                        result[f'{seconds}s/{name}/{label}'] = values
                        print(f'    {name:22s} {label:6s}' + ''.join(f'{v:13.3f}' for v in values))
        # Open-loop prediction along the longest held-out run, restarted every 2 s.
        run = max(val, key=lambda r: len(r['state']))
        state, control = torch.from_numpy(run['state']), torch.from_numpy(run['control'])
        step = round(2.0 / dt)
        figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        time = torch.arange(len(state)) * dt
        for k, axis in enumerate(axes):
            axis.plot(time, state[:, k], 'k', lw=2, label='recorded'); axis.set_ylabel(STATE_NAMES[k]); axis.grid()
        for name, model in models.items():
            colour = 'tab:blue' if name == 'classical' else 'tab:red'
            for start in range(0, len(state) - 2, step):
                stop = min(start + step, len(state) - 1)
                u = to_model_controls(classical, control[None, start:stop])
                pred = rollout(model, state[None, start], u, dt, substeps)[0]
                for k, axis in enumerate(axes):
                    axis.plot(time[start + 1:stop + 1], pred[:, k], colour, lw=1, label=name if start == 0 else None)
        axes[0].legend(); axes[0].set_title(f'{run["file"]}: open-loop prediction restarted every 2 s'); axes[-1].set_xlabel('s')
        figure.tight_layout(); figure.savefig(HERE / 'out' / 'evaluation.png', dpi=80)
    save_json(HERE / 'out' / 'evaluation.json', result)
    print('wrote out/evaluation.png and out/evaluation.json')


if __name__ == '__main__':
    main()
