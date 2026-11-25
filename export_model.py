#!/usr/bin/env python3
"""Step 4: write the trained model as one JSON file for the controller.

    python export_model.py                       # residual of the size in config.yaml
    python export_model.py --hidden 64 64        # another trained size
    python export_model.py --classical-only

out/dynamics_model.json holds the physical parameters, the residual weights and reference
input/output vectors with which the controller's own implementation is checked.
"""
import argparse
import json
import numpy as np
import torch
from core import HERE, Fiala, Residual, load_config, save_json


def numpy_residual(model, inputs):
    """The residual recomputed from the exported numbers only (independent of PyTorch)."""
    r = model['residual']
    y = (inputs - np.array(r['input_mean'])) / np.array(r['input_std'])
    for i, layer in enumerate(r['layers']):
        y = y @ np.array(layer['weight']).T + np.array(layer['bias'])
        if i < len(r['layers']) - 1:
            y = np.tanh(y)
    lo, hi = r['fade_speed_mps']
    gate = np.clip((np.abs(inputs[:, 1]) - lo) / (hi - lo), 0, 1)
    return np.array(r['limit']) * np.tanh(y) * gate[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--classical-only', action='store_true')
    parser.add_argument('--hidden', type=int, nargs='+')
    args = parser.parse_args()
    config = load_config(args.config)
    classical = Fiala(config, json.loads((HERE / 'out' / 'fitted.json').read_text())).requires_grad_(False)
    model = {
        'format': 'f1tenth_dynamics_v1',
        'interface': {
            'input': ['yaw_rate_radps', 'speed_mps', 'sideslip_rad', 'driven_wheel_speed_mps', 'steering_rad',
                      'motor_current_a'],
            'output': ['yaw_acceleration_radps2', 'speed_acceleration_mps2', 'sideslip_rate_radps',
                       'driven_wheel_acceleration_mps2'],
            'state_reference': 'centre of gravity',
            'current': 'positive drives, negative brakes (opposes rotation, zero at standstill)'},
        'integration': {'step_s': config['rollout']['step_s'], 'stable_step_limit_s': classical.step_limit(),
                        'residual_hold_s': config['rollout']['residual_hold_s']},
        'vehicle': config['vehicle'],
        'parameters': {k: v.item() for k, v in classical.parameters_dict().items()},
        'steering_delay_s': config['data']['steering_delay_s'],
        'residual': None}
    residual = None
    if not args.classical_only:
        residual = Residual(config, args.hidden)
        residual.load_state_dict(torch.load(HERE / 'out' / f'residual_{"x".join(map(str, residual.hidden))}.pt'))
        linear = [m for m in residual.net if isinstance(m, torch.nn.Linear)]
        model['residual'] = {
            'hidden': residual.hidden, 'activation': 'tanh', 'output': 'limit * tanh(net) * speed gate',
            'input_mean': residual.mean.tolist(), 'input_std': residual.std.tolist(),
            'limit': residual.limit.tolist(), 'fade_speed_mps': residual.fade.tolist(),
            'layers': [{'weight': m.weight.tolist(), 'bias': m.bias.tolist()} for m in linear]}
    samples = torch.tensor([[0.5, 2.0, 0.05, 2.1, 0.10, 10.0], [-1.5, 3.5, -0.40, 4.5, -0.20, 40.0],
                            [0.0, 1.2, 0.00, 1.2, 0.00, 0.0], [2.0, 3.0, -0.30, 2.0, 0.30, -20.0],
                            [0.3, 0.5, 0.02, 0.5, 0.05, 15.0], [-0.8, 4.5, 0.25, 5.5, -0.10, 55.0]])
    with torch.no_grad():
        outputs = classical(samples)
        correction = residual(samples) if residual is not None else torch.zeros_like(outputs)
    model['test_vectors'] = [dict(input=i, classical=c, residual=r)
                             for i, c, r in zip(samples.tolist(), outputs.tolist(), correction.tolist())]
    save_json(HERE / 'out' / 'dynamics_model.json', model)
    print(f'wrote out/dynamics_model.json ({"classical only" if residual is None else "classical + residual " + "x".join(map(str, residual.hidden))})')
    if residual is not None:
        reread = json.loads((HERE / 'out' / 'dynamics_model.json').read_text())
        difference = float(np.abs(numpy_residual(reread, samples.numpy().astype(float)) - correction.numpy()).max())
        print(f'export self-check, residual recomputed from the file vs PyTorch: maximum difference {difference:.2e}')
        save_json(HERE / 'out' / 'export_check.json', dict(max_difference=difference))


if __name__ == '__main__':
    main()
