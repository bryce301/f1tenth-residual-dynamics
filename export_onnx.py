#!/usr/bin/env python3
"""Step 4: export classical model (+ residual when trained) as one ONNX graph for the MPPI TensorRT interface.

    python export_onnx.py              # writes out/dynamics.onnx and out/dynamics.onnx.json
    python export_onnx.py --classical-only

On the vehicle:  ros2 run xx_mppi xxcar_build_trt_engine dynamics.onnx dynamics.plan <batch> <batch>
"""
import argparse
import json
import torch
from core import HERE, Fiala, FullModel, Residual, load_config, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config'); parser.add_argument('--classical-only', action='store_true')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    config = load_config(args.config)
    classical = Fiala(config, json.loads((HERE / 'out' / 'fitted.json').read_text()))
    residual = None
    if not args.classical_only and (HERE / 'out' / 'residual.pt').exists():
        residual = Residual(config); residual.load_state_dict(torch.load(HERE / 'out' / 'residual.pt'))
    model = FullModel(classical, residual).eval().requires_grad_(False)
    sample = torch.tensor([[0.5, 2.0, 0.05, 2.1, 0.1, 10.0], [-1.5, 3.5, -0.4, 4.5, -0.2, 40.0]] * 4)
    path = HERE / 'out' / 'dynamics.onnx'
    torch.onnx.export(model, (sample,), str(path), input_names=['model_input'], output_names=['state_derivative'],
                      dynamic_axes={'model_input': {0: 'batch'}, 'state_derivative': {0: 'batch'}},
                      opset_version=args.opset, do_constant_folding=True, dynamo=False)
    save_json(str(path) + '.json', {
        'format': 'f1tenth_body_derivative_current_v1', 'residual_included': residual is not None,
        'input': {'name': 'model_input', 'shape': ['batch', 6],
                  'fields': ['yaw_rate_radps', 'speed_mps', 'sideslip_rad', 'driven_wheel_speed_mps',
                             'steering_rad', 'motor_current_a']},
        'output': {'name': 'state_derivative', 'shape': ['batch', 4],
                   'fields': ['yaw_acceleration_radps2', 'speed_acceleration_mps2', 'sideslip_rate_radps',
                              'driven_wheel_acceleration_mps2']},
        'state_reference': 'centre of gravity',
        'vehicle': config['vehicle'], 'parameters': {k: float(v) for k, v in classical.parameters_dict().items()},
        'steering_delay_s': config['data']['steering_delay_s']})
    print(f'wrote {path} ({"classical + residual" if residual is not None else "classical only"})')
    try:
        import onnxruntime
    except ImportError:
        print('onnxruntime not installed: skipped the numerical check'); return
    session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    difference = abs(session.run(None, {'model_input': sample.numpy()})[0] - model(sample).numpy()).max()
    print(f'ONNX vs PyTorch maximum difference: {difference:.2e}')


if __name__ == '__main__':
    main()
