#!/usr/bin/env python3
"""Step 5: check that training followed the agreed setup and write the feedback report.

    python verify.py              # writes out/feedback.md and out/feedback.json
    python verify.py --freeze     # maintainers only: record data checksums and protected settings

Send back out/feedback.md, out/feedback.json, out/dynamics_model.json, out/evaluation.png and out/train_log.txt.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
CODE = ('core.py', 'fit_classical.py', 'train_residual.py', 'evaluate.py', 'export_model.py', 'verify.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected(config):
    return dict(vehicle=config['vehicle'], data=config['data'],
                rollout={k: config['rollout'][k] for k in ('step_s', 'residual_hold_s')})


def snapshot():
    config = yaml.safe_load((HERE / 'config.yaml').read_text())
    return dict(data={p.name: sha(p) for p in sorted((HERE / 'data').glob('*.npz'))},
                protected=protected(config), code={name: sha(HERE / name) for name in CODE})


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else None


def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ''


def close(a, b):
    """Equal up to single-precision round-off."""
    return len(a) == len(b) and all(abs(x - y) <= 1e-5 * max(1.0, abs(y)) for x, y in zip(a, b))


def scaled(rmse, scale):
    return sum((r / s) ** 2 for r, s in zip(rmse, scale)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    if parser.parse_args().freeze:
        (HERE / 'expected.json').write_text(json.dumps(snapshot(), indent=1) + '\n')
        return print('wrote expected.json')

    config = yaml.safe_load((HERE / 'config.yaml').read_text())
    expected, now = json.loads((HERE / 'expected.json').read_text()), snapshot()
    checks, info = [], {}

    def check(name, ok, detail=''):
        checks.append(dict(name=name, result='PASS' if ok else 'FAIL', detail=detail))

    # 1. Same data, same protected settings, same code.
    check('training data identical to the published files', now['data'] == expected['data'],
          '' if now['data'] == expected['data'] else
          f"differs: {sorted(set(now['data'].items()) ^ set(expected['data'].items()))}")
    check('protected settings unchanged (vehicle, data, rollout.step_s, rollout.residual_hold_s)',
          now['protected'] == expected['protected'], json.dumps(now['protected']))
    changed = [n for n in CODE if now['code'][n] != expected['code'][n]]
    check('scripts unchanged', not changed, f'modified: {changed}; explain every change in the feedback' if changed else '')
    r = config['rollout']
    check('rollout horizon is a whole number of residual intervals',
          abs(r['horizon_s'] / r['residual_hold_s'] - round(r['horizon_s'] / r['residual_hold_s'])) < 1e-9, f"{r['horizon_s']} s")

    # 2. Classical fit.
    fitted = load('fitted.json')
    check('out/fitted.json exists', fitted is not None)
    if fitted:
        bounds = config['identified']
        check('fitted parameters inside their bounds', all(bounds[k][1] <= v <= bounds[k][2] for k, v in fitted.items()))
        info['parameters at a bound (not determined by the data)'] = [
            k for k, v in fitted.items() if min(v - bounds[k][1], bounds[k][2] - v) < 0.02 * (bounds[k][2] - bounds[k][1])]
        limit = fitted['driven_wheel_inertia_kgm2'] / (config['vehicle']['wheel_radius_m'] ** 2 * max(
            fitted['front_cornering_stiffness_nprad'], fitted['rear_cornering_stiffness_nprad']))
        check('integration step below the stability limit of the fitted model', r['step_s'] < limit,
              f"step {r['step_s'] * 1e3:.1f} ms, limit {limit * 1e3:.2f} ms")
        info['fitted parameters'] = fitted
        info['controller substeps needed per 0.1 s'] = int(-(-0.1 // limit)) + 1

    # 3. Held-out evaluation.
    evaluation = load('evaluation.json')
    check('out/evaluation.json exists', evaluation is not None)
    if evaluation:
        table = {}
        for key, value in evaluation.items():
            horizon, model, subset = key.split('/')
            table.setdefault(model, {})[f'{horizon} {subset}'] = round(scaled(value['rmse'], r['error_scale']), 4)
        info['held-out scaled error (lower is better)'] = table
        info['held-out RMSE [yaw rate rad/s, speed m/s, sideslip rad, wheel speed m/s]'] = {
            k: [round(x, 4) for x in v['rmse']] for k, v in evaluation.items()}
        base = table.get('classical', {})
        verdict = {}
        for model, row in table.items():
            if model != 'classical':
                verdict[model] = {k: f'{100 * (row[k] / base[k] - 1):+.0f} %' for k in row if k in base and base[k] > 0}
        info['change against the classical model'] = verdict
        info['held-out rollouts'] = {k: v['rollouts'] for k, v in evaluation.items() if '/classical/' in k}

    # 4. Exported model.
    model = load('dynamics_model.json')
    check('out/dynamics_model.json exists', model is not None)
    if model:
        check('model format and interface', model.get('format') == 'f1tenth_dynamics_v1' and
              len(model['interface']['input']) == 6 and len(model['interface']['output']) == 4)
        check('exported integration settings match config.yaml',
              model['integration']['step_s'] == r['step_s'] and model['integration']['residual_hold_s'] == r['residual_hold_s'])
        check('exported parameters equal out/fitted.json', fitted is not None and
              close([model['parameters'][k] for k in fitted], list(fitted.values())))
        res = model['residual']
        if res:
            sizes = [6, *res['hidden'], 4]
            shapes = [[len(layer['weight']), len(layer['weight'][0])] for layer in res['layers']]
            check('residual layer shapes', shapes == [[sizes[i + 1], sizes[i]] for i in range(len(sizes) - 1)], str(shapes))
            check('residual limits and speed fade equal config.yaml',
                  close(res['limit'], config['residual']['limit']) and
                  close(res['fade_speed_mps'], config['residual']['fade_speed_mps']))
            export = load('export_check.json')
            check('export self-check below 1e-4', export is not None and export['max_difference'] < 1e-4,
                  f"{export['max_difference']:.2e}" if export else 'out/export_check.json missing')
        info['exported residual'] = 'none (classical only)' if not res else 'x'.join(map(str, res['hidden']))
        check('reference vectors present', len(model.get('test_vectors', [])) >= 6)

    try:
        import torch
        version = torch.__version__
    except ImportError:
        version = 'not installed'
    info['environment'] = dict(python=sys.version.split()[0], torch=version, platform=platform.platform(),
                               commit=git('rev-parse', 'HEAD'), uncommitted=git('status', '--porcelain', '--', *CODE, 'config.yaml'))
    info['settings'] = {k: config[k] for k in ('rollout', 'classical', 'residual')}
    passed = all(c['result'] == 'PASS' for c in checks)
    OUT.mkdir(exist_ok=True)
    (OUT / 'feedback.json').write_text(json.dumps(dict(passed=passed, checks=checks, info=info), indent=1) + '\n')
    lines = ['# Training feedback', '', f"Overall: {'PASS' if passed else 'FAIL'}", '', '| check | result | detail |', '|---|---|---|']
    lines += [f"| {c['name']} | {c['result']} | {c['detail']} |" for c in checks]
    for title, value in info.items():
        lines += ['', f'## {title}', '', '```', json.dumps(value, indent=1, ensure_ascii=False), '```']
    (OUT / 'feedback.md').write_text('\n'.join(lines) + '\n')
    for c in checks:
        print(f"{c['result']}  {c['name']}" + (f"  [{c['detail']}]" if c['detail'] and c['result'] == 'FAIL' else ''))
    print(f"\nOverall: {'PASS' if passed else 'FAIL'}; wrote out/feedback.md and out/feedback.json")


if __name__ == '__main__':
    main()
