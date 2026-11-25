"""Vehicle model, residual network, data windows and rollouts shared by the fitting / training scripts.

Model interface (what the MPPI controller evaluates):
  input  [batch, 6]: yaw rate [rad/s], speed [m/s], sideslip [rad], wheel speed [m/s], steering [rad], current [A]
  output [batch, 4]: time derivatives of the first four inputs
The body state refers to the centre of gravity. Current > 0 drives, current < 0 brakes (never reverses).

Prediction, identical here and in the controller:
  the classical model is integrated with explicit Euler steps of at most rollout.step_s;
  the residual is evaluated once per control interval (rollout.residual_hold_s) from the state at its
  start and the mean control over it, then added as a constant derivative for that interval.
The classical model is a port of xx_mppi dynamic_bicycle_fiala.hpp, locked-AWD branch.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

GRAVITY = 9.81
STATE_NAMES = ('yaw_rate', 'speed', 'sideslip', 'wheel_speed')
HERE = Path(__file__).resolve().parent


def load_config(path=None):
    return yaml.safe_load(Path(path or HERE / 'config.yaml').read_text())


def fiala_combined(alpha, kappa, stiffness, mu, load):
    """Lateral and longitudinal tyre force for slip angle alpha and slip ratio kappa (FialaCombinedSlip)."""
    alpha = alpha.clamp(-math.pi / 2 + 1e-2, math.pi / 2 - 1e-2)
    limit = (mu * load).clamp_min(1e-6)
    tangent = torch.tan(alpha)
    sigma = torch.sqrt(tangent ** 2 + kappa ** 2 + 1e-12)
    unsaturated = (stiffness * sigma - stiffness ** 2 / (3 * limit) * sigma ** 2
                   + stiffness ** 3 / (27 * limit ** 2) * sigma ** 3)
    magnitude = torch.where(sigma > torch.atan(3 * limit / stiffness), limit.expand_as(sigma), unsaturated)
    return -magnitude * tangent / sigma, magnitude * kappa / sigma


class Fiala(nn.Module):
    """Dynamic bicycle model with Fiala tyres, locked four-wheel drive and motor-current input."""

    def __init__(self, config, fitted=None):
        super().__init__()
        for name, value in config['vehicle'].items():
            self.register_buffer(name, torch.tensor(float(value)))
        self.names = list(config['identified'])
        table = torch.tensor([config['identified'][n] for n in self.names], dtype=torch.float32)
        self.register_buffer('low', table[:, 1])
        self.register_buffer('high', table[:, 2])
        start = table[:, 0] if fitted is None else torch.tensor([fitted[n] for n in self.names])
        fraction = ((start - self.low) / (self.high - self.low)).clamp(1e-3, 1 - 1e-3)
        self.raw = nn.Parameter(torch.logit(fraction))      # sigmoid keeps every parameter inside its bounds

    def parameters_dict(self):
        values = self.low + (self.high - self.low) * torch.sigmoid(self.raw)
        return dict(zip(self.names, values))

    def steering(self, servo):
        return (servo - self.servo_offset) / self.parameters_dict()['servo_gain']

    def step_limit(self):
        """Largest stable explicit integration step [s]: the wheel-speed state is the stiffest mode."""
        p = self.parameters_dict()
        stiffness = torch.maximum(p['front_cornering_stiffness_nprad'], p['rear_cornering_stiffness_nprad'])
        return (p['driven_wheel_inertia_kgm2'] / (self.wheel_radius_m ** 2 * stiffness)).item()

    def forward(self, model_input):
        p = self.parameters_dict()
        r, speed_in, beta, w, delta, current = model_input.unbind(-1)
        a, b = self.cg_to_front_m, self.cg_to_rear_m
        wheelbase, radius, mass = a + b, self.wheel_radius_m, self.mass_kg
        front_load, rear_load = mass * GRAVITY * b / wheelbase, mass * GRAVITY * a / wheelbase
        # Same 1 m/s floor as the C++ model: slip angles use the floored speed, slip ratios the true one.
        sign = torch.where(speed_in < 0, -torch.ones_like(speed_in), torch.ones_like(speed_in))
        speed = sign * speed_in.abs().clamp_min(1.0)
        beta = torch.where(speed_in.abs() > 1e-3, beta, torch.zeros_like(beta))
        vx, vy, vx_true = speed * torch.cos(beta), speed * torch.sin(beta), speed_in * torch.cos(beta)
        front_alpha = torch.atan2(vy + a * r, vx) - delta
        rear_alpha = torch.atan2(vy - b * r, vx)
        rear_kappa = (w - vx_true) / vx_true.abs().clamp_min(1.0)
        front_vx = torch.cos(delta) * vx_true + torch.sin(delta) * (vy + a * r)
        front_kappa = (w - front_vx) / front_vx.abs().clamp_min(1.0)
        fyf, fxf = fiala_combined(front_alpha, front_kappa, p['front_cornering_stiffness_nprad'],
                                  p['front_friction_coefficient'], front_load)
        fyr, fxr = fiala_combined(rear_alpha, rear_kappa, p['rear_cornering_stiffness_nprad'],
                                  p['rear_friction_coefficient'], rear_load)
        cd, sd = torch.cos(delta - beta), torch.sin(delta - beta)
        yaw_acceleration = (a * fxf * torch.sin(delta) + a * fyf * torch.cos(delta) - b * fyr) / p['yaw_inertia_kgm2']
        speed_rate = (cd * fxf - sd * fyf + torch.cos(beta) * fxr + torch.sin(beta) * fyr) / mass
        sideslip_rate = -r + (sd * fxf + cd * fyf - torch.sin(beta) * fxr + torch.cos(beta) * fyr) / (mass * speed)
        # Positive current drives. Negative current is a brake: like the driveline loss it opposes rotation
        # and vanishes at standstill, so it cannot reverse the wheels.
        gain, rolling = p['wheel_torque_per_amp_nm'], torch.tanh(w / 0.2)
        torque = gain * current.clamp_min(0) + (gain * current.clamp_max(0) - p['loss_torque_nm']) * rolling
        wheel_rate = radius * (torque - radius * (fxf + fxr)) / p['driven_wheel_inertia_kgm2']
        return torch.stack([yaw_acceleration, speed_rate, sideslip_rate, wheel_rate], -1)


class Residual(nn.Module):
    """Bounded correction to the model derivative; fades to zero at low speed where there is no data."""

    def __init__(self, config, hidden=None):
        super().__init__()
        c = config['residual']
        self.hidden = list(hidden or c['hidden'])
        sizes = [6, *self.hidden, 4]
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
        self.net = nn.Sequential(*layers[:-1])
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)   # starts as the plain model
        self.register_buffer('limit', torch.tensor(c['limit'], dtype=torch.float32))
        self.register_buffer('fade', torch.tensor(c['fade_speed_mps'], dtype=torch.float32))
        self.register_buffer('mean', torch.zeros(6))
        self.register_buffer('std', torch.ones(6))

    def set_normalization(self, inputs):
        self.mean.copy_(inputs.mean(0)); self.std.copy_(inputs.std(0).clamp_min(1e-3))

    def forward(self, model_input):
        gate = ((model_input[..., 1].abs() - self.fade[0]) / (self.fade[1] - self.fade[0])).clamp(0, 1)
        return self.limit * torch.tanh(self.net((model_input - self.mean) / self.std)) * gate.unsqueeze(-1)


def rollout(classical, residual, state, controls, dt, config):
    """Predict along recorded controls. state [N,4], controls [N,H,2] (steering rad, current A) -> states [N,H,4].

    Each recorded sample is integrated in equal Euler steps no longer than rollout.step_s.
    """
    r = config['rollout']
    substeps, hold = math.ceil(dt / r['step_s'] - 1e-9), round(r['residual_hold_s'] / dt)
    states, correction = [], 0.0
    for k in range(controls.shape[1]):
        if residual is not None and k % hold == 0:
            correction = residual(torch.cat([state, controls[:, k:k + hold].mean(1)], -1))
        for _ in range(substeps):
            state = state + dt / substeps * (classical(torch.cat([state, controls[:, k]], -1)) + correction)
        states.append(state)
    return torch.stack(states, 1)


# --- data ----------------------------------------------------------------------------------------------------
def load_sessions(config, files=None):
    """Portable .npz files -> list of contiguous usable runs, each a dict of float32 arrays at the sample rate.

    A run holds state [T,4] at the centre of gravity and control [T,2] = (delayed servo position, motor current).
    """
    d, lr = config['data'], float(config['vehicle']['cg_to_rear_m'])
    files = sorted((HERE / 'data').glob('*.npz')) if files is None else [Path(f) for f in files]
    runs, dt = [], None
    for path in files:
        z = np.load(path)
        t = z['t']; dt = float(np.median(np.diff(t)))
        vy_cg = z['vy'] + lr * z['r']                       # filter state is at the rear axle
        speed, beta = np.hypot(z['vx'], vy_cg), np.arctan2(vy_cg, z['vx'])
        delay = int(round(d['steering_delay_s'] / dt))
        servo = np.r_[np.full(delay, z['servo'][0]), z['servo'][:len(t) - delay]]
        ok = z['clean'] & (z['vx'] > d['min_speed_mps']) & (np.r_[dt, np.diff(t)] < 2.5 * dt)
        state = np.c_[z['r'], speed, beta, z['w']].astype(np.float32)
        control = np.c_[servo, z['current']].astype(np.float32)
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start >= int(d['min_run_s'] / dt):
                runs.append(dict(file=path.name, t0=float(t[start]), state=state[start:stop], control=control[start:stop]))
    if not runs:
        raise SystemExit('no usable data: check data/*.npz and the data section of config.yaml')
    return runs, dt


def split_runs(runs, config):
    """Whole runs go to validation (every n-th by duration order), never individual samples."""
    n = int(config['data']['validation_every'])
    order = sorted(range(len(runs)), key=lambda i: -len(runs[i]['state']))
    val = {i for k, i in enumerate(order) if k % n == n - 1}
    return [r for i, r in enumerate(runs) if i not in val], [r for i, r in enumerate(runs) if i in val]


def horizon_samples(config, dt, seconds=None):
    """Rollout length in samples; must be a whole number of residual intervals."""
    seconds = config['rollout']['horizon_s'] if seconds is None else seconds
    n, hold = round(seconds / dt), round(config['rollout']['residual_hold_s'] / dt)
    if n % hold:
        raise SystemExit(f'rollout of {seconds} s is not a multiple of residual_hold_s')
    return n


def windows(runs, horizon, stride):
    """Initial state [N,4], controls [N,H,2] and following true states [N,H,4]."""
    x0, u, x = [], [], []
    for run in runs:
        s, c = run['state'], run['control']
        for i in range(0, len(s) - horizon, stride):
            x0.append(s[i]); u.append(c[i:i + horizon]); x.append(s[i + 1:i + horizon + 1])
    if not x0:
        return (torch.zeros(0, 4), torch.zeros(0, horizon, 2), torch.zeros(0, horizon, 4))
    return tuple(torch.from_numpy(np.stack(a)) for a in (x0, u, x))


def to_model_controls(classical, controls):
    """(servo, current) -> (steering rad, current)."""
    return torch.stack([classical.steering(controls[..., 0]), controls[..., 1]], -1)


def rollout_error(classical, residual, batch, dt, config):
    """Scaled squared error of a rollout against the recorded states, and the raw error tensor."""
    x0, u, x = batch
    error = rollout(classical, residual, x0, to_model_controls(classical, u), dt, config) - x
    scale = torch.tensor(config['rollout']['error_scale'])
    return ((error / scale) ** 2).mean(), error


def rmse(error):
    return error.pow(2).mean((0, 1)).sqrt().tolist()


def describe(runs, dt, name):
    seconds = sum(len(r['state']) for r in runs) * dt
    slide = sum(int((np.abs(r['state'][:, 2]) > np.radians(10)).sum()) for r in runs) * dt
    print(f'{name}: {len(runs)} runs, {seconds:.0f} s, of which sideslip > 10 deg: {slide:.1f} s')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1) + '\n')
