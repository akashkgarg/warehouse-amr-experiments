"""Train / evaluate one shared, sensor-only navigation policy for every robot in one or more warehouse sim servers.

Start the sims first (one per GPU), e.g.:
    ./python.sh tools/warehouse_amr/warehouse_sim.py serve --gpu 4 --address 127.0.0.1:47800
    ./python.sh tools/warehouse_amr/warehouse_sim.py serve --gpu 5 --address 127.0.0.1:47801 --seed 1

Then, from a Python with torch (e.g. ~/.venvs/franka-policy/bin/python):
    python tools/warehouse_amr/train_ppo.py train --address 127.0.0.1:47800,127.0.0.1:47801 --run ~/warehouse_amr/runs/ppo1
    python tools/warehouse_amr/train_ppo.py eval --address 127.0.0.1:47800 --checkpoint ~/warehouse_amr/runs/ppo1/policy.pt \
        --episodes 20 --out ~/warehouse_amr/bench/policy.json [--record eval.mp4]

Teacher-student: the actor sees only the sensor observation (noisy lidar, odometry, estimated pose); the critic also
sees the privileged state; DAgger labels come from the privileged scripted expert. Training starts with DAgger
(``--dagger-iters``) and continues with PPO. Every robot is a parallel agent, so a heterogeneous fleet trains a
single type-conditioned policy. ``eval --checkpoint expert`` benchmarks the scripted (privileged) baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils  # noqa: E402
from warehouse_client import FleetClient, WarehouseClient  # noqa: E402

FAILURES = bench_utils.FAILURES


class RunningNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x: torch.Tensor) -> None:
        m, v, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, total = m - self.mean, self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-8, 8)


def mlp(i: int, o: int, h: int = 256) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, priv_dim: int, act_dim: int):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.norm_c = RunningNorm(obs_dim + priv_dim)
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim + priv_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.norm(obs))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp())

    def value(self, obs: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        return self.critic(self.norm_c(torch.cat([obs, priv], -1))).squeeze(-1)


class EpisodeStats:
    def __init__(self, types: list[str], window: int = 200):
        self.types = types
        self.events = defaultdict(lambda: deque(maxlen=window))
        self.returns = defaultdict(lambda: deque(maxlen=window))

    def add(self, infos: list[dict]) -> None:
        for i, info in enumerate(infos):
            if info:
                t = self.types[i]
                self.events[t].append(info["event"])
                self.returns[t].append(info["episode_return"])

    def summary(self) -> dict:
        out = {}
        for t, ev in self.events.items():
            n = len(ev)
            out[t] = {
                "episodes": n,
                **{e: round(sum(x == e for x in ev) / n, 3) for e in ("success", *FAILURES, "timeout")},
                "return": round(float(np.mean(self.returns[t])), 2),
            }
        return out


def save(model: ActorCritic, path: str, meta: dict) -> None:
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), **meta}, tmp)
    os.replace(tmp, path)
    layers = [m for m in model.actor if isinstance(m, nn.Linear)]
    arrays = {"mean": model.norm.mean.cpu().numpy(), "var": model.norm.var.cpu().numpy(), "n_layers": len(layers)}
    for i, m in enumerate(layers):
        arrays[f"w{i}"] = m.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = m.bias.detach().cpu().numpy()
    npz = os.path.splitext(path)[0] + ".npz"
    np.savez(npz + ".tmp.npz", **arrays)
    os.replace(npz + ".tmp.npz", npz)


def load(path: str, device) -> ActorCritic:
    ck = torch.load(path, map_location=device)
    model = ActorCritic(ck["obs_dim"], ck["priv_dim"], ck["act_dim"]).to(device)
    model.load_state_dict(ck["model"])
    return model


def train(args) -> None:
    os.makedirs(args.run, exist_ok=True)
    device = torch.device(args.device)
    env = FleetClient(args.address.split(","))
    n = env.num_robots
    print(f"fleet: {n} robots {env.robot_names}", flush=True)
    model = ActorCritic(env.obs_dim, env.priv_dim, env.act_dim).to(device)
    step0, phase = 0, "dagger"
    if args.resume and os.path.exists(os.path.join(args.run, "policy.pt")):
        ck = torch.load(os.path.join(args.run, "policy.pt"), map_location=device)
        model.load_state_dict(ck["model"])
        step0, phase = ck.get("step", 0), ck.get("phase", "ppo")
        print(f"resumed from step {step0} ({phase})", flush=True)
    elif args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        phase = "ppo"
        with torch.no_grad():
            model.log_std.fill_(args.ppo_init_log_std)
        print(f"initialized from {args.init}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.dagger_lr if phase == "dagger" else args.lr)
    stats = EpisodeStats(env.robot_types)
    log = open(os.path.join(args.run, "metrics.jsonl"), "a")
    meta = lambda step, ph: {"step": step, "phase": ph, "obs_dim": env.obs_dim, "priv_dim": env.priv_dim,
                             "act_dim": env.act_dim, "robots": env.robot_names, "types": env.robot_types}

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    obs = env.reset(seed=args.seed)
    t_start, total = time.time(), step0

    # ---------------------------------------------------------------- DAgger warm start (student imitates teacher)
    if phase == "dagger" and args.dagger_iters > 0:
        data_obs = deque(maxlen=args.dagger_buffer // (args.rollout * n) + 1)
        data_act = deque(maxlen=data_obs.maxlen)
        for it in range(args.dagger_iters):
            beta = max(0.0, 1.0 - it / max(1, args.dagger_iters // 2))
            chunk_o, chunk_a = [], []
            for _ in range(args.rollout):
                expert = env.expert()
                with torch.no_grad():
                    mean = model.mean(torch.as_tensor(obs, device=device)).clamp(-1, 1).cpu().numpy()
                use_expert = np.random.rand(n, 1) < beta
                chunk_o.append(obs.copy())
                chunk_a.append(expert.copy())
                obs, rew, term, trunc, infos = env.step(np.where(use_expert, expert, mean),
                                                        caption=f"DAgger it {it} beta {beta:.2f}")
                stats.add(infos)
                total += n
            data_obs.append(np.concatenate(chunk_o))
            data_act.append(np.concatenate(chunk_a))
            X = torch.as_tensor(np.concatenate(data_obs), device=device)
            Y = torch.as_tensor(np.concatenate(data_act), device=device)
            model.norm.update(torch.as_tensor(data_obs[-1], device=device))
            for _ in range(args.dagger_epochs):
                for b in torch.randperm(len(X), device=device).split(1024):
                    loss = ((model.mean(X[b]) - Y[b]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            emit({"phase": "dagger", "iter": it, "beta": round(beta, 3), "bc_loss": loss.item(), "samples": total,
                  "dataset": len(X), "sps": round((total - step0) / (time.time() - t_start), 1),
                  "stats": stats.summary()})
            save(model, os.path.join(args.run, "policy.pt"), meta(total, "dagger"))
        phase = "ppo"
        save(model, os.path.join(args.run, "policy_dagger.pt"), meta(total, "ppo"))
        # Start PPO from the imitated policy with a small exploration std; the critic warms up first.
        with torch.no_grad():
            model.log_std.fill_(args.ppo_init_log_std)
    for g in opt.param_groups:
        g["lr"] = args.lr

    # ---------------------------------------------------------------- PPO (asymmetric actor-critic)
    T = args.rollout
    it = 0
    priv = env.priv
    ppo_start = total
    while total < args.total_steps:
        # Expert-regularized PPO: an annealed BC term on the (privileged) expert's labels keeps the student from
        # unlearning the DAgger policy while the critic is still poor.
        frac = min(1.0, (total - ppo_start) / max(1, args.bc_anneal))
        bc_coef = args.bc_coef + frac * (args.bc_final - args.bc_coef)
        buf = {k: [] for k in ("obs", "priv", "act", "logp", "val", "rew", "done", "expert")}
        for _ in range(T):
            o = torch.as_tensor(obs, device=device)
            p = torch.as_tensor(priv, device=device)
            expert = env.expert() if bc_coef > 0 else np.zeros((n, env.act_dim), np.float32)
            with torch.no_grad():
                d = model.dist(o)
                a = d.sample()
                logp = d.log_prob(a).sum(-1)
                v = model.value(o, p)
            obs, rew, term, trunc, infos = env.step(np.clip(a.cpu().numpy(), -1, 1), caption=f"PPO it {it}")
            obs = np.nan_to_num(obs)
            priv = np.nan_to_num(env.priv)
            stats.add(infos)
            for k, x in zip(buf, (o, p, a, logp, v, torch.as_tensor(rew, device=device),
                                  torch.as_tensor(term | trunc, device=device, dtype=torch.float32),
                                  torch.as_tensor(expert, device=device))):
                buf[k].append(x)
            total += n
        with torch.no_grad():
            last_v = model.value(torch.as_tensor(obs, device=device), torch.as_tensor(priv, device=device))
        B = {k: torch.stack(v) for k, v in buf.items()}
        adv = torch.zeros_like(B["rew"])
        gae = torch.zeros(n, device=device)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else B["val"][t + 1]
            delta = B["rew"][t] + args.gamma * nv * (1 - B["done"][t]) - B["val"][t]
            gae = delta + args.gamma * args.lam * (1 - B["done"][t]) * gae
            adv[t] = gae
        ret = adv + B["val"]
        model.norm.update(B["obs"].reshape(-1, env.obs_dim))
        model.norm_c.update(torch.cat([B["obs"], B["priv"]], -1).reshape(-1, env.obs_dim + env.priv_dim))
        flat = {k: B[k].reshape(T * n, *B[k].shape[2:]) for k in ("obs", "priv", "act", "logp", "expert")}
        flat_adv, flat_ret = adv.reshape(-1), ret.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        actor_on = it >= args.critic_warmup
        for _ in range(args.epochs):
            for idx in torch.randperm(T * n, device=device).chunk(args.minibatches):
                d = model.dist(flat["obs"][idx])
                v = model.value(flat["obs"][idx], flat["priv"][idx])
                logp = d.log_prob(flat["act"][idx]).sum(-1)
                ratio = (logp - flat["logp"][idx]).exp()
                pg = -torch.min(ratio * flat_adv[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * flat_adv[idx]).mean()
                vl = 0.5 * ((v - flat_ret[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                bc = ((d.mean - flat["expert"][idx]) ** 2).mean()
                loss = args.vf_coef * vl + bc_coef * bc + (pg - args.ent_coef * ent if actor_on else 0.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        emit({"phase": "ppo", "iter": it, "samples": total, "sps": round((total - step0) / (time.time() - t_start), 1),
              "pg": pg.item(), "vf": vl.item(), "bc": bc.item(), "bc_coef": round(bc_coef, 3),
              "entropy": ent.item(), "std": model.log_std.exp().tolist(),
              "actor": actor_on, "reward_mean": B["rew"].mean().item(), "stats": stats.summary()})
        save(model, os.path.join(args.run, "policy.pt"), meta(total, "ppo"))
        if args.snapshot_every and it % args.snapshot_every == 0:
            save(model, os.path.join(args.run, f"policy_{total}.pt"), meta(total, "ppo"))
        it += 1
    env.close()


def evaluate(args) -> None:
    """Run the fixed benchmark episodes on one server with the expert or a checkpoint; ground-truth-free actor."""
    device = torch.device(args.device)
    env = WarehouseClient(args.address.split(",")[0])
    model = None if args.checkpoint == "expert" else load(args.checkpoint, device).eval()
    episodes = env.episodes(per_robot=args.episodes, seed=args.bench_seed)
    total = sum(len(e) for e in episodes)
    if args.record:
        env.record(os.path.abspath(os.path.expanduser(args.record)), follow=args.follow)
    obs = env.reset(seed=args.seed)
    records, step, t0 = [], 0, time.time()
    label = "expert" if model is None else os.path.basename(args.checkpoint)
    while len(records) < total and step < args.max_steps:
        if model is None:
            act = env.expert()
        else:
            with torch.no_grad():
                act = model.mean(torch.as_tensor(obs, device=device)).clamp(-1, 1).cpu().numpy()
        obs, rew, term, trunc, infos = env.step(act, caption=f"{label}  t={step / 10:.1f}s  done {len(records)}/{total}")
        step += 1
        for info in infos:
            if info:
                records.append(info)
                print(f"[{label}] {len(records)}/{total} {info['robot']}: {info['event']} in {info['time']:.1f}s",
                      flush=True)
        if step % 50 == 0 and env.status()["benchmark_done"]:
            break
    skipped = env.status()["skipped_episodes"]
    if args.record:
        env.record(None)
    env.episodes(episodes=None)
    out = os.path.expanduser(args.out) if args.out else None
    meta = {"controller": label, "checkpoint": args.checkpoint, "bench_seed": args.bench_seed,
            "steps": step, "skipped_episodes": skipped, "wall_s": round(time.time() - t0, 1)}
    summary = bench_utils.write(out, records, meta) if out else bench_utils.summarize(records)
    print(json.dumps({**meta, "summary": summary}, indent=1))
    env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("train", "eval"))
    p.add_argument("--address", default="127.0.0.1:47800", help="comma-separated sim server addresses")
    p.add_argument("--run", default=os.path.expanduser("~/warehouse_amr/runs/ppo"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", default=None, help="train: start PPO from this checkpoint (skips DAgger)")
    p.add_argument("--checkpoint", default=None, help="eval: policy.pt, or 'expert' for the scripted baseline")
    p.add_argument("--episodes", type=int, default=10, help="eval: benchmark episodes per robot")
    p.add_argument("--bench-seed", type=int, default=12345, help="eval: seed of the fixed episode list")
    p.add_argument("--max-steps", type=int, default=100000, help="eval: safety cap on control steps")
    p.add_argument("--out", default=None, help="eval: write per-episode results + summary JSON here")
    p.add_argument("--record", default=None, help="eval: MP4 path (written by the sim server)")
    p.add_argument("--follow", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--total-steps", type=int, default=2_000_000, help="robot-steps (samples) to train for")
    p.add_argument("--rollout", type=int, default=128, help="control steps per iteration")
    p.add_argument("--dagger-iters", type=int, default=40)
    p.add_argument("--dagger-epochs", type=int, default=3)
    p.add_argument("--dagger-buffer", type=int, default=400_000, help="most recent samples kept for DAgger")
    p.add_argument("--dagger-lr", type=float, default=3e-4)
    p.add_argument("--ppo-init-log-std", type=float, default=-2.0)
    p.add_argument("--critic-warmup", type=int, default=20, help="PPO iterations that train only the critic")
    p.add_argument("--bc-coef", type=float, default=2.0, help="PPO: initial weight of the expert BC term")
    p.add_argument("--bc-final", type=float, default=0.1, help="PPO: final weight of the expert BC term")
    p.add_argument("--bc-anneal", type=int, default=1_500_000, help="PPO: samples over which the BC weight anneals")
    p.add_argument("--snapshot-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    args = p.parse_args()
    args.run = os.path.expanduser(args.run)
    if args.mode == "eval" and not args.checkpoint:
        p.error("eval needs --checkpoint")
    train(args) if args.mode == "train" else evaluate(args)


if __name__ == "__main__":
    main()
