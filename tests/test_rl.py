"""Run with `uv run --extra rl pytest tests/test_rl.py`. Tests spawn real processes."""

import time
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
import psutil
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from mujoco_viser_usage.rl.config import TrainConfig
from mujoco_viser_usage.rl.envs import RobotEnv, make_env
from mujoco_viser_usage.rl.evaluate import evaluate
from mujoco_viser_usage.rl.storage import EpisodeRecorder, read_json, read_jsonl
from mujoco_viser_usage.rl.supervisor import RunSupervisor
from mujoco_viser_usage.rl.train import train


@pytest.mark.parametrize(
    "task", ["go2_velocity", "aloha_reach", "aloha_pick_place", "franka_insert"]
)
def test_environment_contract_seed_and_episode_boundary(task):
    config = TrainConfig(task=task, max_episode_steps=3, control_dt=0.04)
    env = make_env(config)
    check_env(env)
    a, _ = env.reset(seed=91)
    b, _ = env.reset(seed=91)
    np.testing.assert_array_equal(a, b)
    for _ in range(3):
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape)
        )
        assert np.isfinite(observation).all() and np.isfinite(reward)
        assert info["reward_terms"]
        if terminated or truncated:
            break
    assert terminated or truncated
    with pytest.raises(RuntimeError):
        env.step(np.zeros(env.action_space.shape))
    env.close()


def test_insertion_baseline_beats_random_but_not_every_offset():
    env = make_env(
        TrainConfig(
            task="franka_insert", part="peg_16", control_dt=0.04, max_episode_steps=60
        )
    )
    outcomes = {}
    for name, policy in (
        ("scripted", lambda: env.scripted_action()),
        ("random", lambda: env.action_space.sample()),
    ):
        wins = 0
        for seed in range(6):
            env.reset(seed=seed)
            while True:
                _, _, terminated, truncated, info = env.step(policy())
                if terminated or truncated:
                    break
            wins += info["is_success"]
            assert info["max_penetration"] < 0.001
        outcomes[name] = wins
    assert outcomes["random"] < outcomes["scripted"] < 6, outcomes
    env.close()


def test_episode_trace_retains_real_state_and_seed(tmp_path):
    env = EpisodeRecorder(
        RobotEnv(TrainConfig(task="aloha_reach", max_episode_steps=3, record_every=2)),
        tmp_path,
        0,
    )
    env.reset(seed=123)
    for _ in range(3):
        env.step(np.zeros(14))
    record = read_jsonl(tmp_path / "workers/0/episodes.jsonl")[0]
    assert record["seed"] == 123 and record["length"] == 3
    assert record["truncated"]
    with np.load(tmp_path / "workers/0" / record["trace"]) as trace:
        assert len(trace["qpos"]) == 3  # reset, step two, terminal step three
        np.testing.assert_array_equal(trace["qpos"][-1], env.unwrapped.data.qpos)
        assert trace["time"][-1] == pytest.approx(0.06)
    env.close()


@pytest.mark.parametrize("task", ["go2_velocity", "aloha_reach"])
def test_spawned_training_changes_policy_and_supports_resume_evaluation(tmp_path, task):
    config = TrainConfig(
        task=task,
        workers=2,
        total_timesteps=256,
        n_steps=64,
        batch_size=128,
        max_episode_steps=32,
        save_every=128,
    )
    run = tmp_path / task
    train(config, run)
    status = read_json(run / "status.json")
    assert status["state"] == "completed"
    assert len(set(status["worker_pids"])) == 2
    assert status["updates"] > 0
    assert status["metrics"]["train/value_loss"] >= 0
    for pid in status["worker_pids"]:
        assert not psutil.pid_exists(pid)
    initial = PPO.load(run / "checkpoints/initial.zip", device="cpu")
    final = PPO.load(run / "checkpoints/final.zip", device="cpu")
    changes = [
        float((a - b).detach().abs().max())
        for a, b in zip(initial.policy.parameters(), final.policy.parameters())
    ]
    assert max(changes) > 1e-5
    assert read_jsonl(run / "workers/0/episodes.jsonl")
    resumed = tmp_path / (task + "-resumed")
    train(replace(config, total_timesteps=128), resumed, run / "checkpoints/final.zip")
    assert read_json(resumed / "status.json")["steps"] > status["steps"]
    result = evaluate(
        resumed / "checkpoints/final.zip", tmp_path / (task + "-evaluation"), episodes=1
    )
    assert result["episodes"] == 1 and np.isfinite(result["mean_return"])


def wait_until(predicate, seconds=45):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail("Timed out waiting for live training state")


def test_supervisor_pause_resume_stop_and_cleanup(tmp_path):
    supervisor = RunSupervisor(tmp_path)
    try:
        config = TrainConfig(
            workers=2,
            total_timesteps=1000000,
            n_steps=128,
            batch_size=128,
            max_episode_steps=50,
        )
        run = supervisor.start(config)
        wait_until(lambda: read_json(run / "status.json", {}).get("state") == "running")
        supervisor.command(run, "pause")
        wait_until(lambda: read_json(run / "status.json", {}).get("state") == "paused")
        paused = read_json(run / "status.json")
        assert all(psutil.pid_exists(pid) for pid in paused["worker_pids"])
        time.sleep(0.2)
        assert read_json(run / "status.json")["steps"] == paused["steps"]
        supervisor.command(run, "run")
        wait_until(
            lambda: read_json(run / "status.json", {}).get("steps", 0) > paused["steps"]
        )
        supervisor.command(run, "stop")
        wait_until(lambda: supervisor.jobs[run].poll() is not None)
        assert read_json(run / "status.json")["state"] == "stopped"
        assert (run / "checkpoints/final.zip").exists()
        for pid in paused["worker_pids"]:
            assert not psutil.pid_exists(pid)
    finally:
        supervisor.close()


def test_failed_worker_marks_run_failed_and_reaps_siblings(tmp_path):
    supervisor = RunSupervisor(tmp_path)
    try:
        run = supervisor.start(TrainConfig(workers=2, total_timesteps=1000000))
        wait_until(lambda: read_json(run / "status.json", {}).get("state") == "running")
        pids = read_json(run / "status.json")["worker_pids"]
        psutil.Process(pids[0]).terminate()
        wait_until(lambda: supervisor.jobs[run].poll() is not None)
        supervisor.poll()
        assert read_json(run / "status.json")["state"] == "failed"
        wait_until(lambda: all(not psutil.pid_exists(pid) for pid in pids))
    finally:
        supervisor.close()
