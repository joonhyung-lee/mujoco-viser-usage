"""Own training/evaluation subprocesses; only their process groups are terminated."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import MjlabTrainConfig, TrainConfig
from .storage import new_run, read_json, write_json


class RunSupervisor:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[Path, subprocess.Popen] = {}
        self.viewers: dict[Path, subprocess.Popen] = {}

    @staticmethod
    def mjlab_python():
        executable = Path(
            os.environ.get(
                "MUJOCO_VISER_MJLAB_PYTHON",
                str(
                    Path(__file__).resolve().parents[3]
                    / "integrations/mjlab/.venv/bin/python"
                ),
            )
        )
        if not executable.is_file():
            raise ValueError(
                "Run uv sync --project integrations/mjlab before starting mjlab"
            )
        return str(executable)

    def start(
        self, config: TrainConfig | MjlabTrainConfig, resume: Path | None = None
    ) -> Path:
        if sum(proc.poll() is None for proc in self.jobs.values()) >= 2:
            raise ValueError(
                "Two experiments are already active; stop one before launching another"
            )
        native = isinstance(config, MjlabTrainConfig)
        executable = self.mjlab_python() if native else sys.executable
        run = new_run(self.root, config.task)
        write_json(run / "request.json", config.to_dict())
        command = [
            executable,
            "-m",
            "viser_robotics.mjlab.job" if native else "viser_robotics.rl.train",
            "--config",
            str(run / "request.json"),
            "--run-dir",
            str(run),
        ]
        if resume:
            command.extend(["--resume", str(Path(resume).resolve())])
        self._launch(run, command)
        return run

    def compare(self, checkpoint: Path) -> Path:
        if sum(proc.poll() is None for proc in self.jobs.values()) >= 2:
            raise ValueError("Two experiments are already active")
        executable = self.mjlab_python()
        run = new_run(self.root, "mjlab-comparison")
        self._launch(
            run,
            [
                executable,
                "-m",
                "viser_robotics.mjlab.compare_job",
                "--checkpoint",
                str(checkpoint.resolve()),
                "--run-dir",
                str(run),
            ],
        )
        return run

    def open_comparison(self, run: Path):
        if run in self.viewers and self.viewers[run].poll() is None:
            return
        (run / "viewer.json").unlink(missing_ok=True)
        with (run / "viewer.log").open("w") as log:
            self.viewers[run] = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "viser_robotics.rl.comparison",
                    str(run),
                    "--port",
                    "0",
                    "--announce",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def evaluate(self, checkpoint: Path, episodes: int = 3) -> Path:
        if sum(proc.poll() is None for proc in self.jobs.values()) >= 2:
            raise ValueError("Two experiments are already active")
        config = TrainConfig.load(checkpoint.parent.parent / "config.json")
        run = new_run(self.root, config.task + "-eval")
        self._launch(
            run,
            [
                sys.executable,
                "-m",
                "viser_robotics.rl.evaluate",
                "--checkpoint",
                str(checkpoint.resolve()),
                "--episodes",
                str(episodes),
                "--run-dir",
                str(run),
            ],
        )
        return run

    def _launch(self, run, command):
        write_json(
            run / "status.json",
            {"state": "starting", "updated_at": time.time(), "steps": 0},
        )
        write_json(run / "control.json", {"command": "run"})
        environment = os.environ.copy()
        environment.update(
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        with (run / "process.log").open("w") as log:
            self.jobs[run] = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )

    def command(self, run: Path, command: str):
        if command not in {"run", "pause", "stop"}:
            raise ValueError("Unknown process command")
        run = Path(run).resolve()
        if run.parent != self.root:
            raise ValueError("Run must belong to this control room")
        write_json(
            run / "control.json", {"command": command, "requested_at": time.time()}
        )

    def poll(self):
        for run, process in self.jobs.items():
            code = process.poll()
            status = read_json(run / "status.json", {})
            if code is not None and status.get("state") not in {
                "completed",
                "stopped",
                "failed",
            }:
                status.update(
                    state="failed",
                    error=f"Process exited with code {code}; see process.log",
                    updated_at=time.time(),
                )
                write_json(run / "status.json", status)

    def close(self):
        alive = [(run, p) for run, p in self.jobs.items() if p.poll() is None]
        for run, _ in alive:
            self.command(run, "stop")
        deadline = time.monotonic() + 8
        while any(p.poll() is None for _, p in alive) and time.monotonic() < deadline:
            time.sleep(0.05)
        for run, process in alive:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                status = read_json(run / "status.json", {})
                status.update(
                    state="stopped",
                    error="Supervisor terminated its process group during shutdown",
                    updated_at=time.time(),
                )
                write_json(run / "status.json", status)
        self.poll()
        for process in self.viewers.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
