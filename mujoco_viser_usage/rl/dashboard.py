"""Viser control room for real worker processes, training runs and episode replay."""

import argparse
import time
from dataclasses import replace
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
import numpy as np
import psutil
import viser

from ..overlays import ForceOverlay
from ..scenes import TERRAINS
from ..simulation import Simulation
from ..viewer import PhysicsViewer
from .comparison import apply_frame
from .config import MjlabTrainConfig, TrainConfig, load_config, parse_config, scene_path
from .overview import ParallelOverview, base_body_name
from .playback import PlaybackClock, RolloutPlayer, sample_trace
from .storage import read_json, read_jsonl
from .supervisor import RunSupervisor

TASK_LABELS = {
    "ALOHA · pick & place": "aloha_pick_place",
    "ALOHA · reach": "aloha_reach",
    "Go2 · walk": "go2_velocity",
    "Franka · insert": "franka_insert",
}


class ControlRoom:
    def __init__(self, root: Path, host="127.0.0.1", port=8090):
        self.supervisor = RunSupervisor(root)
        self.server = viser.ViserServer(host=host, port=port)
        self.commands = SimpleQueue()
        self.selected_run = None
        self.loaded_task = None
        self.loaded_terrain = "flat"
        self.loaded_model_path = None
        self.players = {}
        self.viewer = None
        self.overview = None
        self.overview_data = []
        self.replay = None
        self.latest_stamp = None
        self.episode_paths = {}
        self.clock = PlaybackClock()
        self.display_workers = 1
        self._build_controls()
        self._load_scene("aloha_pick_place", terrain="flat")
        self.server.on_client_connect(self._focus)
        self._refresh_runs()
        native_runs = [
            path
            for path in self.supervisor.root.iterdir()
            if path.is_dir()
            and read_json(path / "config.json", {}).get("backend") == "mjlab"
            and (path / "status.json").exists()
        ]
        if native_runs:
            self._select_run(
                max(native_runs, key=lambda path: path.stat().st_mtime_ns).name
            )
        else:
            self._preview_task(self.task_choice.value)

    def _build_controls(self):
        gui = self.server.gui
        gui.add_markdown("## Robot rollouts")
        self.task_choice = gui.add_dropdown(
            "Scene", options=tuple(TASK_LABELS), initial_value="ALOHA · pick & place"
        )
        self.task_choice.on_update(
            lambda e: (
                self.commands.put(lambda: self._preview_task(e.target.value))
                if e.client
                else None
            )
        )
        self.status = gui.add_markdown("Choose a scene, then open Training to start.")
        self.notice = gui.add_markdown("")
        self.comparison_link = gui.add_markdown("")
        self.progress = gui.add_progress_bar(0)
        self.play_button = gui.add_button("Pause rollout", icon=viser.Icon.PLAYER_PAUSE)
        self.play_button.on_click(lambda _: self.commands.put(self._toggle_playback))
        actions = gui.add_button_group("", options=("Restart", "Focus"))
        actions.on_click(
            lambda e: (
                self.commands.put(self._restart_playback)
                if e.target.value == "Restart"
                else self._focus(e.client)
            )
        )
        with gui.add_folder("View", expand_by_default=False):
            self.view_mode = gui.add_dropdown(
                "Layout", options=("Selected worker", "Parallel overview")
            )
            self.view_mode.on_update(
                lambda e: self.commands.put(self._change_view) if e.client else None
            )
            self.worker_choice = gui.add_dropdown("Worker", options=("0",))
            self.worker_choice.on_update(
                lambda e: self.commands.put(self._live) if e.client else None
            )
            self.follow_camera = gui.add_checkbox("Follow robot", initial_value=True)
            self.replay_speed = gui.add_dropdown(
                "Speed", options=("0.5", "1", "2"), initial_value="1"
            )
            self.loop_rollout = gui.add_checkbox(
                "Repeat finished rollout", initial_value=True
            )
            self.stream_mode = gui.add_dropdown(
                "Source", options=("Smooth rollouts (buffered)", "Live snapshots")
            )
            self.stream_mode.on_update(
                lambda e: self.commands.put(self._live) if e.client else None
            )
        with gui.add_folder("Training", expand_by_default=False):
            self.terrain_choice = gui.add_dropdown(
                "Terrain", options=TERRAINS, initial_value="flat", visible=False
            )
            self.terrain_choice.on_update(
                lambda e: (
                    self.commands.put(lambda: self._preview_terrain())
                    if e.client
                    else None
                )
            )
            self.start_button = gui.add_button(
                "Train new policy", icon=viser.Icon.PLAYER_PLAY
            )
            self.start_button.on_click(lambda _: self.commands.put(self._start))
            with gui.add_folder("Settings", expand_by_default=False):
                self.backend = gui.add_dropdown(
                    "Backend", options=("mjlab", "SB3"), initial_value="mjlab"
                )
                self.backend.on_update(
                    lambda e: (
                        self.commands.put(
                            lambda: self._preview_task(self.task_choice.value)
                        )
                        if e.client
                        else None
                    )
                )
                self.device = gui.add_dropdown(
                    "Training device", options=("cpu", "cuda:0"), initial_value="cpu"
                )
                self.worker_count = gui.add_dropdown(
                    "Parallel workers", options=("1", "2", "4", "8"), initial_value="4"
                )
                self.budget = gui.add_number(
                    "Training steps", initial_value=16384, min=512, step=512
                )
                self.seed = gui.add_number("Seed", initial_value=7, min=0, step=1)
                actions = gui.add_button_group(
                    "Training process", options=("Pause", "Resume", "Stop")
                )
                actions.on_click(
                    lambda e: self.commands.put(
                        lambda: self._command(
                            {"Pause": "pause", "Resume": "run", "Stop": "stop"}[
                                e.target.value
                            ]
                        )
                    )
                )
                self.checkpoint = gui.add_dropdown("Checkpoint", options=("None",))
                resume = gui.add_button("Continue from checkpoint")
                resume.on_click(lambda _: self.commands.put(self._resume_checkpoint))
                evaluate = gui.add_button("Evaluate checkpoint · 3 episodes")
                evaluate.on_click(lambda _: self.commands.put(self._evaluate))
                self.cancel_comparison = gui.add_button(
                    "Stop comparison", visible=False
                )
                self.cancel_comparison.on_click(
                    lambda _: self.commands.put(self._stop_comparison)
                )
            with gui.add_folder("Metrics", expand_by_default=False):
                self.plot = gui.add_uplot(
                    data=(np.array([0.0]), np.array([0.0])),
                    series=(
                        {},
                        {"label": "Episode return", "stroke": "#76b900", "width": 2},
                    ),
                    scales={"x": {"time": False}},
                    height=150,
                )
                self.worker_table = gui.add_markdown("No workers yet.")
                self.metrics = gui.add_markdown("")
            gui.add_markdown(
                "PPO policy rollout. A short run does not imply successful grasping."
            )
        with gui.add_folder("History", expand_by_default=False):
            self.run_choice = gui.add_dropdown("Experiment", options=("No runs",))
            self.run_choice.on_update(
                lambda e: (
                    self.commands.put(lambda: self._select_run(e.target.value))
                    if e.client
                    else None
                )
            )
            self.episode_choice = gui.add_dropdown(
                "Recorded episode", options=("None",)
            )
            replay = gui.add_button("Replay selected episode")
            replay.on_click(lambda _: self.commands.put(self._replay))
            live = gui.add_button("Latest rollout")
            live.on_click(lambda _: self.commands.put(self._live))
            self.episode_table = gui.add_markdown("")

    def _toggle_playback(self):
        self.clock.advance(time.monotonic(), float(self.replay_speed.value))
        self.clock.paused = not self.clock.paused
        self.play_button.label = (
            "Play rollout" if self.clock.paused else "Pause rollout"
        )

    def _restart_playback(self):
        if self.replay is not None:
            self.replay_started = self.clock.value
        else:
            self._live()
        self.clock.paused = False
        self.play_button.label = "Pause rollout"

    def _focus(self, client):
        if client is None:
            return
        if self.overview is not None:
            self.overview.focus(client)
            return
        if self.viewer is None:
            return
        if self.loaded_task == "go2_velocity":
            target = self.sim.data.body(self.base_body).xpos.copy()
            client.camera.position = target + np.array([1.8, -2.4, 1.8])
            client.camera.look_at = target + np.array([0.5, 0.0, 0.1])
            client.camera.fov = np.deg2rad(45)
        elif self.loaded_task == "franka_insert":
            client.camera.position = (0.95, -0.6, 0.42)
            client.camera.look_at = (0.56, -0.05, 0.03)
            client.camera.fov = np.deg2rad(43)
        else:
            client.camera.position = (1.25, -1.45, 1.3)
            client.camera.look_at = (0.0, 0.04, 0.15)
            client.camera.fov = np.deg2rad(43)

    def _load_scene(self, task, force=False, terrain=None):
        terrain = self.loaded_terrain if terrain is None else terrain
        if task != "go2_velocity":
            terrain = "flat"
        model_path = scene_path(TrainConfig(task=task, terrain=terrain))
        exported = self.selected_run / "scene/scene.xml" if self.selected_run else None
        native = self._native_run() and exported is not None and exported.exists()
        if native:
            model_path = exported
        if (
            self.loaded_task == task
            and self.loaded_terrain == terrain
            and self.loaded_model_path == model_path
            and not force
        ):
            return
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        self.overview = None
        self.server.scene.reset()
        self.sim = Simulation(model_path)
        self.loaded_model_path = model_path
        self.base_body = base_body_name(self.sim.model)
        if native and self.sim.model.nkey:
            mujoco.mj_resetDataKeyframe(self.sim.model, self.sim.data, 0)
            mujoco.mj_forward(self.sim.model, self.sim.data)
        self.loaded_terrain = terrain
        self.players = {}
        if native:
            pass  # Exported runtime options and actuator gains are authoritative.
        elif task == "go2_velocity":
            from ..scenes.go2_task import configure_servos

            configure_servos(self.sim)
        elif task.startswith("aloha"):
            from ..scenes.aloha_task import configure_grasp_contacts

            configure_grasp_contacts(self.sim)
        if self.view_mode.value == "Parallel overview":
            self.overview = ParallelOverview(
                self.server, self.sim.model, count=self.display_workers
            )
            self.overview_data = [
                mujoco.MjData(self.sim.model)
                for _ in range(min(4, self.display_workers))
            ]
            self.overview_stamps = [None] * 4
            self.overview_forces = [
                ForceOverlay(
                    self.server, self.sim.model, namespace=f"/agents/{i}/forces"
                )
                for i in range(len(self.overview_data))
            ]
        else:
            self.viewer = PhysicsViewer(
                self.server, self.sim.model, connect_camera=False, show_plan=False
            )
            if task == "go2_velocity":
                for i, handle in self.viewer.handles.items():
                    if self.sim.model.geom(i).name.startswith("wall_"):
                        handle.visible = False
            self.goal_handles = [
                self.server.scene.add_icosphere(
                    f"/rl_goal/{i}",
                    radius=0.008 if task == "franka_insert" else 0.023,
                    color=(118, 185, 0),
                    opacity=0.5,
                    visible=False,
                )
                for i in range(2)
            ]
        self.loaded_task = task
        if self.viewer is not None:
            self.viewer.update(self.sim.data)
        for client in self.server.get_clients().values():
            self._focus(client)
        self.latest_stamp = None
        self._camera_target = None
        self.progress.visible = self.overview is None

    def _change_view(self):
        self.replay = None
        self._load_scene(self.loaded_task, force=True)
        self.notice.content = (
            "Parallel overview · buffered rollouts at 1×, up to four workers"
            if self.overview
            else "Selected worker · choose buffered rollouts or live snapshots"
        )

    def _update_backend_ui(self):
        native = self.backend.value == "mjlab"
        self.worker_count.label = "Parallel worlds" if native else "Parallel workers"
        options = (
            ("1", "2", "4", "8", "16", "64", "128", "512")
            if native
            else ("1", "2", "4", "8", "16")
        )
        self.worker_count.options = options
        if self.worker_count.value not in options:
            self.worker_count.value = "4"
        self.device.visible = native
        self.start_button.label = "Train with mjlab" if native else "Train with SB3"

    def _native_run(self):
        if self.selected_run is None:
            return False
        payload = read_json(self.selected_run / "config.json") or read_json(
            self.selected_run / "request.json", {}
        )
        return payload.get("backend") == "mjlab"

    def _preview_task(self, label):
        task = TASK_LABELS[label]
        self.terrain_choice.visible = task == "go2_velocity"
        self._update_backend_ui()
        matches = []
        alternatives = []
        for directory in self.supervisor.root.iterdir():
            payload = (
                read_json(directory / "config.json", {}) if directory.is_dir() else {}
            )
            if payload.get("task") == task and (directory / "status.json").exists():
                alternatives.append(directory)
            if (
                payload.get("task") == task
                and payload.get("backend", "sb3") == self.backend.value.lower()
                and (directory / "status.json").exists()
            ):
                matches.append(directory)
        if not matches:
            # Show the task's latest run from the other backend rather than nothing.
            matches = alternatives
        if matches:
            latest = max(matches, key=lambda p: (p / "status.json").stat().st_mtime_ns)
            self._select_run(latest.name)
        else:
            self.selected_run = None
            self.display_workers = 1
            self._load_scene(task, force=True, terrain=self.terrain_choice.value)
            self._live()
            self.status.content = "No recorded policy yet. Open Training to start."
            self.notice.content = "Scene preview"

    def _preview_terrain(self):
        if self.selected_run is None:
            self._load_scene(self.loaded_task, terrain=self.terrain_choice.value)

    def _start(self):
        config_type = MjlabTrainConfig if self.backend.value == "mjlab" else TrainConfig
        config = config_type(
            **(
                {"device": self.device.value} if config_type is MjlabTrainConfig else {}
            ),
            task=TASK_LABELS[self.task_choice.value],
            terrain=self.terrain_choice.value
            if TASK_LABELS[self.task_choice.value] == "go2_velocity"
            else "flat",
            workers=int(self.worker_count.value),
            total_timesteps=int(self.budget.value),
            seed=int(self.seed.value),
        )
        run = self.supervisor.start(config)
        self._refresh_runs()
        self._select_run(run.name)
        self.notice.content = (
            "mjlab starting; recorded worlds appear after initialization."
            if isinstance(config, MjlabTrainConfig)
            else "Spawn requested; worker PIDs appear when initialization finishes."
        )

    def _select_run(self, name):
        run = self.supervisor.root / name
        if not run.is_dir():
            return
        payload = read_json(run / "config.json") or read_json(run / "request.json")
        if payload is None:
            return
        config = parse_config(payload)
        self.selected_run = run
        self.display_workers = (
            min(4, config.workers)
            if isinstance(config, MjlabTrainConfig)
            else config.workers
        )
        self.task_choice.value = next(
            label for label, task in TASK_LABELS.items() if task == config.task
        )
        self.terrain_choice.visible = config.task == "go2_velocity"
        self.terrain_choice.value = config.terrain
        self.backend.value = "mjlab" if isinstance(config, MjlabTrainConfig) else "SB3"
        self._update_backend_ui()
        self.device.visible = isinstance(config, MjlabTrainConfig)
        if isinstance(config, MjlabTrainConfig):
            self.device.value = config.device
        self.run_choice.value = name
        self._load_scene(config.task, force=True, terrain=config.terrain)
        self.worker_choice.options = tuple(str(i) for i in range(self.display_workers))
        if self.worker_choice.value not in self.worker_choice.options:
            self.worker_choice.value = "0"
        self._live()

    def _command(self, command):
        if self.selected_run is None:
            raise ValueError("Select an experiment first")
        status = read_json(self.selected_run / "status.json", {})
        if status.get("state") in {"completed", "failed", "stopped"}:
            raise ValueError(
                "This run has ended. Continue from a checkpoint to train again."
            )
        self.supervisor.command(self.selected_run, command)
        self.notice.content = f"Process command requested: {command}"

    def _checkpoint_path(self):
        if self.selected_run is None or self.checkpoint.value == "None":
            raise ValueError("Select a saved checkpoint first")
        return (
            self.selected_run / self.checkpoint.value
            if self._native_run()
            else self.selected_run / "checkpoints" / self.checkpoint.value
        )

    def _resume_checkpoint(self):
        checkpoint = self._checkpoint_path()
        original = load_config(self.selected_run / "config.json")
        config = replace(
            original,
            workers=int(self.worker_count.value),
            total_timesteps=int(self.budget.value),
        )
        run = self.supervisor.start(config, resume=checkpoint)
        self._refresh_runs()
        self._select_run(run.name)
        self.notice.content = (
            "Resumed policy in a new experiment; the original run remains intact."
        )

    def _evaluate(self):
        checkpoint = self._checkpoint_path()
        if self._native_run():
            self.comparison_job = self.supervisor.compare(checkpoint)
            self.comparison_opened = False
            self.cancel_comparison.visible = True
            self.notice.content = (
                "Comparing scripted and policy rollouts on three held-out seeds."
            )
            return
        run = self.supervisor.evaluate(checkpoint)
        self._refresh_runs()
        # Config is written after the evaluator imports its dependencies.
        self.pending_selection = run.name
        self.notice.content = f"Evaluating on three new seeds: {run.name}"

    def _stop_comparison(self):
        comparison = getattr(self, "comparison_job", None)
        if comparison:
            self.supervisor.command(comparison, "stop")

    def _live(self):
        self.replay = None
        self.latest_stamp = None
        if self.overview is not None:
            self.overview_stamps = [None] * 4
        if self.viewer is not None:
            for trajectory in self.viewer.trajectories:
                trajectory.reset()
        self.players = {}
        self.progress.value = 0
        self.notice.content = (
            "Buffered rollout · 1× simulation time · waiting for a completed episode; switches only at episode boundaries"
            if self.stream_mode.value.startswith("Smooth")
            else "Live snapshots · unpaced training state; may skip frames"
        )

    def _replay(self):
        if self.overview is not None:
            self.view_mode.value = "Selected worker"
            self._load_scene(self.loaded_task, force=True)
        path = self.episode_paths.get(self.episode_choice.value)
        if path is None:
            raise ValueError("Choose a recorded episode")
        with np.load(path, allow_pickle=False) as archive:
            self.replay = {key: archive[key].copy() for key in archive.files}
        for trajectory in self.viewer.trajectories:
            trajectory.reset()
        self.replay_started = self.clock.value
        self.notice.content = f"Replaying {self.episode_choice.value}"

    def _apply_frame(self, frame):
        apply_frame(self.sim.model, self.sim.data, frame)
        self.viewer.update(self.sim.data)
        if "contact_force" in frame:
            self.viewer.forces.update_vectors(
                frame["contact_pos"].reshape(-1, 3),
                frame["contact_force"].reshape(-1, 3),
            )
        if self.loaded_task == "go2_velocity":
            target = self.sim.data.body(self.base_body).xpos.copy()
            if self._camera_target is not None and self.follow_camera.value:
                delta = target - self._camera_target
                for client in self.server.get_clients().values():
                    look_at = np.asarray(client.camera.look_at).copy()
                    client.camera.position = np.asarray(client.camera.position) + delta
                    client.camera.look_at = look_at + delta
            self._camera_target = target
        targets = frame["targets"] if self.loaded_task != "go2_velocity" else []
        for i, handle in enumerate(self.goal_handles):
            handle.visible = i < len(targets)
            if handle.visible:
                handle.position = targets[i]

    def _refresh_frame(self):
        self.clock.advance(time.monotonic(), float(self.replay_speed.value))
        if self.clock.paused:
            return
        if self.replay is None and self.stream_mode.value.startswith("Smooth"):
            self._refresh_rollouts()
            return
        if self.overview is not None:
            if self.selected_run is None:
                return
            for worker, data in enumerate(self.overview_data):
                directory = self.selected_run / "workers" / str(worker)
                path = directory / "latest.npz"
                try:
                    stamp = path.stat().st_mtime_ns
                    if stamp == self.overview_stamps[worker]:
                        continue
                    with np.load(path, allow_pickle=False) as archive:
                        frame = dict(archive)
                    apply_frame(self.sim.model, data, frame)
                    state = read_json(directory / "state.json", {})
                    self.overview.update(
                        worker,
                        data,
                        f"Worker {worker} · PID {state.get('pid', '—')} · R {state.get('return', 0):.1f}",
                        targets=frame.get("targets"),
                    )
                    if "contact_force" in frame:
                        self.overview_forces[worker].update_vectors(
                            frame["contact_pos"].reshape(-1, 3),
                            frame["contact_force"].reshape(-1, 3),
                        )
                    self.overview_stamps[worker] = stamp
                except FileNotFoundError:
                    continue
            return
        if self.replay is not None:
            times = self.replay["time"]
            position = times[0] + self.clock.value - self.replay_started
            if position > times[-1] + 0.6 and self.loop_rollout.value:
                self.replay_started = self.clock.value
                position = times[0]
            self._apply_frame(sample_trace(self.sim.model, self.replay, position))
            self.progress.value = float(
                np.clip(
                    (position - times[0])
                    / max(float(times[-1] - times[0]), 0.001)
                    * 100,
                    0,
                    100,
                )
            )
            if position >= times[-1]:
                self.notice.content = "Episode complete"
        elif self.selected_run is not None:
            path = (
                self.selected_run / "workers" / self.worker_choice.value / "latest.npz"
            )
            try:
                stamp = path.stat().st_mtime_ns
                if stamp == self.latest_stamp:
                    return
                with np.load(path, allow_pickle=False) as archive:
                    self._apply_frame(dict(archive))
                self.latest_stamp = stamp
            except FileNotFoundError:
                return

    def _refresh_rollouts(self):
        if self.selected_run is None:
            return
        workers = (
            range(len(self.overview_data))
            if self.overview is not None
            else (int(self.worker_choice.value),)
        )
        for worker in workers:
            player = self.players.setdefault(worker, RolloutPlayer(self.sim.model))
            directory = self.selected_run / "workers" / str(worker)
            frame = player.sample(
                directory, self.clock.value, loop=self.loop_rollout.value
            )
            if frame is None:
                continue
            duration = float(player.trace["time"][-1])
            elapsed = float(frame["time"])
            label = f"Worker {worker} · episode {int(player.path.stem.split('-')[-1])} · {elapsed:.1f} / {duration:.1f}s"
            self.progress.value = min(100.0, elapsed / max(duration, 0.001) * 100)
            if self.overview is None:
                self._apply_frame(frame)
                self.notice.content = label + (
                    " · end of episode" if player.finished else ""
                )
            else:
                data = self.overview_data[worker]
                apply_frame(self.sim.model, data, frame)
                self.overview.update(worker, data, label, targets=frame.get("targets"))
                if "contact_force" in frame:
                    self.overview_forces[worker].update_vectors(
                        frame["contact_pos"].reshape(-1, 3),
                        frame["contact_force"].reshape(-1, 3),
                    )

    def _refresh_runs(self):
        runs = sorted(
            [
                p.name
                for p in self.supervisor.root.iterdir()
                if p.is_dir()
                and (p / "status.json").exists()
                and ((p / "config.json").exists() or (p / "request.json").exists())
            ],
            reverse=True,
        )
        options = tuple(runs) or ("No runs",)
        if self.run_choice.options != options:
            self.run_choice.options = options
        pending = getattr(self, "pending_selection", None)
        if pending and (self.supervisor.root / pending / "config.json").exists():
            self._select_run(pending)
            self.pending_selection = None

    def _refresh_status(self):
        self.supervisor.poll()
        self._refresh_runs()
        comparison = getattr(self, "comparison_job", None)
        if comparison:
            comparison_status = read_json(comparison / "status.json", {})
            self.cancel_comparison.visible = comparison_status.get("state") not in {
                "completed",
                "stopped",
                "failed",
            }
            if comparison_status.get("state") == "stopped":
                self.comparison_link.content = (
                    "Comparison stopped. Partial traces retained."
                )
            if comparison_status.get("state") == "completed":
                if not self.comparison_opened:
                    self.supervisor.open_comparison(comparison)
                    self.comparison_opened = True
                viewer = read_json(comparison / "viewer.json", {})
                if viewer.get("port"):
                    self.comparison_link.content = f"[Open same-seed comparison](http://localhost:{viewer['port']}) · scripted / policy"
            elif comparison_status.get("state") == "failed":
                self.notice.content = f"**Comparison failed:** {comparison_status.get('error', 'See process.log')}"
        if self.selected_run is None:
            return
        run = self.selected_run
        state = read_json(run / "status.json", {})
        if (
            self._native_run()
            and (run / "scene/scene.xml").exists()
            and self.loaded_model_path != run / "scene/scene.xml"
        ):
            self._load_scene(self.loaded_task, force=True)
        mean = state.get("mean_return")
        reward = f"{mean:.2f}" if mean is not None else "—"
        self.status.content = f"**{'mjlab / RSL-RL' if self._native_run() else 'SB3'} PPO · {state.get('state', 'starting')}** · {state.get('run_steps', 0):,} steps · R {reward}"
        if state.get("error"):
            self.notice.content = f"**Process error:** {state['error']}"
        records = []
        rows = [
            "| Worker | PID | State | Episode | Return |",
            "| --- | --- | --- | --- | --- |",
        ]
        for directory in sorted((run / "workers").glob("*")):
            worker = read_json(directory / "state.json", {})
            if (
                worker.get("pid")
                and worker.get("state") != "closed"
                and not psutil.pid_exists(worker["pid"])
            ):
                worker["state"] = "exited"
            if state.get("state") == "paused" and worker.get("state") not in {
                "closed",
                "exited",
            }:
                worker["state"] = "paused"
            rows.append(
                f"| {directory.name} | {worker.get('pid', '—')} | {worker.get('state', 'starting')} | {worker.get('episode', '—')} | {worker.get('return', 0):.2f} |"
            )
            for record in read_jsonl(directory / "episodes.jsonl")[-100:]:
                records.append((record, directory / record["trace"]))
        self.worker_table.content = "\n".join(rows)
        self.metrics.content = "\n\n".join(
            f"**{k}:** {v:.5g}" for k, v in state.get("metrics", {}).items()
        ) or (
            "RSL-RL losses: TensorBoard in this run directory. Worlds share one process."
            if self._native_run()
            else "PPO metrics appear after the first policy update."
        )
        checkpoints = (
            tuple(p.name for p in sorted(run.glob("*.pt")))
            if self._native_run()
            else tuple(p.name for p in sorted((run / "checkpoints").glob("*.zip")))
        )
        checkpoints = checkpoints or ("None",)
        if self.checkpoint.options != checkpoints:
            self.checkpoint.options = checkpoints
            self.checkpoint.value = (
                "final.pt"
                if "final.pt" in checkpoints
                else "final.zip"
                if "final.zip" in checkpoints
                else checkpoints[-1]
            )
        if self._native_run() and records:
            recorded_return = np.mean([record["return"] for record, _ in records[-20:]])
            self.status.content = f"**mjlab / RSL-RL PPO · {state.get('state', 'starting')}** · {state.get('run_steps', 0):,} steps · recorded R {recorded_return:.2f}"
        records.sort(key=lambda item: item[0]["finished_at"])
        records = records[-200:]
        self.episode_paths = {
            f"w{r['worker']} · ep{r['episode']} · R={r['return']:.2f} · {r['state']}": path
            for r, path in reversed(records)
        }
        options = tuple(self.episode_paths) or ("None",)
        if self.episode_choice.options != options:
            self.episode_choice.options = options
        self.episode_table.content = "\n".join(
            ["| Worker/episode | Seed | Return | End |", "| --- | --- | --- | --- |"]
            + [
                f"| {r['worker']}/{r['episode']} | {r['seed']} | {r['return']:.2f} | {'success' if r['is_success'] else 'terminated' if r['terminated'] else r['state'] if r['state'] == 'interrupted' else 'time limit'} |"
                for r, _ in reversed(records[-12:])
            ]
        )
        if records:
            self.plot.data = (
                np.arange(len(records), dtype=float),
                np.array([r["return"] for r, _ in records]),
            )

    def run(self):
        last_status = 0.0
        try:
            while True:
                start = time.monotonic()
                while True:
                    try:
                        command = self.commands.get_nowait()
                    except Empty:
                        break
                    try:
                        command()
                    except (ValueError, TypeError, OSError) as error:
                        self.notice.content = f"**Cannot perform action:** {error}"
                if start - last_status > 0.5:
                    self._refresh_status()
                    last_status = start
                self._refresh_frame()
                time.sleep(max(0.0, 1 / 30 - (time.monotonic() - start)))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        self.supervisor.close()
        if self.viewer:
            self.viewer.close()
        self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    ControlRoom(args.runs, args.host, args.port).run()


if __name__ == "__main__":
    main()
