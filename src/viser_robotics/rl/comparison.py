"""Synchronized Viser replay of same-seed controller and policy evaluations."""

import argparse
import time
from pathlib import Path
from threading import RLock

import mujoco
import numpy as np
import viser

from ..overlays import ForceOverlay
from .overview import ParallelOverview
from .playback import sample_trace
from .rgb import ReplayCameras
from .storage import read_json, write_json


def apply_frame(model, data, frame):
    for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "mocap_pos", "mocap_quat"):
        if name in frame:
            getattr(data, name)[:] = frame[name]
    data.time = float(frame["time"])
    mujoco.mj_forward(model, data)


class ComparisonViewer:
    def __init__(self, directory: Path, port=8091):
        self._lock = RLock()
        self.directory = Path(directory)
        self.report = read_json(self.directory / "comparison.json")
        if not self.report:
            raise ValueError("Missing comparison.json; run mjlab evaluation first")
        self.seeds = sorted({item["seed"] for item in self.report["reports"]})
        first = (
            self.directory / self.report["reports"][0]["directory"] / "scene/scene.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(first))
        self.data = [mujoco.MjData(self.model) for _ in range(2)]
        self.server = viser.ViserServer(port=port, label="ALOHA · Policy comparison")
        self.overview = ParallelOverview(self.server, self.model, count=2)
        self.server.on_client_connect(self.overview.focus)
        self.forces = [
            ForceOverlay(self.server, self.model, namespace=f"/agents/{i}/forces")
            for i in range(2)
        ]
        self.heading = self.server.gui.add_markdown("**Same-seed comparison · mjlab**")
        self.seed = self.server.gui.add_dropdown(
            "Seed",
            options=[str(s) for s in self.seeds],
            initial_value=str(self.seeds[0]),
        )
        self.play = self.server.gui.add_checkbox("Play", initial_value=True)
        self.timeline = self.server.gui.add_slider(
            "Time (s)", min=0, max=1, step=0.01, initial_value=0.0
        )
        self.restart = self.server.gui.add_button("Restart")
        self.jump = self.server.gui.add_dropdown(
            "Event", options=["Select event"], initial_value="Select event"
        )
        self.status = self.server.gui.add_markdown("Loading")
        self.rgb = ReplayCameras(self.server, self.model)
        with self.server.gui.add_folder("Details", expand_by_default=False):
            self.metrics = self.server.gui.add_markdown("")
            self.show_rgb = self.server.gui.add_checkbox(
                "Replay RGB",
                initial_value=self.rgb.available,
                visible=self.rgb.available,
            )
            self.speed = self.server.gui.add_dropdown(
                "Speed", options=["0.5×", "1×", "2×"], initial_value="1×"
            )
            self.show_force = self.server.gui.add_checkbox(
                "Contact forces", initial_value=True
            )
            self.server.gui.add_markdown(
                "Recorded strongest contact per object. Orange arrows: 2 cm/N, capped at 25 cm. Events describe measurements; lowering can be intentional placement."
            )
        self.position = 0.0
        self.previous = time.monotonic()
        self._load(self.seeds[0])
        self.seed.on_update(lambda _: self._load(int(self.seed.value)))
        self.timeline.on_update(self._seek)
        self.restart.on_click(lambda _: self.seek(0.0))
        self.jump.on_update(self._jump)

    def _load(self, seed):
        with self._lock:
            self.items = sorted(
                [item for item in self.report["reports"] if item["seed"] == seed],
                key=lambda item: item["controller"] == "policy",
            )
            if len(self.items) != 2:
                raise ValueError("Each seed requires scripted and policy traces")
            self.traces = []
            for item in self.items:
                with np.load(
                    self.directory / item["directory"] / "workers/0" / item["trace"],
                    allow_pickle=False,
                ) as archive:
                    self.traces.append(dict(archive))
            self.duration = max(float(trace["time"][-1]) for trace in self.traces)
            self.timeline.max = max(self.duration, 0.01)
            self.events = {}
            for item in self.items:
                for event in item["events"]:
                    label = (
                        f"{item['controller']} · {event['kind']} · {event['time']:.2f}s"
                    )
                    self.events[label] = event["time"]
            self.jump.options = ["Select event", *self.events]
            self.jump.value = "Select event"
            self.metrics.content = "\n\n".join(
                f"**{item['controller'].title()}** · {'Success' if item['is_success'] else 'Not successful'}\n\n"
                f"Return {item['return']:.3f} · final error {item['final_goal_error_m'] * 100:.1f} cm\n\n"
                f"Peak force {item['max_recorded_contact_force_n']:.2f} N · penetration {item['max_recorded_penetration_m'] * 1000:.2f} mm"
                for item in self.items
            )
            self.seek(0.0)

    def seek(self, position):
        with self._lock:
            self.rgb.invalidate()
            self.position = float(position)
            self.previous = time.monotonic()
            for trails, trace in zip(self.overview.trails, self.traces, strict=True):
                end = np.searchsorted(trace["time"], self.position, side="right")
                for arm, trail in enumerate(trails):
                    if "eef_pos" in trace:
                        trail.set_history(trace["eef_pos"][:end, arm])
                    else:
                        trail.reset()
            self.render()

    def _seek(self, event):
        if event.client is not None:
            self.play.value = False
            self.seek(self.timeline.value)

    def _jump(self, _):
        with self._lock:
            if self.jump.value in self.events:
                self.play.value = False
                self.seek(self.events[self.jump.value])

    def render(self):
        with self._lock:
            states = []
            for i, (item, trace) in enumerate(
                zip(self.items, self.traces, strict=True)
            ):
                frame = sample_trace(self.model, trace, self.position)
                apply_frame(self.model, self.data[i], frame)
                index = min(
                    int(np.searchsorted(trace["time"], self.position)),
                    len(trace["time"]) - 1,
                )
                phase = item["phases"][min(max(index - 1, 0), len(item["phases"]) - 1)]
                ended = self.position >= float(trace["time"][-1])
                outcome = (
                    "Success"
                    if item["is_success"]
                    else "Timeout"
                    if item["truncated"]
                    else "Terminated"
                )
                label = f"{item['controller'].title()} · {outcome if ended else phase}"
                self.overview.update(i, self.data[i], label, frame["targets"])
                self.forces[i].enabled = self.show_force.value
                self.forces[i].update_vectors(
                    frame["contact_pos"].reshape(-1, 3),
                    frame["contact_force"].reshape(-1, 3),
                )
                states.append(label)
            self.rgb.enabled = self.show_rgb.value
            self.rgb.update(self.data)
            self.timeline.value = round(self.position, 2)
            self.status.content = (
                f"{self.position:.2f} / {self.duration:.2f} s\n\n" + "\n\n".join(states)
            )

    def tick(self, now):
        with self._lock:
            elapsed = max(0.0, now - self.previous)
            self.previous = now
            if self.play.value:
                self.position = min(
                    self.duration,
                    self.position + elapsed * float(self.speed.value[:-1]),
                )
                if self.position >= self.duration:
                    self.play.value = False
            self.render()

    def run(self):
        try:
            while True:
                self.tick(time.monotonic())
                time.sleep(1 / 30)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        self.rgb.close()
        self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--announce",
        action="store_true",
        help="Write the bound port for the owning dashboard",
    )
    args = parser.parse_args()
    viewer = ComparisonViewer(args.directory, args.port)
    if args.announce:
        write_json(args.directory / "viewer.json", {"port": viewer.server.get_port()})
    viewer.run()


if __name__ == "__main__":
    main()
