"""Reproduce excessive penetration in the full physical pick/place trajectory."""

from collections import defaultdict

from mujoco_viser_usage import ViserParser
from mujoco_viser_usage.simulation import PICK_PLACE_SCENE

worst = defaultdict(lambda: (0.0, 0.0, ""))
with ViserParser(PICK_PLACE_SCENE, viewer=False, pick_place=True) as robot:
    robot.task.start()
    for _ in range(10500):
        robot.step()
        for contact in robot.data.contact:
            pair = tuple(
                sorted(
                    robot.model.geom(int(i)).name or str(i)
                    for i in (contact.geom1, contact.geom2)
                )
            )
            depth = max(0.0, -float(contact.dist))
            if depth > worst[pair][0]:
                worst[pair] = (depth, robot.data.time, robot.task.status)
        if not robot.task.active:
            break
    print("Task:", robot.task.status)
for pair, (depth, time, phase) in sorted(worst.items(), key=lambda item: -item[1][0]):
    print(f"{depth * 1000:8.3f} mm | {time:6.3f} s | {phase:20s} | {' / '.join(pair)}")
assert max(v[0] for v in worst.values()) < 0.001, "Penetration exceeds 1 mm"
