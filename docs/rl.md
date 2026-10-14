# RL experiments and control room

MuJoCo + Viser 씬을 실제 에피소드 환경으로 사용하고, PPO 학습과 병렬 worker를 관제합니다.
알고리즘 구현은 Stable-Baselines3, 환경 규약은 Gymnasium을 사용합니다.
이 디렉터리에는 실행 설정과 결과가 있고, Python 구현은 [`mujoco_viser_usage/rl`](../mujoco_viser_usage/rl)에 있습니다.

## 시작

```bash
uv sync --extra rl
uv run --extra rl mujoco-viser-rl --port 8090
```

http://127.0.0.1:8090 에서:

1. **Scene**에서 ALOHA pick/place, ALOHA reach, Go2 walk를 선택합니다. 최근 해당 실험을 자동으로 불러옵니다.
2. **Pause rollout / Restart / Focus**로 재생을 제어합니다. PPO 학습과 독립적입니다.
3. **Training**을 펼쳐 **Train new policy**를 누르면 실제 worker 프로세스가 생성됩니다.
4. **Training / Settings**에서 worker 수·step·seed·학습 중지/재개·checkpoint를 설정합니다.
5. **Training / Metrics**에 PID·return·PPO loss가 있습니다.
6. **View / Layout**에서 단일/병렬 보기를 선택합니다. ALOHA는 작업대에 맞춘 간격과 카메라를 사용합니다.
7. **History**에서 다른 실험·episode를 재생합니다. 완료 episode 우선, 기본 반복 재생입니다.

병렬 worker는 한 PPO 정책의 rollout을 수집하는 독립 환경입니다. 각자 MuJoCo model/data,
seed, 에피소드 상태, 기록 파일을 갖습니다. LLM agent나 로봇 간 협업 MARL을 구현한 것은 아닙니다.
CPU 학습이며, 기본 worker 수는 4개입니다. UI는 1/2/4/8개, 설정 파일은 최대 16개를 지원합니다.
관제 서버가 생성한 실험은 동시에 2개까지 실행할 수 있습니다.

## 환경

| Task | 관측 / 행동 | 보상 / 종료 |
| --- | --- | --- |
| `go2_velocity` | 관절 상태, 몸체 속도·중력 방향, 목표 속도, gait phase / trot 기준 관절각에 대한 12차원 residual | 목표 속도 추종·upright, lateral/angular/action 비용 / 3 m 전진 성공, 넘어짐·통로 이탈 실패, 시간 제한 |
| `aloha_reach` | 양팔 관절, tool·물체·목표 위치, 이전 행동 / 관절 목표 증분과 gripper opening 14차원 | 두 tool의 목표 접근, 거리 감소, 성공 보너스 / 두 tool이 각 목표 2.5 cm 이내, 물체 낙하, 시간 제한 |
| `aloha_pick_place` | 양팔·물체·목표 상태 / 관절 목표 증분과 gripper opening 14차원 | tool 접근, lift, 물체의 목표 접근 / 두 물체가 각 목표 2.5 cm 이내, 물체 낙하, 시간 제한 |

`aloha_reach` 목표는 관절 범위 안의 scratch FK에서 생성합니다. `go2_velocity`는
스크립트 trot 위에 RL 보정을 학습하는 residual-control 예제입니다. 처음부터 보행을
학습하는 설정과 구별됩니다. `aloha_pick_place`는 더 어려운 학습 환경이며, 짧은 기본 학습으로
집기 성공이나 정책 수렴이 보장되지는 않습니다. 샘플용 ALOHA 자동 pick/place는 별도의 IK 제어기입니다.

기본 제어 간격은 20 ms이며 내부 물리는 `mj_step`입니다. 물리 dt의 정수 배수만 허용합니다.
`terminated`는 성공·실패, `truncated`는 시간 제한으로 구분합니다. 종료 후에는 reset이 필요합니다.
reset에서만 초기화하고, step에서는 모터 목표만 설정합니다. ALOHA에는 팔 중력 보상을 적용합니다.

## CLI

```bash
uv run --extra rl mujoco-viser-train --config assets/rl/go2_velocity.json
uv run --extra rl mujoco-viser-train --config assets/rl/aloha_reach.json
uv run --extra rl mujoco-viser-train --config assets/rl/aloha_pick_place.json
```

학습 명령은 실제 결과 경로를 `RUN_DIR=...`로 출력합니다. 다음 `RUN`에는 그 경로를 넣습니다.

```bash
uv run --extra rl mujoco-viser-train --config assets/rl/go2_velocity.json \
  --resume RUN/checkpoints/final.zip --steps 16384
uv run --extra rl python -m mujoco_viser_usage.rl.evaluate \
  --checkpoint RUN/checkpoints/final.zip --run-dir runs/my-evaluation --episodes 3
```

재학습은 기존 결과를 덮어쓰지 않고 새 run을 만듭니다. 정책·optimizer 상태를 이어받고 환경은
새로 reset합니다. 평가에는 학습과 다른 seed와 deterministic action을 사용합니다.
`total_timesteps`는 환경 전체의 transition 수입니다. PPO rollout 크기(`workers × n_steps`)에
맞추므로 완료 시 요청 step 수를 조금 넘을 수 있습니다.

## 결과 구조

```text
rl/
├── configs/                       # task, seed, workers, episode/rollout 길이
└── runs/<timestamp-task-id>/       # Git 제외
    ├── config.json                # 실행에 사용한 설정
    ├── status.json                # PID, 상태, steps, PPO metrics, heartbeat
    ├── control.json               # run / pause / stop 요청
    ├── process.log                # 관제 서버가 실행한 프로세스 stdout/stderr
    ├── events.jsonl               # checkpoint 저장 이벤트
    ├── metrics.jsonl              # rollout별 학습 지표
    ├── evaluation.json            # 평가 실행의 return/success 결과
    ├── checkpoints/               # initial, step-*, final 정책 + optimizer
    └── workers/<worker-id>/
        ├── state.json             # PID, seed, episode, return, 상태
        ├── latest.npz             # 최근 qpos/qvel/ctrl/외력/time/target
        ├── episodes.jsonl         # seed, 길이, return, 성공, 종료 사유
        └── episode-*.npz          # reset부터 terminal까지 재생용 상태
```

snapshot과 checkpoint는 임시 파일을 완성한 후 atomic rename합니다. 각 worker가 자기 파일만
작성합니다. 완료된 에피소드와 학습 중단으로 끊긴 에피소드를 구분해 남깁니다.
강제 종료나 worker crash 직전의 에피소드는 `latest.npz`까지만 남을 수 있습니다.
관제 화면은 별도 모델을 사용하므로 학습 물리를 조작하지 않습니다. 기본 **Smooth rollouts (buffered)**는
완료된 episode를 1× 속도·30 Hz 보간으로 표시하고, episode 경계에서 최신 기록으로 전환합니다.
즉시 상태는 **Live snapshots**에서 볼 수 있습니다. 첫 episode가 끝나기 전에는 버퍼를 기다립니다.
Overview의 칸 배치는 화면 표시를 위한 좌표 이동입니다. 실제 worker 상태에는 적용하지 않습니다.

관제 서버에서 띄운 프로세스는 서버 종료 시 먼저 정상 중단을 요청하며, 응답하지 않는 경우
해당 서버가 소유한 프로세스 그룹을 정리합니다. 별도 터미널에서 실행한 학습은 관제 서버 종료로
중단하지 않습니다. **Stop**은 마지막 정책을 저장하며, 중단된 rollout은 새 PPO update에 쓰지 않습니다.

## 코드 경계

- `config.py`: 실험 설정과 검증
- `envs.py`: Gymnasium reset/step, 관측, 행동, 보상과 종료 조건
- `storage.py`: atomic 저장, worker별 에피소드·재생 데이터
- `train.py`: PPO, SubprocVecEnv, checkpoint, pause/resume/stop
- `evaluate.py`: 새 seed에서 저장된 정책 평가
- `supervisor.py`: 학습/평가 프로세스의 생성과 종료
- `dashboard.py`, `overview.py`: Viser 관제와 동시 worker 표시

## 검증

```bash
uv run --extra rl pytest -q
uv run --extra rl ruff check .
```

테스트는 두 분야에서 실제 worker 프로세스 2개를 생성하고 학습 전후 파라미터 변화,
checkpoint 재학습·평가, pause/resume/stop 후 자식 종료, seed 재현성과 에피소드 기록을 검사합니다.
짧은 실행의 loss와 weight update 검증은 최종 task 성능 검증과 다릅니다.

API 참고: [Gymnasium environment 규약](https://gymnasium.farama.org/api/env/),
[SB3 병렬 환경](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html),
[SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).


Go2 지형은 **Training / Terrain** 또는 설정의 `"terrain": "mixed"`로 선택합니다.
`flat`, `ramps`, `steps`, `rough`, `mixed`를 지원하며 기존 설정은 `flat`입니다.
새 실행 기본은 `flat`; resume/evaluate는 저장된 원래 지형을 유지합니다.
