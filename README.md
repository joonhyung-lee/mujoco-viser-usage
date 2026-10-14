# mujoco-viser-usage

MuJoCo 물리 시뮬레이션과 Viser 브라우저 뷰어를 결합한 로보틱스 실험 프로젝트입니다.
`mujoco-robotics-usage`의 parser 사용 방식과 `mjviser`의 씬/루프 분리를 참고했습니다.

## mjlab 기반 학습·rollout 비교

ALOHA 작업을 mjlab 1.6.0 / MuJoCo-Warp / RSL-RL로 실행하는 별도 환경을 제공합니다.
같은 seed에서 scripted controller와 checkpoint를 비교하고, Viser에서 타임라인·EEF
궤적·기록된 접촉력을 동기 재생합니다. 설치·학습·평가 명령과 현재 한계는
[docs/mjlab.md](docs/mjlab.md)를 참고하세요. RL 대시보드의 `Training → Settings → Backend`에서 mjlab/SB3를 선택합니다.
mjlab은 ALOHA와 Go2 velocity(지형 포함)에 연결되어 있으며, 평가 후 같은 seed 비교 화면 링크를 제공합니다.

## 실행

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync
uv run mujoco-viser-usage
```

기본 씬은 **UR5e + RG2 pick & place**입니다.
http://127.0.0.1:8080 에서 **Run pick & place**를 누르면 자동 동작을 시작합니다.
**물리 스텝은 실행 직후 자동으로 시작**됩니다. `--paused`로 시작 시 일시정지할 수 있습니다.
`--demo`를 붙이면 해당 씬의 샘플 동작도 자동으로 시작합니다.
종료는 Ctrl+C입니다. 다른 XML과 포트도 지정할 수 있습니다.

```bash
uv run mujoco-viser-usage path/to/scene.xml --port 8081
```

원격 접속 시 `--host 0.0.0.0`을 지정합니다. 접속자들은 물리 상태와 제어값을 공유합니다.
카메라 초기화는 버튼을 누른 사용자의 화면에만 적용됩니다.

## 세 가지 씬 트리

```text
src/viser_robotics/scenes/
├── manipulation/   ur5e / franka / aloha
├── locomotion/     go1 / go2 / spot (Spot+Arm 모델 공유)
├── navigation/     husky / turtlebot3 / controller.py
├── maze.py         공통 격자 맵, 좌표 변환, 최단 경로
└── __init__.py     씬 카탈로그 및 XML 경로 해석
```

```bash
uv run mujoco-viser-usage --list
uv run mujoco-viser-usage manipulation/franka
uv run mujoco-viser-usage manipulation/aloha
uv run mujoco-viser-usage locomotion/go1
uv run mujoco-viser-usage locomotion/go2
uv run mujoco-viser-usage locomotion/spot
uv run mujoco-viser-usage locomotion/spot_arm
uv run mujoco-viser-usage navigation/husky
uv run mujoco-viser-usage navigation/turtlebot3
```

동시에 열려면 `--port 8081`처럼 서로 다른 포트를 사용합니다.

| 영역 | 현재 가능한 작업 |
| --- | --- |
| UR5e + RG2 | 기존 접촉 기반 자동 pick & place |
| Franka Panda | 테이블·자유 물체·목표 표시, 관절/그리퍼 수동 제어 |
| ALOHA | 두 팔의 순차 pick & place, 실제 접촉 기반 lift/배치, 양쪽 EEF 궤적 |
| Go2 | 첫 미로 통로에서 약 2 m trot 후 정지하는 주행 샘플 |
| Go1 / Spot / Spot+Arm | 충돌 미로, floating-base 물리, 초기 자세와 관절 위치 서보 |
| Husky / TurtleBot3 | **Run navigate maze**로 격자 최단 경로 추종, 실제 바퀴 속도 제어와 접촉으로 이동 |

Go2 샘플은 두 링크 다리 IK와 관절 서보를 사용하는 결정적 trot입니다. 전체 미로 경로 추종은
바퀴 로봇에 구현되어 있습니다. Go1·Spot의 보행 제어기와 Franka 자동 조작은 아직 없습니다.
Husky는 `mujoco-robotics-usage`의 실제 MJCF/STL과 URDF 관성을 사용합니다. 바퀴는 안정적인
원통 collision으로 접촉하며 1 ms 물리 스텝을 사용합니다. TurtleBot3는 단순 형상 모델입니다.
나머지 모델은 라이선스를 포함해 Menagerie에서 가져왔습니다.

미로는 2 m 셀로 구성했습니다. `maze.py`를 변경한 뒤 개발용 생성 스크립트를 실행하면
충돌 벽과 목표 표시를 갱신할 수 있습니다. 실행 자체는 인접 레포나 학습 데이터가 필요 없습니다.

```bash
uv run python scripts/build_scene_assets.py ../mujoco_menagerie
```

경로 계획은 네 방향 격자 BFS, 추종은 현재 위치·방향 오차를 사용하는 바퀴 속도 제어입니다.
로봇 위치를 매 프레임 덮어쓰지 않습니다. 목표 도착 시 모터 목표를 0으로 설정하고,
180초 시간 초과나 전복 시 중단합니다. 녹색 선은 계획 경로, 투명 궤적은 실제 tool/base 이동입니다.

`ocbench`의 환경/제어기 분리와 `ogbench`의 로봇/미로 분리를 참고했습니다.
벤치마크 API·데이터셋·성능 수치를 재현한 구현은 아닙니다. 자세한 출처는 `THIRD_PARTY.md`에 있습니다.

## 바로 실행하는 샘플

```bash
uv run mujoco-viser-usage manipulation/aloha --demo --port 8082
uv run mujoco-viser-usage locomotion/go2 --demo --port 8083
uv run mujoco-viser-usage navigation/husky --demo --port 8081
```

`examples/aloha_pick_place.py`, `examples/go2_walk.py`, `examples/husky_maze.py`에도
동일한 Python 사용 예제가 있습니다. ALOHA는 약 37초에 걸쳐 왼팔과 오른팔로 두 물체를 옮깁니다.
Go2는 약 2 m 전진 후 정지하고, Husky는 표시된 미로 경로를 끝까지 따라갑니다.
샘플이 끝나도 물리 스텝은 계속 실행됩니다. **Reset → Run ...**으로 처음부터 다시 실행합니다.

## RL 학습과 관제

```bash
uv sync --extra rl
uv run --extra rl mujoco-viser-rl --port 8090
```

**http://127.0.0.1:8090**에서 ALOHA reach/pick-place와 Go2 velocity 환경의 에피소드,
PPO 학습, 병렬 worker 생성, 체크포인트 재학습·평가와 궤적 재생을 관리합니다.
**View / Layout / Parallel overview**에서는 최대 4개 worker의 실제 상태를 함께 볼 수 있습니다.
설정·결과는 `rl/`에, 구현 코드는 `src/viser_robotics/rl/`에 있습니다.
[RL 실행·환경·기록 구조](rl/README.md)를 참고하세요.

## 패널

- 상단: 실행/정지, 시간·접촉 수, **Step**(일시정지 중 1스텝), **Reset**, 속도 선택
- **Joints**: 관절 목표 슬라이더와 숫자 직접 입력. 기본 UR5e는 도(°) 단위
- **View**: 카메라 초기화, 충돌 형상, 툴 좌표계, 상자 낙하 실험
- **State**: 관절별 실제 자세와 물리 timestep

목표값은 모터의 `ctrl`에 적용됩니다. 실제 자세는 중력·관성·접촉·토크 제한을 포함한
MuJoCo 계산 결과입니다. 일시정지 중에는 목표값만 바뀌며 실행 또는 Step으로 움직입니다.
Reset은 시간, 속도, 자세와 목표값을 초기화하고 일시정지합니다.

씬 규모에 맞춘 회색 미터 그리드, 방향광, 그림자, 다크 패널의 Isaac-inspired 스타일을 사용합니다.

## Pick & place

로봇은 base table 위에 고정되고, 앞쪽 object table의 반투명한 밝은 흰색 블록을
RG2로 집어 초록색 표시 안으로 옮깁니다. 한 사이클은 약 18초의 시뮬레이션 시간입니다.

접근 → 하강 → 닫기 → 들어 올리기 → 이동 → 내려놓기 → 열기 → 후퇴 순서이며
IK는 별도 scratch data에서 목표 관절각을 계산합니다. 실제 로봇은 모터와
팔의 bias-force 보상으로 움직이고, 물체는 손가락 접촉/마찰로만 이동합니다.
이동 중 물체 위치를 덮어쓰거나 weld로 붙이지 않습니다.

- **Run pick & place**: 자동 사이클 시작; 진행 단계가 버튼 아래에 표시됩니다.
- **Pause simulation / Start simulation**: 자동 동작 중 일시정지/재개
- **Reset**: 자동 동작 중단, 로봇·물체·목표값 초기화
- **Gripper (% open)**: 수동 열림 조작 (0 = 닫힘, 100 = 열림)
- 자동 실행 중에는 관절 슬라이더를 잠가 목표 충돌을 방지합니다.

이 예제는 포함된 블록, 테이블 배치와 RG2 모델에 맞춘 결정적 동작입니다.
임의 물체 인식이나 장애물 회피 경로 계획은 포함하지 않습니다.

```python
from viser_robotics import ViserParser
from viser_robotics.simulation import PICK_PLACE_SCENE

with ViserParser(PICK_PLACE_SCENE, pick_place=True) as robot:
    robot.run()
```

## Python 인터페이스

```python
from viser_robotics import ViserParser

with ViserParser() as robot:
    robot.run()
```

뷰어 없이 계산할 수 있습니다. Python 제어값은 MuJoCo 기본 단위입니다.

```python
from viser_robotics import ViserParser

with ViserParser(viewer=False) as robot:
    robot.set_control("shoulder_pan", -1.3)  # rad
    robot.step(nstep=500)                    # 기본 모델에서 1초
    q = robot.get_q()                       # 복사본
    position, rotation = robot.get_body_pose("wrist_3_link")
    robot.forward(q)                        # FK 갱신, 물리 시간 진행 없음
    robot.reset()
```

직접 루프를 구성하는 예시는 [examples/manual_loop.py](examples/manual_loop.py)에 있습니다.
직접 루프에서는 호출자가 물리 진행을 제어하며 패널의 실행/정지는 `run()` 모드에서 사용합니다.

## 구조

```text
src/viser_robotics/
  viser_parser.py    # 공개 진입점과 수명 관리
  simulation.py     # 모델/상태, 제어 입력, 물리 계산
  runtime.py        # 명령 큐, 실행 상태, 시간 누적
  kinematics.py     # scratch-data site IK
  pick_place.py     # 단계별 자동 동작과 성공 확인
  viewer.py         # geometry 변환, pose 갱신, 카메라
  controls.py       # 조작 패널과 표시 단위
  scene_style.py    # 시각 스타일
  app.py            # CLI
  scenes/manipulation/ur5e/      # 독립 실행용 MJCF, mesh, 원본 라이선스
```

설계 근거와 소유권 규칙은 [docs/architecture.md](docs/architecture.md)에 정리했습니다.

## 검증

```bash
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

## 현재 범위

기본 UR5e와 낙하 상자 외에 XML 로딩을 지원합니다. renderer는 mesh, plane,
box, sphere, ellipsoid, cylinder, capsule을 지원하며 텍스처를 재현하지 않습니다.
충돌 표시 토글은 Menagerie의 group 3 규칙을 사용합니다. 도 단위 변환은
unit-gear hinge position actuator에만 적용하고 다른 모터는 원래 제어값으로 표시합니다.
범용 조작 경로 계획은 아직 없습니다. 접촉력 오버레이와 RL 병렬 환경 관제는 지원합니다.

외부 모델의 출처와 라이선스: [THIRD_PARTY.md](THIRD_PARTY.md).

## 접촉 확인

**View → Contact points**에서 접촉점을, **State → Max contact penetration**에서
현재 프레임의 최대 침투 깊이를 볼 수 있습니다. 물리 진행은 `mj_step`(UR5e 기준 2 ms)이며,
IK는 모터 목표값만 계산합니다. 자동 경로의 침투 회귀 검사는 다음으로 실행합니다.

```bash
uv run python scripts/check_contacts.py
```

RG2 닫힘 관절 토크는 최대 0.35 N·m로 제한합니다(손끝 힘의 N 단위와 다릅니다).
현재 검증된 경로의 최대 침투는 약 0.058 mm입니다. 이는 soft contact의 수치적
침투이며 완전한 0을 보장하는 모델은 아닙니다. 임의 수동 조작에는 같은 상한을
보장하지 않습니다. RG2 내부 링크 간 자체 충돌은 필터링되어 있습니다.

## 시각화 오버레이

기본적으로 다음 표시가 켜져 있습니다. **View** 폴더에서 각각 전환할 수 있습니다.

- **Contact forces**: `mj_contactForce`의 접촉 법선·마찰 힘을 월드 좌표로 변환한
  주황색 화살표. 방향은 해당 접촉의 geom2에 작용하는 힘입니다.
  길이는 1 N당 2 cm, 최대 25 cm이며 0.02 N 이하를 생략하고 큰 힘 순으로 최대 32개를 표시합니다.
- **Ego RGB overlay**: 좌측 상단의 420px 패널에 손목 카메라 영상을 표시합니다.
  MuJoCo offscreen renderer가 실제 640×400 RGB 영상을 최대 10 Hz로 갱신합니다.
  MuJoCo와 Viser의 조명·재질 표현은 다를 수 있습니다. OpenGL 렌더링 환경이 필요합니다.
- **Tracked trajectory · 10 anchors**: 실제 grasp-site 이동 경로를 옅은 파란색
  반투명 튜브로 표시합니다(불투명도 28%). 누적 경로 길이에 균등하게 앵커 10개를
  배치하고 검정 테두리를 그립니다. 이동 전에는 궤적이 없고, Reset하면 지워집니다.
  최소 2mm 간격으로 수집한 최근 1,500개 표본을 유지하며 경로 메시를 약 7 Hz로 갱신합니다.

오버레이는 `overlays.py`에서 읽기 전용으로 구성되며 물리 상태를 수정하지 않습니다.


### 밝은 지형과 rollout 재생

Go2와 Husky는 기본 미로 외에 `ramps`, `steps`, `rough`, `mixed` 지형을 제공합니다.
지형은 모두 MuJoCo 접촉 geometry이며 Viser와 RGB 카메라 모두 밝은 회색 바닥을 사용합니다.

```bash
uv run mujoco-viser-usage navigation/husky_mixed --demo --port 8081
uv run mujoco-viser-usage locomotion/go2_mixed --port 8083
uv run --extra rl mujoco-viser-rl --port 8090
```

RL **Training / Terrain**에서 선택합니다. 새 실행 기본은 `flat`이며, 실험을 선택하면
해당 실험의 지형을 불러옵니다. 기존 설정 파일은 `flat`으로 해석합니다. 경사·요철을 해결하는 정책으로
수렴했음을 뜻하지는 않습니다.

**Smooth rollouts (buffered)**는 완료된 episode 하나를 1× 속도·30 Hz로 표시합니다.
기록 사이 자세는 quaternion을 포함해 보간하며, episode가 끝난 뒤 최신 기록으로 바뀝니다.
첫 episode가 끝날 때까지 기다리며, 학습이 빠르면 중간 episode는 표시하지 않습니다.
**Live snapshots**는 보간 없이 worker의 즉시 상태를 보여줍니다. History의 수동 재생도
동일한 보간을 사용합니다. 이 표시 기능은 학습의 `mj_step`이나 속도를 변경하지 않습니다.


### 간단한 rollout 화면

`Scene`에서 **ALOHA · pick & place**, **ALOHA · reach**, **Go2 · walk**를 선택합니다.
해당 작업의 최근 실험을 자동으로 불러옵니다. 기본 화면은 재생/정지, Restart, Focus와
episode 진행 표시만 노출합니다. View, Training, History는 기본 접힘 상태입니다.

- **View**: 단일/병렬 보기, worker, 속도, 반복, 추적 카메라, raw snapshot.
- **Training**: 새 PPO 학습. Settings 안에 worker 수, 학습량, seed, 학습 중지·재개, checkpoint.
- **History**: 다른 실험과 episode 재생.

재생 정지는 표시만 멈추고 PPO 학습은 계속합니다. Go2 카메라는 사용자 회전/줌을 유지하며
로봇 위치를 따라갑니다. ALOHA는 양팔·작업대를 가까이 보여주고 병렬 보기에서도 양손 궤적과
목표를 표시합니다. 완료된 episode를 우선 재생하며 새 기록이 없으면 반복합니다.
RL 보기에서 maze 벽만 숨깁니다. 물리 충돌은 그대로 유지되며 navigation 씬에서는 벽이 보입니다.
경사 메시의 바깥 면 방향, 요철 사이 간격, 불투명 maze 벽으로 시각적 겹침을 줄였습니다.
