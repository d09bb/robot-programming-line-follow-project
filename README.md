# Project 1 — 카메라 기반 자율주행 라인 추종 로봇

로봇프로그래밍 | 지능로봇공학과

## 1. 프로젝트 개요

ROS 2 Jazzy와 Gazebo Harmonic 환경에서 TurtleBot3 Waffle Pi가 카메라로 바닥의 검은색 주행선을 인식하고 자율적으로 추종하는 프로그램이다. Bird’s Eye View(BEV) 투영과 Wheel Odometry·IMU 기반 자세 추정을 이용해 관측한 주행선을 지도에 누적하며, Pure Pursuit 및 곡률 기반 속도 계획으로 직선, 급커브, 교차로, 말발굽 형태의 곡선, 경사로를 주행하고 T자 종점에서 정지한다.

- 기본 제어기: `reference_follower`
- 최대 직선 속도: 0.45 m/s
- 경사로 제한 속도: 0.32 m/s
- 제어기 리비전: `fb441252c1ed`
- 실행 환경: Ubuntu 24.04 기반 Docker, ROS 2 Jazzy, Gazebo Harmonic
- 주요 라이브러리: Python, OpenCV, NumPy, CvBridge

## 2. 실행 방법

### 2.1 Docker 실행 (권장)

Docker 데몬에 접근할 수 있는 Linux 호스트에서 실행한다. GUI 모드는 X11 또는 XWayland의 `DISPLAY`와 `xhost`가 필요하다. 최초 빌드 시 의존성 다운로드를 위한 인터넷 연결이 필요하다.

최종 제출 ZIP을 압축 해제하고 `Dockerfile`이 있는 `Project1` 폴더로 이동한다. 아래는 제출 파일명이 `Project1(1).zip`인 경우의 예시이다.

```bash
unzip 'Project1(1).zip'
cd Project1
bash scripts/run_docker.sh
```

실행 스크립트는 Dockerfile을 이용해 로컬 이미지 `line-follower:jazzy`를 빌드하고, ROS 2 환경과 Gazebo를 실행한다. ROS_DOMAIN_ID 및 GZ_PARTITION은 실행마다 설정된다. Gazebo와 센서 브리지가 시작되고 기본 제어기는 약 20초의 초기화 시간 동안 영상을 누적한 뒤, 유효한 센서 데이터가 있을 때 주행을 시작한다.

Gazebo 창 없이 실행하려면 다음 명령을 사용한다. Headless 모드도 카메라 영상 생성을 위한 렌더링 환경이 필요하다.

```bash
bash scripts/run_docker.sh --headless
```

종료하려면 실행 터미널에서 `Ctrl+C`를 누른다. 스크립트는 해당 컨테이너를 정리한다.

### 2.2 ROS 2 실행 명령

Docker 컨테이너 또는 패키지 빌드 및 환경 설정이 완료된 ROS 2 Jazzy 터미널에서는 다음 명령을 실행한다.

```bash
ros2 launch line_follower line_follow.launch.py
```

직접 실행 시 주행 기록을 저장하려면 다음과 같이 경로를 지정한다.

```bash
ros2 launch line_follower line_follow.launch.py capture_dir:=/tmp/line_captures
```

### 2.3 Docker 없이 로컬에서 실행 (선택)

Ubuntu 24.04 및 ROS 2 Jazzy가 이미 설치된 환경에서 `Project1` 폴더를 기준으로 실행한다.

```bash
sudo apt-get update
sudo apt-get install -y ros-jazzy-ros-gz ros-jazzy-cv-bridge \
  python3-opencv python3-colcon-common-extensions
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch line_follower line_follow.launch.py
```

### 2.4 실행 옵션

| 옵션 | 설명 |
|---|---|
| `bash scripts/run_docker.sh` | 기본 GUI 주행 |
| `bash scripts/run_docker.sh --headless` | GUI 없이 주행 |
| `PROJECT1_STRAIGHT_SPEED=0.30 bash scripts/run_docker.sh` | 기본 제어기의 최대 주행 속도 변경 (단위: m/s) |
| `PROJECT1_REFERENCE_CONTROLLER=false bash scripts/run_docker.sh` | 비교용 기존 제어기 실행 |
| `PROJECT1_RENDERER=hardware bash scripts/run_docker.sh` | 지원되는 환경에서 하드웨어 렌더링 시도 |

기본 제어기의 직접 launch 속도 파라미터는 `straight_speed`이며, `max_speed`는 비교용 기존 제어기에 전달된다. 최종 보고서의 실험은 기본 제어기 및 0.45 m/s 설정을 기준으로 한다.

Docker Hub 이미지 주소는 제출 환경에서 별도로 확인해야 한다. 위 실행 절차는 원격 프로젝트 이미지 없이 첨부 Dockerfile로 로컬 이미지를 빌드한다.

## 3. 프로그램 구성

```text
Project1/
├── README.md
├── Dockerfile
├── scripts/
│   ├── run_docker.sh
│   ├── entrypoint.sh
│   ├── compare_speeds.py
│   └── make_report.py
├── models/
├── worlds/
└── src/
    └── line_follower/
        ├── package.xml
        ├── setup.py
        ├── launch/
        │   └── line_follow.launch.py
        ├── models/
        ├── worlds/
        │   └── line_track.sdf
        └── line_follower/
            ├── reference_node.py
            ├── reference_tracker.py
            ├── node.py
            ├── vision.py
            ├── path.py
            ├── recovery.py
            ├── progress.py
            ├── finish.py
            ├── timing.py
            └── tracing.py
```

| 파일 | 주요 역할 |
|---|---|
| `reference_node.py` | 기본 ROS 2 노드, 센서 수신, 자세 추정, 제어 명령 발행 및 기록 |
| `reference_tracker.py` | BEV, 지도 누적, 경로 생성, 조향 및 속도 제어, 종점 처리 |
| `line_follow.launch.py` | Gazebo, 센서 브리지, 제어기 통합 실행 |
| `line_track.sdf`, `models/` | 주행 환경 및 로봇·트랙 모델 |
| `Dockerfile`, `scripts/run_docker.sh` | 의존성 설치, 패키지 빌드 및 컨테이너 실행 |
| `scripts/entrypoint.sh` | ROS 2 및 워크스페이스 환경 설정 |
| `scripts/compare_speeds.py`, `scripts/make_report.py` | 주행 비교 및 보고서 생성 보조 도구 |
| `node.py`, `vision.py`, `path.py` 등 | 선택 실행 가능한 기존 제어기 및 보조 모듈 |

## 4. ROS 2 토픽

| 토픽 | 메시지 타입 | 기능 |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/msg/Image` | 카메라 영상 입력 |
| `/odom` | `nav_msgs/msg/Odometry` | 전진 이동량 및 측정 속도 입력 |
| `/imu` | `sensor_msgs/msg/Imu` | 방향 및 기울기 입력 |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 선속도·각속도 출력 |
| `/line_follower/debug` | `sensor_msgs/msg/Image` | 주행 상태 및 명령 속도 표시 영상 |

Gazebo와 ROS 2 사이의 통신에는 `ros_gz_bridge` 및 `ros_gz_image`를 사용한다. 기본 제어기는 `/world/` 토픽·서비스나 Gazebo의 절대 위치 조회를 사용하지 않고 카메라, Odometry 및 IMU로 주행한다.

## 5. 주요 알고리즘

1. **영상 처리 및 BEV**: 카메라 영상을 640 × 480으로 통일하고 Grayscale로 변환한다. 최신 필터링된 Pitch를 반영해 바닥 좌표로 투영한 후 밝기 80 미만인 유효 픽셀을 주행선으로 검출한다.
2. **자세 추정**: Wheel Odometry의 전진 이동량과 IMU Yaw를 결합한다. 지도 누적에 사용하는 x, y, Yaw는 카메라 타임스탬프에 맞춰 보간한다.
3. **지도 누적 및 경로 생성**: 관측 정보를 0.005 m/cell 해상도의 14 m × 14 m 지도에 누적하고, 전방의 연속적인 주행 경로를 생성한다. 근거리 관측을 더 강하게 반영한다.
4. **Pure Pursuit**: 전방 목표점을 추종하며 각속도를 계산한다. 목표점 각도 오차의 절댓값이 0.7 rad를 초과하면 `ω = 1.5α`로 전환하며 각속도 및 각가속도를 제한한다.
5. **속도 제어**: 전방 곡률에 맞춰 미리 감속하고, 직선에서는 최대 0.45 m/s까지 가속한다. 경사 감지 시 속도 상한을 0.32 m/s로 제한한다. 회전 이후 직선 재가속 조건이 0.15초 유지되면 기본 가속도의 1.3배를 적용한다.
6. **복구 및 안전 정지**: 경로 유실 시 제한된 전진·회전 동작으로 재탐색한다. 시간이나 회전각 한도를 초과하면 정지한다. 센서 갱신이 0.5초를 초과해 중단되면 Watchdog이 0속도 명령을 발행한다.
7. **종점 제어**: T자 종점을 확정하면 진입 직선 방향을 기준축으로 고정하고 방향 오차와 횡방향 오차를 미세 조향으로 보정하며 감속·정지한다.

## 6. 실험 결과

보고서에 기록된 2026년 11월 20일의 동일 제어기 리비전(`fb441252c1ed`) 4회 반복 주행 결과이다.

| 회차 | 시작 시각 | 시뮬레이션 시간 | 실제 경과 시간 | 결과 |
|---|---|---:|---:|---|
| 1 | 16:39:24 | 74.30초 | 79.84초 | 완주 |
| 2 | 16:44:06 | 72.53초 | 77.49초 | 완주 |
| 3 | 16:45:58 | 72.93초 | 77.60초 | 완주 |
| 4 | 16:50:22 | 73.00초 | 77.81초 | 완주 |
| 평균 | — | 73.19초 | 78.18초 | 4/4 |

- 완주율: 100% (해당 조건의 4회 실험 기준)
- 최단 시뮬레이션 주행 시간: 72.53초
- 각 회차의 Odometry 기준 최고 속도: 0.45 m/s
- 복구 동작 기록: 0회
- 4차 실험 최종 누적 주행 거리: 약 15.80 m
- 4차 종점 접근 시작·정지 시점의 방향 차이: 약 0.42°

주행 시간은 최초 전진 명령을 발행한 프레임부터 종점 판정으로 0속도 명령을 발행한 프레임까지 측정했다. 시뮬레이션 시간은 두 카메라 메시지의 타임스탬프 차이, 실제 경과 시간은 단조 시계(Monotonic Clock) 기준이다. 평균은 반올림 전 원본 기록을 기준으로 보고서에 기재한 값이다.

위 결과는 기존 실험 기록에 근거하며 원본 로그는 현재 제출 ZIP에 포함되어 있지 않다. 동일 트랙 이외의 일반화 성능 및 실제 로봇 환경에서의 성능은 추가 검증이 필요하다.

## 7. 주행 확인 및 기록

기본 제어기가 준비되면 `TRACK_READY controller=fb441252c1ed`가 출력된다. 정상적으로 종점에 도달하면 `TRACK_FINISHED sim_elapsed=...s elapsed=...s`가 출력된다.

별도 ROS 2 터미널에서 토픽을 확인하는 경우, 실행 스크립트가 출력한 ROS_DOMAIN_ID와 동일한 값을 설정한 뒤 사용한다.

```bash
ros2 topic list
ros2 topic hz /camera/image_raw
ros2 topic echo /cmd_vel
```

Docker 실행 시 `captures/<실행 식별자>/`가 자동 생성되며 기본 제어기는 다음 데이터를 저장한다.

| 파일 | 내용 |
|---|---|
| `control.jsonl` | 프레임별 주행 상태, 속도, 거리, 제어기 리비전 등 |
| `camera_*.png` | 30프레임마다 저장하는 카메라 원본 영상 |
| `camera_*.json` | 해당 시점의 자세, 경로 및 제어 정보 |
| `terminal_stop.png` | 종점 정지 시점 카메라 영상 |
| `terminal_stop.json` | 종점 정지 상태 및 완주 시간 |

Gazebo 외부 시점의 전체 트랙 화면은 별도로 촬영한다.

## 8. 실행 문제 해결

- **Docker 접근 오류**: 호스트에서 `docker info`로 Docker 데몬과 사용 권한을 확인한다.
- **GUI가 표시되지 않는 경우**: `DISPLAY`, X11/XWayland 환경, 그래픽 드라이버를 확인한다. 실행 스크립트는 기본적으로 Mesa 소프트웨어 렌더링을 사용한다.
- **그래픽 문제 분리 검사**: `bash scripts/run_docker.sh --check-gl`로 OpenGL을, `bash scripts/run_docker.sh --empty-world`로 간단한 Gazebo 월드의 표시를 검사한다.
- **로봇이 출발하지 않는 경우**: 초기 약 20초의 대기 시간과 카메라·Odometry·IMU 수신 상태를 확인한다.

## 9. 기존 코드 활용 및 개선 사항

기본 지도 누적, 경로 추종 및 속도 계획 코어는 제공받은 `robot_project1.zip`의 `project1_memory_v2/line_tracker.py`를 기반으로 구성하였다. 해당 출처는 `reference_tracker.py`와 `reference_node.py`에도 명시되어 있다. 본 프로젝트에서는 ROS 2 실행 환경에 통합하고 회전 이후 직선 재가속 조건, 종점 진입축 기반 정지 제어, 실행별 데이터 기록 및 Docker 실행 환경을 구성하였다.

TurtleBot3 모델 관련 라이선스는 `src/line_follower/models/LICENSE.turtlebot3`를 참고한다. 기존 코드의 라이선스 표기는 원본 소스의 표기를 따른다.
