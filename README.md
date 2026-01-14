# Isaac Sim 4.5 Pick and Place Project

Franka Panda 로봇을 사용한 Pick and Place 데모 프로젝트입니다. NVIDIA Isaac Sim 4.5 환경에서 로봇이 DynamicCuboid 객체를 집어서 컨테이너에 놓는 작업을 수행합니다.

## 주요 성과

| 항목 | 결과 |
|------|------|
| 총 테스트 횟수 | 26회 |
| 최종 정확도 | **7.1mm** (0.0071m) |
| 성공률 | 100% (수정 후) |

## 시스템 요구사항

- **OS**: Ubuntu 22.04 LTS
- **GPU**: NVIDIA RTX 시리즈 (RTX 4090 권장)
- **RAM**: 32GB 이상 (64GB 권장)
- **Isaac Sim**: 4.5 버전

## 프로젝트 구조

```
ClaudeCode_PlanMode_PickAndPlace/
├── franka_ycb_pick_place.py      # 메인 GUI 스크립트
├── test_headless.py              # Headless 테스트 스크립트
├── work_log.txt                  # 개발 이력 및 문제 해결 기록
├── generate_technical_report.py  # 기술 문서 생성기
├── README.md                     # 이 파일
└── documents/                    # 문서 및 시각화 자료
    ├── *_project_presentation.html
    └── *_Technical_Report.docx
```

## 실행 방법

### GUI 모드 (시각화 포함)

```bash
cd ~/isaac_sim_4.5
./python.sh ~/ClaudeCode_PlanMode_PickAndPlace/franka_ycb_pick_place.py
```

> 창이 열리면 **'S' 키**를 눌러 Pick and Place 작업을 시작합니다.

### Headless 모드 (자동 테스트)

```bash
cd ~/isaac_sim_4.5
./python.sh ~/ClaudeCode_PlanMode_PickAndPlace/test_headless.py
```

## Pick and Place 10단계 Phase

| Phase | 동작 |
|-------|------|
| 0 | Moving above pick - 객체 위로 이동 |
| 1 | Lowering to grasp - 그립 위치로 하강 |
| 2 | Waiting settle - 안정화 대기 |
| 3 | Closing gripper - 그리퍼 닫기 |
| 4 | Lifting object - 객체 들어올리기 |
| 5 | Moving to place XY - 목표 위치로 수평 이동 |
| 6 | Lowering to place - 놓을 위치로 하강 |
| 7 | Opening gripper - 그리퍼 열기 |
| 8 | Lifting up - 그리퍼 상승 |
| 9 | Going home - 홈 위치로 복귀 |

## 핵심 해결책

26회의 테스트를 통해 발견한 핵심 해결책:

```python
# 1. 테이블 없이 GroundPlane만 사용 (RMPFlow 간섭 방지)
GroundPlane(prim_path="/World/GroundPlane", z_position=0)

# 2. DynamicCuboid 사용 (YCB 객체 대신 안정적인 큐브)
cube = DynamicCuboid(
    prim_path="/World/PickCube",
    position=np.array([0.3, 0.3, 0.3]),
    scale=np.array([0.0515, 0.0515, 0.0515]),
)

# 3. 그리퍼 초기화 - 가장 중요!
my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
my_world.reset()

# 4. Physics settle 후 실제 위치 사용
for i in range(50):
    my_world.step(render=False)
object_position = cube.get_local_pose()[0]
```

## 주요 발견 사항

1. **RMPFlow & 테이블 충돌**: 테이블이 있으면 RMPFlow가 잘못된 경로를 생성함
2. **로봇 도달 범위**: EE 최저 높이 z=0.287m, 객체는 z~0.27에 배치 필요
3. **그리퍼 초기화**: `set_default_state()` 호출이 성공의 핵심

## 문서

- `work_log.txt`: 전체 개발 이력 및 문제 해결 과정
- `documents/*_Technical_Report.docx`: 기술 보고서
- `documents/*_project_presentation.html`: 3D 시각화 프레젠테이션

## 참고 자료

- Isaac Sim 공식 예제: `~/isaac_sim_4.5/standalone_examples/api/isaacsim.robot.manipulators/franka_pick_up.py`
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)

## 프로젝트 기간

2026-01-14 ~ 2026-01-15
