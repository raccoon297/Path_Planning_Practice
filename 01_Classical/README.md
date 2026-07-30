# Classical Path Planning Algorithms

> A*, Artificial Potential Field(APF), Rapidly-exploring Random Tree(RRT), D* Lite를 동일한 2차원 환경에서 구현하고 비교한 프로젝트다.

## 1. 프로젝트 개요

본 프로젝트는 격자 탐색, 포텐셜 필드, 샘플링, 증분 탐색으로 대표되는 네 가지 고전 경로 계획 방식을 하나의 공통 환경에서 비교한다. 정적 환경에서는 경로 길이와 계산 결과를 비교하고, 동적 환경에서는 D* Lite가 새 장애물을 감지한 뒤 기존 탐색 정보를 재사용해 경로를 갱신하는 과정을 확인한다.

특정 논문의 실험을 그대로 재현한 것이 아니라, 각 알고리즘의 핵심 개념을 학습하고 비교할 수 있도록 공통 시나리오와 평가 방식에 맞게 단순화한 구현이다.

### 주요 구현 내용

- A*, APF, RRT, D* Lite의 핵심 탐색 구조 구현
- 격자 기반과 연속 공간 기반 알고리즘의 공통 시나리오 비교
- 경로 길이, 계산 시간, waypoint 수, 최소 여유 거리 기록
- 정적 탐색과 D* Lite 동적 재계획 과정 시각화

---

## 2. 문제 정의와 공통 설계

### 2.1 실험 환경

| 항목 | 설정 |
|---|---:|
| 지도 크기 | `50 × 50` |
| 시작점 | `(5, 5)` |
| 목표점 | `(45, 45)` |
| 장애물 표현 | 원형 장애물 |
| 안전 여유 | `0.5` |
| 격자 해상도 | `1.0` |
| RRT 대표 난수 시드 | `42` |

정적 실험에서는 모든 알고리즘이 같은 `Scenario` 객체를 입력받는다. A*와 D* Lite는 연속 지도를 occupancy grid로 변환하며, APF와 RRT는 연속 좌표와 선분 충돌 검사를 사용한다.

### 2.2 성공 조건과 평가 지표

경로는 시작점과 목표점을 연결하고 지도 경계와 장애물 안전영역을 침범하지 않을 때 성공으로 판정한다.

- `success`: 목표 도달 여부
- `planning_time_ms`: 경로 계산 시간
- `path_length`: 연속 waypoint 사이의 유클리드 거리 합
- `waypoint_count`: 반환된 경로점 수
- `minimum_clearance`: waypoint와 장애물 또는 지도 경계 사이의 최소 여유 거리

현재 `minimum_clearance`는 경로 선분 전체가 아니라 저장된 waypoint를 기준으로 계산한다.

지도와 알고리즘 설정은 `config/scenarios.py`에서 변경할 수 있다. 환경을 크게 변경하면 APF와 RRT의 `step_size`, 반복 횟수 등의 파라미터를 다시 조정해야 할 수 있다.

---

## 3. 구현 알고리즘

### 3.1 A*

A*는 시작점부터 현재 노드까지의 실제 누적 비용과 목표점까지의 추정 비용을 합해 탐색 우선순위를 결정한다.

$$
f(n)=g(n)+h(n)
$$

본 구현에서는 유클리드 거리를 휴리스틱으로 사용한다.

$$
h(n)=\sqrt{(x_g-x_n)^2+(y_g-y_n)^2}
$$

- Boolean occupancy grid와 8방향 이동을 사용한다.
- 직선 이동 비용은 `1`, 대각선 이동 비용은 $\sqrt{2}$다.
- 막힌 두 격자 사이를 대각선으로 통과하는 corner cutting을 금지한다.
- `heapq` 우선순위 큐와 `came_from` 관계로 경로를 복원한다.

### 3.2 Artificial Potential Field

APF는 목표점의 인력과 장애물의 척력을 합성해 다음 이동 방향을 결정한다.

$$
\mathbf{F}(q)=\mathbf{F}_{att}(q)+\mathbf{F}_{rep}(q)
$$

$$
\mathbf{F}_{att}(q)=k_{att}\frac{q_{target}-q}{\lVert q_{target}-q\rVert}
$$

- 연속 좌표 공간에서 작은 `step_size`만큼 반복 이동한다.
- 장애물 반지름과 safety margin을 합한 영역을 금지 영역으로 처리한다.
- 희망 방향이 충돌하면 각도를 보정해 실행 가능한 이동을 찾는다.
- 진행이 정체되면 임시 접선형 waypoint를 생성해 로컬 미니마를 완화한다.

따라서 본 구현은 기본 APF에 결정론적 로컬 미니마 복구 로직을 추가한 형태다.

### 3.3 Rapidly-exploring Random Tree

RRT는 탐색 공간에서 무작위 점을 샘플링하고, 기존 트리의 최근접 노드에서 샘플 방향으로 가지를 확장한다.

$$
q_{\mathrm{rand}}\sim U(X)
$$

$$
q_{\mathrm{near}}=\underset{q\in T}{\mathrm{arg\,min}}\;d(q,q_{\mathrm{rand}})
$$

- 일정 확률로 목표점을 직접 선택하는 goal bias를 적용한다.
- 최근접 노드에서 `step_size`만큼 확장한다.
- 새 노드뿐 아니라 부모 노드와 새 노드를 잇는 선분 전체를 검사한다.
- 부모 인덱스를 역추적해 최종 경로를 복원한다.
- 재현 가능한 비교를 위해 난수 시드를 고정한다.

<p align="center">
  <img src="results/static/rrt_animation.gif" width="620" alt="RRT 트리 성장 및 경로 추종 애니메이션">
</p>

본 구현은 기본 단일 트리 RRT이며, RRT*의 rewiring과 점근적 최적화는 포함하지 않는다.

### 3.4 D* Lite

D* Lite는 환경이 바뀌었을 때 전체 경로를 처음부터 다시 계산하지 않고 기존 탐색 정보를 재사용하는 증분 탐색 알고리즘이다.

$$
rhs(s)=\min_{s'\in Succ(s)}\left(c(s,s')+g(s')\right)
$$

- A*와 같은 8방향 occupancy grid를 사용한다.
- 각 정점의 `g`, `rhs`, 우선순위 키와 $k_m$을 유지한다.
- 새 장애물이 감지되면 변경된 격자와 인접 정점만 갱신한다.
- 기존 탐색 상태를 유지한 채 `compute_shortest_path`를 다시 수행한다.

---

## 4. 실험 결과

### 4.1 정적 환경 비교

<p align="center">
  <img src="results/static/path_comparison.png" width="900" alt="A*, APF, RRT, D* Lite 정적 경로 비교">
</p>

| 알고리즘 | 성공 | 경로 길이 | 계산 시간 | 경로 특성 |
|---|---:|---:|---:|---|
| A* | Yes | 63.01 | 2.96 ms | 비교적 짧고 각진 격자 경로 |
| APF | Yes | 79.89 | 4.19 ms | 부드럽지만 우회가 큰 연속 경로 |
| RRT | Yes | 67.15 | 7.46 ms | 무작위 트리에서 발견한 실행 가능한 경로 |
| D* Lite | Yes | 63.01 | 25.57 ms | 정적 환경에서는 A*와 유사한 격자 경로 |

계산 시간은 실행 환경에 따라 달라지며, 현재 결과는 하나의 시나리오와 고정된 RRT 난수 시드에서 얻은 대표 실행이다.

### 4.2 D* Lite 동적 재계획

동적 실험에서는 에이전트가 처음부터 알지 못하는 장애물을 설정한다. 에이전트가 이동 중 새 장애물을 감지하면 현재 위치를 새 시작점으로 설정하고 변경된 정점만 갱신해 경로를 복구한다.

<p align="center">
  <img src="results/dynamic/dstar_lite_replanning.gif" width="650" alt="D* Lite 동적 장애물 발견 및 재계획 애니메이션">
</p>

```text
초기 경로 생성
→ 센서 범위 이동
→ 숨겨진 장애물 감지
→ 변경 정점 갱신
→ 현재 위치에서 재계획
```

---

## 5. 실행 방법과 프로젝트 구조

### 5.1 설치 및 실행

```bash
python -m venv .venv
pip install -r requirements.txt
```

정적 비교:

```bash
python run_static_comparison.py
```

D* Lite 동적 재계획:

```bash
python run_dynamic_replanning.py
```

### 5.2 프로젝트 구조

```text
01_Classical/
├─ config/
│  └─ scenarios.py
├─ planners/
│  ├─ astar.py
│  ├─ apf.py
│  ├─ rrt.py
│  └─ dstar_lite.py
├─ utils/
│  ├─ collision.py
│  ├─ metrics.py
│  └─ visualization.py
├─ results/
│  ├─ static/
│  └─ dynamic/
├─ run_static_comparison.py
├─ run_dynamic_replanning.py
├─ requirements.txt
└─ README.md
```

---

## 6. 구현 범위와 한계

- 2차원 평면과 원형 장애물을 중심으로 구성하였다.
- APF에는 로컬 미니마 완화를 위한 보조 waypoint 로직을 추가하였다.
- RRT는 기본 단일 트리 방식이며 RRT* 최적화와 path smoothing을 포함하지 않는다.
- D* Lite는 장애물 추가에 따른 격자 변화를 처리하지만 센서 잡음과 위치 추정 오차는 모델링하지 않는다.
- 단일 시나리오의 실행 결과만으로 알고리즘의 일반적인 우열을 판단할 수 없다.

---

## 7. 참고문헌

본 프로젝트는 아래 연구의 핵심 알고리즘을 공통 시뮬레이션 환경에 맞게 재구성하였다. 특정 논문의 전체 실험을 그대로 재현한 것은 아니다.

1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). **A Formal Basis for the Heuristic Determination of Minimum Cost Paths.** *IEEE Transactions on Systems Science and Cybernetics, 4*(2), 100–107. https://doi.org/10.1109/TSSC.1968.300136
2. Khatib, O. (1986). **Real-Time Obstacle Avoidance for Manipulators and Mobile Robots.** *The International Journal of Robotics Research, 5*(1), 90–98. https://doi.org/10.1177/027836498600500106
3. LaValle, S. M. (1998). **Rapidly-exploring random trees: A new tool for path planning.** Technical Report TR 98-11, Iowa State University. https://www.lavalle.pl/papers/Lav98c.pdf
4. Koenig, S., & Likhachev, M. (2002). **D* Lite.** *Proceedings of the 18th AAAI Conference on Artificial Intelligence*, 476–483. https://ojs.aaai.org/index.php/AAAI/article/view/8035
