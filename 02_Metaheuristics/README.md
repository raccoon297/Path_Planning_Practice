# Metaheuristic Path Planning

> ACO, GA, GWO, PSO가 동일한 단일 드론 경로 계획 문제를 서로 다른 방식으로 탐색하는 과정을 구현하고 시각화한 프로젝트다.

## 1. 프로젝트 개요

본 프로젝트는 네 메타휴리스틱 알고리즘의 최종 순위를 결정하기보다, 동일한 지도와 목적함수에서 **후보 경로를 생성하고 개선하는 탐색 방식의 차이**를 이해하는 데 목적이 있다.

ACO는 페로몬을 축적하고, GA는 세대를 진화시키며, GWO는 Alpha·Beta·Delta를 따르고, PSO는 개인·집단 최적해를 참고한다. ant, chromosome, wolf, particle은 실제 드론이 아니라 하나의 경로 후보 또는 경로를 구성하는 가상 탐색 개체다.

<p align="center">
  <img src="results/single/path_comparison.png" alt="ACO, GA, GWO, PSO 최종 경로 비교" width="1000">
</p>

### 주요 구현 내용

- 네 메타휴리스틱의 고유한 탐색 메커니즘 구현
- 동일 시나리오와 목적함수를 이용한 단일 드론 경로 비교
- 탐색 population의 변화와 최종 경로를 GIF로 시각화
- 고정 난수 시드를 이용한 대표 실행 재현

---

## 2. 문제 정의와 공통 설계

### 2.1 실험 환경

크기 `100 × 100`인 연속 2차원 공간에서 시작점 `(5, 5)`부터 목표점 `(95, 95)`까지 이동하는 충돌 없는 경로를 탐색한다.

| 항목 | 설정 |
|---|---:|
| 지도 크기 | `100 × 100` |
| 시작점 | `(5, 5)` |
| 목표점 | `(95, 95)` |
| 정적 원형 장애물 | 6개 |
| 안전 여유 | `3.0` |
| GA/GWO/PSO waypoint | 5개 |
| 대표 난수 시드 | `42` |

실선 원은 실제 장애물을, 점선 원은 장애물 반지름에 safety margin을 더한 금지 영역을 나타낸다. 경로는 시작점과 목표점을 포함하고, 지도 경계·장애물·안전 여유 조건을 모두 만족할 때 성공으로 판정한다.

### 2.2 경로 표현

GA, GWO, PSO는 중간 waypoint 5개의 좌표를 하나의 10차원 실수 벡터로 표현한다.

```text
[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
```

```text
start → waypoint 1 → ... → waypoint 5 → goal
```

ACO는 동일한 연속 지도를 해상도 `2.5`의 `41 × 41` 격자로 변환하고 8방향 연결을 이용해 경로를 구성한다. 탐색 후에는 line-of-sight 단순화를 적용하고 연속 공간에서 다시 검증한다.

### 2.3 목적함수와 탐색 조건

목적함수는 후보 경로를 선택하기 위한 내부 비용이며 값이 작을수록 짧고 안전하며 부드러운 경로에 가깝다.

$$
J(P)=w_LJ_{length}+w_CJ_{collision}+w_DJ_{clearance}+w_SJ_{smoothness}+w_BJ_{boundary}
$$

| 항 | 의미 | 가중치 |
|---|---|---:|
| `length` | 전체 polyline 길이 | `1.0` |
| `collision` | 장애물과 교차한 선분 수 | `10,000.0` |
| `clearance` | safety margin 침범 | `100.0` |
| `smoothness` | 회전각 제곱합 | `5.0` |
| `boundary` | 지도 경계 이탈량 | `10,000.0` |

```text
Population / ants        : 80
Iterations / generations : 200
Candidate attempts       : 16,080
```

ACO는 격자에서 경로를 순차적으로 구성하므로 후보 하나의 계산 과정은 연속형 알고리즘과 다르다. 지도와 탐색 파라미터는 `config/scenario.py`와 실행 스크립트의 옵션에서 변경할 수 있다.

---

## 3. 구현 알고리즘

### 3.1 Ant Colony Optimization

ACO는 ant가 격자 경로를 확률적으로 구성하고, 우수한 경로의 간선에 pheromone을 축적하는 방식이다.

$$
p_{ij}\propto \tau_{ij}^{\alpha}\eta_{ij}^{\beta}
$$

- 회색 선은 현재 colony의 ant 경로다.
- 주황색은 pheromone 강도다.
- 파란색은 현재까지의 최적 경로다.
- pheromone 증발과 보강을 반복하며 좋은 corridor에 탐색이 집중된다.

<p align="center">
  <img src="results/single/aco_evolution.gif" alt="ACO pheromone-guided path reinforcement" width="850">
</p>

구현 파일: [`optimizers/aco.py`](optimizers/aco.py)

### 3.2 Genetic Algorithm

GA에서는 chromosome 하나가 전체 waypoint 경로 하나를 의미한다.

```text
fitness evaluation
→ tournament selection
→ waypoint crossover
→ Gaussian mutation
→ elite survival
```

- 보라색은 생성된 offspring이다.
- 주황색은 다음 세대로 보존되는 elite다.
- 초록색은 best-so-far 경로다.
- 좌표를 분리하지 않도록 `(x, y)` waypoint 블록 단위 교차를 사용한다.

<p align="center">
  <img src="results/single/ga_evolution.gif" alt="GA selection crossover mutation and elitism" width="850">
</p>

구현 파일: [`optimizers/ga.py`](optimizers/ga.py)

### 3.3 Grey Wolf Optimizer

GWO에서는 wolf 하나가 전체 waypoint 경로 하나를 나타낸다. 상위 세 후보인 Alpha, Beta, Delta가 나머지 wolf의 이동 방향을 결정한다.

$$
\mathbf{X}(t+1)=\frac{\mathbf{X}_1+\mathbf{X}_2+\mathbf{X}_3}{3}
$$

제어계수 `a`는 `2 → 0`으로 감소해 초반에는 넓게 탐색하고 후반에는 leader 주변의 exploitation을 강화한다.

<p align="center">
  <img src="results/single/gwo_evolution.gif" alt="GWO guided by Alpha Beta and Delta" width="850">
</p>

구현 파일: [`optimizers/gwo.py`](optimizers/gwo.py)

### 3.4 Particle Swarm Optimization

PSO에서는 particle 하나가 전체 waypoint 경로 하나를 의미한다. 각 particle은 자신의 최적 위치 `pbest`와 swarm의 최적 위치 `gbest`를 참고한다.

$$
\mathbf{v}_i(t+1)=w(t)\mathbf{v}_i(t)+c_1\mathbf{r}_1\odot(\mathbf{p}_i-\mathbf{x}_i)+c_2\mathbf{r}_2\odot(\mathbf{g}-\mathbf{x}_i)
$$

관성가중치 `w`는 `0.9 → 0.4`로 감소해 초기 탐색과 후기 수렴의 균형을 조절한다.

<p align="center">
  <img src="results/single/pso_evolution.gif" alt="PSO particles attracted by pbest and gbest" width="850">
</p>

구현 파일: [`optimizers/pso.py`](optimizers/pso.py)

---

## 4. 실험 결과

대표 난수 시드 `42`에서 네 알고리즘 모두 지도 경계와 장애물 안전 여유 조건을 만족하는 경로를 생성하였다.

| 알고리즘 | 성공 | 경로 길이 | 계산 시간* |
|---|---:|---:|---:|
| ACO | True | 131.50 | 18.09 s |
| GA | True | 132.35 | 4.42 s |
| GWO | True | 150.51 | 4.11 s |
| PSO | True | 130.83 | 4.11 s |

\* 계산 시간은 알고리즘 내부 CPU 시간이며 운영체제와 하드웨어 상태에 따라 달라질 수 있다.

PSO는 가장 짧은 경로를 반환했고 ACO와 GA도 유사한 길이의 경로에 도달했다. GWO는 다른 우회 영역으로 수렴해 더 긴 경로를 생성하였다. 이 결과는 하나의 대표 실행이며 알고리즘의 일반적인 우열을 의미하지 않는다.

---

## 5. 실행 방법과 프로젝트 구조

### 5.1 설치 및 실행

```bash
pip install -r requirements.txt
python run_single_comparison.py
```

다른 난수 시드 지정:

```bash
python run_single_comparison.py --seed 7
```

GIF 없이 정적 결과만 생성:

```bash
python run_single_comparison.py --skip-gifs
```

Matplotlib 창 표시:

```bash
python run_single_comparison.py --show
```

### 5.2 프로젝트 구조

```text
02_Metaheuristics/
├─ config/
│  └─ scenario.py
├─ optimizers/
│  ├─ aco.py
│  ├─ ga.py
│  ├─ gwo.py
│  └─ pso.py
├─ utils/
│  ├─ collision.py
│  ├─ metrics.py
│  ├─ objective.py
│  ├─ optimizer_worker.py
│  ├─ path_utils.py
│  ├─ reporting.py
│  └─ visualization.py
├─ results/
│  └─ single/
├─ run_single_comparison.py
├─ requirements.txt
└─ README.md
```

`utils/optimizer_worker.py`는 네 알고리즘을 독립된 Python 프로세스에서 실행하기 위한 내부 도구다.

---

## 6. 구현 범위와 한계

- 정적 2차원 환경의 단일 드론 global path planning만 다룬다.
- 속도, 가속도, 회전 반경과 같은 동역학 제약은 포함하지 않는다.
- GA, GWO, PSO는 고정된 waypoint 수를 사용한다.
- ACO는 격자 해상도에 따라 경로 품질과 계산 비용이 달라진다.
- 목적함수 가중치는 현재 시나리오를 기준으로 설정하였다.
- 확률적 메타휴리스틱은 전역 최적해를 보장하지 않으며, 대표 시드 하나의 결과만으로 일반적인 우열을 판단할 수 없다.

---

## 7. 참고문헌

본 프로젝트는 아래 문헌의 핵심 아이디어를 정적 2차원 경로 계획 문제에 맞게 재구성하였다. 특정 논문의 전체 실험을 그대로 재현한 것은 아니다.

1. M. Dorigo, V. Maniezzo, and A. Colorni, “Ant system: Optimization by a colony of cooperating agents,” *IEEE Transactions on Systems, Man, and Cybernetics, Part B*, vol. 26, no. 1, pp. 29–41, 1996. [DOI](https://doi.org/10.1109/3477.484436)
2. C. C. Hsu, R. Y. Hou, and W. Y. Wang, “Path planning for mobile robots based on improved ant colony optimization,” *2013 IEEE International Conference on Systems, Man, and Cybernetics*, pp. 2777–2782, 2013. [DOI](https://doi.org/10.1109/SMC.2013.474)
3. J. H. Holland, *Adaptation in Natural and Artificial Systems*. University of Michigan Press, 1975; MIT Press reprint, 1992.
4. I. Ashiru, C. Czarnecki, and T. Routen, “Characteristics of a genetic based approach to path planning for mobile robots,” *Journal of Network and Computer Applications*, vol. 19, no. 2, pp. 149–169, 1996. [DOI](https://doi.org/10.1006/jnca.1996.0012)
5. S. Mirjalili, S. M. Mirjalili, and A. Lewis, “Grey Wolf Optimizer,” *Advances in Engineering Software*, vol. 69, pp. 46–61, 2014. [DOI](https://doi.org/10.1016/j.advengsoft.2013.12.007)
6. S. Zhang, Y. Zhou, Z. Li, and W. Pan, “Grey wolf optimizer for unmanned combat aerial vehicle path planning,” *Advances in Engineering Software*, vol. 99, pp. 121–136, 2016. [DOI](https://doi.org/10.1016/j.advengsoft.2016.05.015)
7. J. Kennedy and R. Eberhart, “Particle swarm optimization,” *Proceedings of ICNN’95*, vol. 4, pp. 1942–1948, 1995. [DOI](https://doi.org/10.1109/ICNN.1995.488968)
8. Y. Shi and R. Eberhart, “A modified particle swarm optimizer,” *1998 IEEE International Conference on Evolutionary Computation*, pp. 69–73, 1998. [DOI](https://doi.org/10.1109/ICEC.1998.699146)
9. M. S. Alam, M. U. Rafique, and M. U. Khan, “Mobile robot path planning in static environments using particle swarm optimization,” 2020. [arXiv](https://arxiv.org/abs/2008.10000)
