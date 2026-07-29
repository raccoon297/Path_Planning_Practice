# 🚀 Path Planning Algorithms Collection (From Scratch)

이 Repository는 다양한 경로 계획(Path Planning) 알고리즘들을 외부 라이브러리 없이 From Scratch로 직접 구현한 프로젝트이다. 
고전적 방식부터 최신 강화학습 기법까지, 각 알고리즘의 수학적 원리와 구현 코드를 포함하고 있다.

---

## 📂 Repository Structure

- 01_Classical_Methods**: A*, D* Lite, APF, RRT* (Planned)
- 02_Metaheuristics**: PSO, GWO, GA (Planned), ACO (Planned)
- 03_Reinforcement_Learning**:
    - Single_Agent: DQN, PPO
    - Multi_Agent (MARL): MADDPG (Planned), MAPPO (Planned)

---

## 📑 Algorithm Summaries & Mathematical Foundations

### 1. Classical Methods (고전적 방식)

#### A*
노드 $n$의 우선순위를 결정하는 평가 함수를 기반으로 최단 경로를 탐색합니다.
$$f(n) = g(n) + h(n)$$
- $g(n)$: 시작 노드에서 현재 노드 $n$까지의 실제 비용
- $h(n)$: 현재 노드에서 목표 노드까지의 추정 비용 (Heuristic, 예: 유클리드 거리)

#### D Lite*
동적 환경에 대응하기 위해 두 개의 키($k$)를 사용하여 우선순위를 관리하고 효율적인 재계획(Replanning)을 수행합니다.
$$k_1(s) = \min(g(s), rhs(s)) + h(s_{start}, s) + k_m$$
$$k_2(s) = \min(g(s), rhs(s))$$
- $rhs(s)$: 한 단계 앞을 내다본(look-ahead) 추정값

#### APF (Artificial Potential Field)
로봇에 가해지는 총 에너지를 인력과 척력의 합으로 정의하여 물리적인 흐름에 따라 경로를 결정합니다.
$$U_{total} = U_{att} + U_{rep}$$
$$\vec{F} = -\nabla U_{total}$$
- $U_{att}$: 목표 지점으로 이끄는 에너지 (이차함수 형태)
- $U_{rep}$: 장애물에서 밀어내는 에너지 (거리의 역수 형태)

---

### 2. Metaheuristic (메타휴리스틱)

#### PSO (Particle Swarm Optimization)
입자들의 개별 최적해와 전체 최적해를 공유하며 속도와 위치를 업데이트합니다.
$$v_{i,t+1} = w v_{i,t} + c_1 r_1 (pBest_{i} - x_{i,t}) + c_2 r_2 (gBest - x_{i,t})$$
$$x_{i,t+1} = x_{i,t} + v_{i,t+1}$$
- $w$: 관성 가중치 / $c_1, c_2$: 인지 및 사회적 가속 계수

#### GWO (Grey Wolf Optimizer)
회색 늑대 무리의 서열($\alpha, \beta, \delta$)에 따른 사냥 방식을 모방하여 최적의 해를 추정합니다.
$$\vec{D} = |\vec{C} \cdot \vec{X}_p(t) - \vec{X}(t)|$$
$$\vec{X}(t+1) = \vec{X}_p(t) - \vec{A} \cdot \vec{D}$$
- $\vec{A}, \vec{C}$: 탐색을 조절하는 계수 벡터

---

### 3. Reinforcement Learning (강화학습)

#### DQN (Deep Q-Network)
벨만 최적 방정식(Bellman Equation)을 기반으로 신경망을 통해 Q-함수를 근사합니다.
$$Q(s, a) \approx R + \gamma \max_{a'} Q(s', a'; \theta^-)$$
$$L(\theta) = \mathbb{E} \left[ (Target - Q(s, a; \theta))^2 \right]$$
- $\gamma$: 할인율 (Discount factor) / $\theta^-$: 타겟 네트워크 파라미터

#### PPO (Proximal Policy Optimization)
급격한 정책 변화를 방지하는 Clipped Objective를 사용하여 안정적인 학습을 도모합니다.
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
- $r_t(\theta)$: 정책 확률 비율 / $\hat{A}_t$: 어드밴티지(Advantage) 추정치

---

### 4. Multi-Agent RL (MARL - In Progress)

#### MAPPO / MADDPG
중앙 집중형 학습 및 분산 실행(**CTDE**) 구조를 사용하여 여러 에이전트 간의 협력을 학습합니다.
$$L(\theta) = \sum_{i=1}^N \mathbb{E}_{s, a \sim \mathcal{D}} [ L_i(\theta_i) ]$$
- 에이전트는 로컬 관측값으로 행동하되, 학습 시에는 전역 상태 정보($S$)를 공유합니다.
