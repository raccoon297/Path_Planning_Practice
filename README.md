# Path Planning Practice

> 고전 탐색, 메타휴리스틱, 강화학습, 다중 에이전트 최적화를 이용해 경로 계획 문제를 단계적으로 구현하고 비교한 프로젝트 모음이다.

## 1. 저장소 개요

본 저장소는 경로 계획 알고리즘의 핵심 원리와 구현 방식을 학습하기 위해 구성하였다. 각 프로젝트는 관련 선행연구의 문제 정의와 알고리즘 구조를 참고하되, 학습·비교·시각화가 가능한 규모의 공통 시뮬레이션으로 축소하고 재구성하였다.

단일 에이전트의 2차원 경로 탐색에서 시작해 전역 최적화, 3차원 강화학습 제어, 다중 에이전트의 시공간 공동 계획으로 문제 범위를 확장한다. 세부 환경 설정, 알고리즘 수식, 실험 결과와 실행 방법은 각 폴더의 `README.md`에 정리되어 있다.

## 2. 저장소 구성

| 폴더 | 주제 | 구현 알고리즘 및 내용 |
|---|---|---|
| [`01_Classical`](01_Classical/) | Classical Path Planning | A*, APF, RRT, D* Lite를 이용한 정적 경로 탐색과 동적 재계획 |
| [`02_Metaheuristics`](02_Metaheuristics/) | Metaheuristic Path Planning | ACO, GA, GWO, PSO를 이용한 단일 에이전트 경로 최적화 |
| [`03_Reinforcement_Learning`](03_Reinforcement_Learning/) | Reinforcement Learning Path Planning | DQN과 PPO를 이용한 3차원 도시 환경의 이산·연속 제어 비교 |
| [`04_Multi_Agent_Path_Planning`](04_Multi_Agent_Path_Planning/) | Multi-Agent Path Planning | ACO, GA, GWO, PSO를 이용한 세 에이전트의 경로와 출발 시각 공동 최적화 |

```text
Path_Planning_Practice/
├─ 01_Classical/
├─ 02_Metaheuristics/
├─ 03_Reinforcement_Learning/
├─ 04_Multi_Agent_Path_Planning/
└─ README.md
```
