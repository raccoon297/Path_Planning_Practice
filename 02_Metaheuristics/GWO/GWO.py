import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import comb

# ==========================================
# 1. 시뮬레이션 환경 및 파라미터 설정
# ==========================================
MAP_SIZE = 300
OBS_RADIUS = 20
SAFETY_MARGIN = 5 
DRONE_MIN_DIST = 15     # 비행 중 드론 간 최소 유지 거리
SURROUND_RADIUS = 20    # 타겟을 포위하는 반경 (이 원주 위에 도착해야 함)
MIN_END_DIST = 30       # 도착 시 드론 간 최소 유지 거리 (서로 퍼지도록 유도)

obstacles = np.array([[200, 100], [196, 187], [100, 200], [100, 100]])
target_pos = np.array([250, 250])  

# 드론 3대 시작 위치
start_positions = np.array([[25, 25], [50, 25], [25, 50]])
drone_colors = ['#6A0DAD', '#FF00FF', '#4B0082']

# GWO 하이퍼파라미터
SearchAgents_no = 50 
Max_iter = 150
# 차원 수 변경: 3대 * (중간 제어점 2개 + 최종 도착점 1개) * x,y 2좌표 = 18차원
dim = 18 

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def bezier_curve(points, num_points=100):
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(n + 1):
        curve += points[i] * (comb(n, i) * (1 - t)**(n - i) * t**i)[:, None]
    return curve

def calc_fitness(wolf_position):
    penalty = 0
    paths = []
    end_points = []
    
    # 1. 개별 드론 경로 생성
    for i in range(3):
        # 6개의 값을 [중간점1, 중간점2, 도착점]으로 변환
        pts = wolf_position[i*6:(i+1)*6].reshape(3, 2)
        ctrls = pts[:2]
        end_pt = pts[2]
        end_points.append(end_pt)

        p = np.vstack([start_positions[i], ctrls, end_pt])
        path = bezier_curve(p, num_points=100)
        paths.append(path)

        # 맵 이탈 및 장애물 충돌 페널티
        if np.any(path < 0) or np.any(path > MAP_SIZE):
            penalty += 200000
        for obs in obstacles:
            dists = np.linalg.norm(path - obs, axis=1)
            if np.any(dists < (OBS_RADIUS + SAFETY_MARGIN)):
                penalty += 200000 

        # 경로 길이 페널티
        penalty += np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

        # [핵심] 포위망 형성 페널티: 타겟으로부터 SURROUND_RADIUS 만큼 떨어져야 함
        dist_to_target = np.linalg.norm(end_pt - target_pos)
        penalty += abs(dist_to_target - SURROUND_RADIUS) * 5000 

    # 2. 드론 간 상호작용 (충돌 회피 및 포위 대형 전개)
    for i in range(3):
        for j in range(i + 1, 3):
            # [핵심] 도착 지점에서 서로 너무 가까우면 엄청난 페널티 (포위망 분산 유도)
            end_dist = np.linalg.norm(end_points[i] - end_points[j])
            if end_dist < MIN_END_DIST:
                penalty += (MIN_END_DIST - end_dist) * 3000

            # 비행 중 충돌 회피 (연산 속도를 위해 10프레임 간격으로 검사)
            for t in range(0, 100, 10):
                dist_flight = np.linalg.norm(paths[i][t] - paths[j][t])
                if dist_flight < DRONE_MIN_DIST:
                    penalty += 2000

    return penalty

# ==========================================
# 3. GWO 메인 루프 
# ==========================================
Alpha_pos, Alpha_score = np.zeros(dim), float("inf")
Beta_pos, Beta_score = np.zeros(dim), float("inf")
Delta_pos, Delta_score = np.zeros(dim), float("inf")

# 에이전트 초기화 (목표 지점 주변으로 초기 도착점 분산)
X = np.zeros((SearchAgents_no, dim))
for i in range(SearchAgents_no):
    for d_idx in range(3):
        p_s = start_positions[d_idx]
        # 중간 제어점 초기화
        X[i, d_idx*6 + 0:d_idx*6 + 2] = p_s + (target_pos - p_s)*0.33 + np.random.randn(2)*20
        X[i, d_idx*6 + 2:d_idx*6 + 4] = p_s + (target_pos - p_s)*0.66 + np.random.randn(2)*20
        # 도착점 초기화 (타겟 주변 360도 무작위 흩뿌림)
        angle = np.random.rand() * 2 * np.pi
        X[i, d_idx*6 + 4:d_idx*6 + 6] = target_pos + np.array([np.cos(angle), np.sin(angle)]) * SURROUND_RADIUS

for l in range(Max_iter):
    for i in range(SearchAgents_no):
        X[i, :] = np.clip(X[i, :], 0, MAP_SIZE)
        fitness = calc_fitness(X[i, :])

        if fitness < Alpha_score:
            Alpha_score, Alpha_pos = fitness, X[i, :].copy()
        elif fitness < Beta_score:
            Beta_score, Beta_pos = fitness, X[i, :].copy()
        elif fitness < Delta_score:
            Delta_score, Delta_pos = fitness, X[i, :].copy()

    a = 2 - l * (2 / Max_iter) 

    for i in range(SearchAgents_no):
        for j in range(dim):
            r1, r2 = np.random.rand(), np.random.rand()
            A1, C1 = 2*a*r1-a, 2*r2
            D_alpha = abs(C1*Alpha_pos[j] - X[i, j])
            X1 = Alpha_pos[j] - A1*D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2, C2 = 2*a*r1-a, 2*r2
            D_beta = abs(C2*Beta_pos[j] - X[i, j])
            X2 = Beta_pos[j] - A2*D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3, C3 = 2*a*r1-a, 2*r2
            D_delta = abs(C3*Delta_pos[j] - X[i, j])
            X3 = Delta_pos[j] - A3*D_delta

            X[i, j] = (X1 + X2 + X3) / 3

# ==========================================
# 4. 시각화 (VS Code 로컬 환경용으로 수정됨)
# ==========================================
best_paths = []
for i in range(3):
    pts = Alpha_pos[i*6:(i+1)*6].reshape(3, 2)
    p = np.vstack([start_positions[i], pts]) # 시작점 + 제어점2개 + 도착점
    best_paths.append(bezier_curve(p, num_points=100))

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
ax.set_title("Organic Swarm Intelligence: GWO Path Planning - Target Surrounding mode")

for obs in obstacles:
    ax.add_patch(plt.Circle((obs[0], obs[1]), OBS_RADIUS, color='orange', alpha=0.8))

# 타겟 및 포위 반경 표시
ax.add_patch(plt.Circle((target_pos[0], target_pos[1]), SURROUND_RADIUS, color='green', fill=False, linestyle='--', alpha=0.5, zorder=4))
ax.add_patch(plt.Circle((target_pos[0], target_pos[1]), 5, color='red', zorder=5))

drone_scatters = [ax.plot([], [], 'o', color=drone_colors[i], markersize=7)[0] for i in range(3)]
drone_tails = [ax.plot([], [], '-', color=drone_colors[i], linewidth=1.5, alpha=0.4)[0] for i in range(3)]

def animate(frame):
    for i in range(3):
        drone_tails[i].set_data(best_paths[i][:frame+1, 0], best_paths[i][:frame+1, 1])
        drone_scatters[i].set_data([best_paths[i][frame, 0]], [best_paths[i][frame, 1]])
    return drone_scatters + drone_tails

# ani 객체를 변수에 할당해두어야 가비지 컬렉터에 의해 애니메이션이 멈추는 것을 방지합니다.
ani = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True, repeat=False)

# 로컬 창을 띄우는 함수 (프로그램이 여기서 대기하며 애니메이션을 보여줍니다)
plt.show()