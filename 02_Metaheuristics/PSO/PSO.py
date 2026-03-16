import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. 환경 및 파라미터 설정 ---
WIDTH, HEIGHT = 50, 50
TARGET = np.array([45, 45])   
NUM_DRONES = 3                
MAX_ITER = 200                

# PSO 파라미터 (장애물 회피를 위해 사회적 힘 c2를 약간 줄임)
w = 0.6     # 관성 (이전 속도 유지)
c1 = 1.2    # 인지 (내 경험)
c2 = 1.0    # 사회 (목적지/리더 추종) - 무조건 돌진하는 성향을 줄임
max_vel = 0.7  # 최대 속도

# 장애물 (x, y, 반지름)
OBSTACLES = [
    (15, 25, 6),
    (30, 15, 7),
    (25, 35, 6)
]

# --- 2. PSO 클래스 및 핵심 함수 ---
class Drone:
    def __init__(self, id):
        # 시작 위치를 랜덤하게 하되, 장애물과 겹치지 않게 (0~10 구역)
        self.position = np.random.rand(2) * 10 
        self.velocity = np.random.rand(2) 
        
        self.pbest_pos = self.position.copy()
        self.pbest_val = float('inf')
        
        self.path_history = [self.position.copy()]

def calculate_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

# [수정됨] 1. 장애물 회피 벡터 계산 (소프트 힘)
def get_avoidance_vector(position, obstacles):
    avoid_vec = np.zeros(2)
    for ox, oy, r in obstacles:
        obs_pos = np.array([ox, oy])
        dist = calculate_distance(position, obs_pos)
        
        # 감지 거리 (반지름 + 3.0)
        detection_range = r + 3.0
        
        if dist < detection_range:
            # 장애물 중심에서 드론 방향으로 밀어내는 벡터
            push_dir = position - obs_pos
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
            
            # 거리가 가까울수록 기하급수적으로 강한 힘 작용
            force = 10.0 * (1.0 / (dist - r + 0.5)) 
            avoid_vec += push_dir * force
            
    return avoid_vec

# [신규 추가] 2. 물리적 충돌 처리 (하드 리밋) - 이게 핵심입니다!
def resolve_collisions(drone, obstacles):
    for ox, oy, r in obstacles:
        obs_pos = np.array([ox, oy])
        dist = calculate_distance(drone.position, obs_pos)
        
        # 드론이 장애물 원 안으로 들어왔다면? (충돌 발생)
        if dist < r:
            # 1. 드론을 장애물 표면 밖으로 강제 이동 (Overlap 만큼 밀어냄)
            direction = drone.position - obs_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            # 반지름보다 아주 살짝(0.1) 바깥으로 위치 재설정
            drone.position = obs_pos + direction * (r + 0.1)
            
            # 2. 벽에 부딪혔으므로 속도를 반사시키거나 죽임
            # 여기서는 튕겨나가는 효과를 위해 속도 반전 (-0.5는 탄성 계수)
            drone.velocity = drone.velocity * -0.5

# --- 3. 메인 시뮬레이션 ---
drones = [Drone(i) for i in range(NUM_DRONES)]
global_best_pos = TARGET 

history_positions = [] 
convergence_data = []  

for t in range(MAX_ITER):
    current_positions = []
    total_dist = 0
    
    for drone in drones:
        # Fitness 계산
        dist_to_target = calculate_distance(drone.position, TARGET)
        total_dist += dist_to_target
        
        # Pbest 업데이트
        if dist_to_target < drone.pbest_val:
            drone.pbest_val = dist_to_target
            drone.pbest_pos = drone.position.copy()
            
        # 속도 업데이트
        r1 = np.random.rand(2)
        r2 = np.random.rand(2)
        
        inertial = w * drone.velocity
        cognitive = c1 * r1 * (drone.pbest_pos - drone.position)
        social = c2 * r2 * (global_best_pos - drone.position)
        
        # [1] 회피 힘 추가
        avoidance = get_avoidance_vector(drone.position, OBSTACLES)
        
        new_velocity = inertial + cognitive + social + avoidance
        
        # 속도 제한
        speed = np.linalg.norm(new_velocity)
        if speed > max_vel:
            new_velocity = (new_velocity / speed) * max_vel
            
        drone.velocity = new_velocity
        
        # 위치 업데이트
        drone.position += drone.velocity
        
        # [2] 충돌 해결 (위치 이동 후 장애물 안에 있으면 끄집어냄)
        resolve_collisions(drone, OBSTACLES)
        
        # 맵 이탈 방지
        drone.position = np.clip(drone.position, 0, WIDTH)
        
        # 기록
        drone.path_history.append(drone.position.copy())
        current_positions.append(drone.position.copy())
        
    history_positions.append(current_positions)
    convergence_data.append(total_dist / NUM_DRONES) 

# --- 4. 시각화 (VS Code 로컬 환경용으로 수정됨) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 맵 설정
ax1.set_xlim(0, WIDTH)
ax1.set_ylim(0, HEIGHT)
ax1.set_title(f"PSO Drone Swarm (Collisions Fixed)")
ax1.grid(True)
ax1.set_aspect('equal')

# 장애물 그리기
for ox, oy, r in OBSTACLES:
    circle = plt.Circle((ox, oy), r, color='blue', alpha=0.5)
    ax1.add_patch(circle)
    # 안전거리 점선 표시
    dashed = plt.Circle((ox, oy), r, color='blue', fill=False, linestyle='--', alpha=1.0)
    ax1.add_patch(dashed)

ax1.plot(TARGET[0], TARGET[1], 'ro', markersize=12, label='Target')

dots = []
trails = []
for i in range(NUM_DRONES):
    dot, = ax1.plot([], [], 'go', markersize=8, label='Drone' if i==0 else "")
    trail, = ax1.plot([], [], 'g-', alpha=0.3, linewidth=1)
    dots.append(dot)
    trails.append(trail)

ax1.legend(loc='upper left')

# 그래프 설정
ax2.set_xlim(0, MAX_ITER)
ax2.set_ylim(0, max(convergence_data) + 10)
ax2.set_title("Avg Distance to Target")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Distance")
ax2.grid(True)

line_graph, = ax2.plot([], [], 'k-', linewidth=2)

def init():
    for dot in dots: dot.set_data([], [])
    for trail in trails: trail.set_data([], [])
    line_graph.set_data([], [])
    return dots + trails + [line_graph]

def update(frame):
    current_pos_list = history_positions[frame]
    
    for i, drone_pos in enumerate(current_pos_list):
        dots[i].set_data([drone_pos[0]], [drone_pos[1]])
        
        past_path = np.array(drones[i].path_history[:frame+1])
        if len(past_path) > 0:
            trails[i].set_data(past_path[-30:, 0], past_path[-30:, 1]) # 꼬리 길이 조정

    line_graph.set_data(range(frame+1), convergence_data[:frame+1])
    return dots + trails + [line_graph]

# 앞서 말씀해주신 대로 한 번만 재생되도록 repeat=False 옵션을 추가했습니다.
ani = FuncAnimation(fig, update, frames=len(history_positions), init_func=init, blit=True, interval=50, repeat=False)

# 로컬 창을 띄우는 함수
plt.show()