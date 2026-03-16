import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. APF 파라미터 ---
# 인력(목적지로 가는 힘)을 더 강하게 키워서 주저하지 않고 나아가게 설정
K_att = 5.0        
K_rep = 100.0      
RR = 2.0           # 장애물 인식 범위 (작을수록 좁은 길을 잘 통과함)
STEP_SIZE = 0.5    
MAX_ITERS = 1000   

# --- 2. APF 알고리즘 (탈출 로직 포함) ---
def get_potential_force(current_pos, goal_pos, obstacles):
    x, y = current_pos
    gx, gy = goal_pos
    
    # 인력 (Attraction)
    dist_goal = np.hypot(gx - x, gy - y)
    u_att = 0.5 * K_att * (dist_goal ** 2)
    f_att_x = K_att * (gx - x)
    f_att_y = K_att * (gy - y)

    # 척력 (Repulsion)
    f_rep_x = 0.0
    f_rep_y = 0.0
    u_rep = 0.0

    for (ox, oy, r) in obstacles:
        dist_obs = np.hypot(x - ox, y - oy)
        influence_dist = r + RR 

        if dist_obs < influence_dist:
            eff_dist = dist_obs - r 
            if eff_dist <= 0.1: eff_dist = 0.1

            # 척력 계산
            rep_mag = K_rep * ((1.0 / eff_dist) - (1.0 / RR)) * (1.0 / (eff_dist**2))
            obs_dir_x = (x - ox) / dist_obs
            obs_dir_y = (y - oy) / dist_obs
            
            f_rep_x += rep_mag * obs_dir_x
            f_rep_y += rep_mag * obs_dir_y
            
            u_rep += 0.5 * K_rep * ((1.0 / eff_dist) - (1.0 / RR))**2

    return (f_att_x + f_rep_x, f_att_y + f_rep_y), u_att + u_rep

def run_apf_simulation(start, goal, obstacles):
    path = [start]
    potentials = []
    
    current_pos = np.array(start, dtype=float)
    goal_pos = np.array(goal, dtype=float)
    
    stuck_counter = 0
    previous_pos = np.array(start, dtype=float)

    for i in range(MAX_ITERS):
        force, u_total = get_potential_force(current_pos, goal_pos, obstacles)
        potentials.append(u_total)

        dist_to_goal = np.hypot(goal_pos[0]-current_pos[0], goal_pos[1]-current_pos[1])
        if dist_to_goal < 1.0:
            print(f"목표 도달! ({i} step)")
            break

        force_mag = np.hypot(force[0], force[1])
        
        dx, dy = 0.0, 0.0
        if force_mag > 0:
            dx = (force[0] / force_mag) * STEP_SIZE
            dy = (force[1] / force_mag) * STEP_SIZE

        # Local Minima 탈출 로직
        dist_moved = np.hypot(current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1])
        
        if dist_moved < 0.05:
            stuck_counter += 1
        else:
            stuck_counter = 0
        
        previous_pos = current_pos.copy()

        if stuck_counter > 15: # 갇힘 감지
            # 랜덤 방향으로 튕겨내기
            current_pos[0] += np.random.uniform(-1, 1) * 2.0
            current_pos[1] += np.random.uniform(-1, 1) * 2.0
            stuck_counter = 5
        else:
            current_pos[0] += dx
            current_pos[1] += dy
        
        # 맵 밖으로 나가는 것 방지
        current_pos[0] = np.clip(current_pos[0], 0, 50)
        current_pos[1] = np.clip(current_pos[1], 0, 50)

        path.append(tuple(current_pos))

    return path, potentials

# --- 3. 환경 설정 (장애물 위치 수정됨!) ---
obstacles = [
    (8, 15, 6),   # <--- 여기를 수정했습니다 (왼쪽으로 이동)
    (30, 10, 7), 
    (25, 35, 8)
]
start = (5, 5)
end = (45, 45)

print("시뮬레이션 시작...")
path_data, potential_data = run_apf_simulation(start, end, obstacles)

# --- 4. 시각화 (VS Code 용) ---
if path_data:
    skip_step = 1 
    plot_path = path_data[::skip_step]
    plot_potential = potential_data[::skip_step]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 왼쪽: 맵
    ax1.set_title("APF Navigation (Path Widened)")
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 50)
    ax1.set_aspect('equal')
    ax1.grid(True)

    for (ox, oy, r) in obstacles:
        circle = plt.Circle((ox, oy), r, color='blue', alpha=0.5)
        ax1.add_patch(circle)
        dashed = plt.Circle((ox, oy), r + RR, color='blue', fill=False, linestyle='--', alpha=0.3)
        ax1.add_patch(dashed)

    ax1.plot(end[0], end[1], 'ro', markersize=10, label='Goal')
    drone_dot, = ax1.plot([], [], 'go', markersize=10, label='Drone')
    path_line, = ax1.plot([], [], 'k-', linewidth=1, alpha=0.5)
    ax1.legend(loc='upper left')

    # 오른쪽: 에너지 그래프
    ax2.set_title("Potential Energy")
    ax2.set_xlim(0, len(plot_potential))
    ax2.set_ylim(0, max(plot_potential) * 1.1)
    ax2.grid(True)
    energy_line, = ax2.plot([], [], 'r-', linewidth=2)

    def init():
        drone_dot.set_data([], [])
        path_line.set_data([], [])
        energy_line.set_data([], [])
        return drone_dot, path_line, energy_line

    def update(frame):
        current_p = plot_path[:frame+1]
        x_data = [p[0] for p in current_p]
        y_data = [p[1] for p in current_p]
        
        drone_dot.set_data([x_data[-1]], [y_data[-1]])
        path_line.set_data(x_data, y_data)
        
        current_e = plot_potential[:frame+1]
        energy_line.set_data(range(len(current_e)), current_e)
        return drone_dot, path_line, energy_line

    # repeat=False 추가
    ani = FuncAnimation(fig, update, frames=len(plot_path), init_func=init, blit=True, interval=30, repeat=False)
    
    # 로컬 창 띄우기
    plt.show()
else:
    print("경로 생성 실패")