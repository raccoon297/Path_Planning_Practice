import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev

# ==========================================
# 1. 환경 및 파라미터 설정
# ==========================================
MAP_SIZE = 300
GRID_RES = 10      # 연산 속도를 위해 맵을 10x10 단위 격자로 나눔
START = (20, 20)   # 왼쪽 아래 (드론 시작 위치)
TARGET = (280, 280) # 오른쪽 위 (목적지)

# 20 반경의 원형 장애물 5개 (x, y, radius)
OBSTACLES = [
    (100, 100, 20),
    (160, 90, 20),
    (140, 220, 20),
    (230, 160, 20),
    (220, 260, 20)
]
SAFETY_MARGIN = 15 # 스무딩 시 코너를 돌 때 깎이는 공간 확보용

# ==========================================
# 2. D* Lite 초기 탐색 로직 (Backward Search)
# ==========================================
def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def is_collision(x, y):
    if x < 0 or x > MAP_SIZE or y < 0 or y > MAP_SIZE:
        return True
    for ox, oy, r in OBSTACLES:
        if np.hypot(x - ox, y - oy) < (r + SAFETY_MARGIN):
            return True
    return False

def d_star_lite_initial_plan(start, goal):
    start_g = (start[0]//GRID_RES, start[1]//GRID_RES)
    goal_g = (goal[0]//GRID_RES, goal[1]//GRID_RES)
    
    open_set = []
    heapq.heappush(open_set, (0, goal_g))
    
    came_from = {}
    g_score = {goal_g: 0}
    
    movements = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == start_g:
            break
            
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            real_x, real_y = neighbor[0]*GRID_RES, neighbor[1]*GRID_RES
            
            if is_collision(real_x, real_y):
                continue
                
            cost = np.hypot(dx, dy)
            tentative_g = g_score[current] + cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + heuristic(neighbor, start_g)
                heapq.heappush(open_set, (f_score, neighbor))
                
    path = []
    current = start_g
    while current in came_from:
        path.append((current[0]*GRID_RES, current[1]*GRID_RES))
        current = came_from[current]
    path.append((goal_g[0]*GRID_RES, goal_g[1]*GRID_RES))
    
    return path

# ==========================================
# 3. 경로 스무딩 (B-Spline 기반 곡선화)
# ==========================================
print("D* Lite 격자 탐색 중...")
grid_path = d_star_lite_initial_plan(START, TARGET)

if not grid_path:
    raise ValueError("경로를 찾을 수 없습니다. 장애물이 길을 막고 있습니다.")

print("드론 비행 역학에 맞게 경로 스무딩 진행 중...")
x_coords = [p[0] for p in grid_path]
y_coords = [p[1] for p in grid_path]

tck, u = splprep([x_coords, y_coords], s=5.0) 
u_new = np.linspace(0, 1, 300) 
smooth_x, smooth_y = splev(u_new, tck)

smooth_path = list(zip(smooth_x, smooth_y))

# ==========================================
# 4. 애니메이션 및 시각화 (VS Code용)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- [왼쪽 맵] ---
ax1.set_title("D* Lite Path Planning (Smoothed for UAV)")
ax1.set_xlim(0, MAP_SIZE)
ax1.set_ylim(0, MAP_SIZE)
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':', alpha=0.6)

for ox, oy, r in OBSTACLES:
    circle = plt.Circle((ox, oy), r, color='gold', alpha=0.7, zorder=2)
    ax1.add_patch(circle)
    safe_zone = plt.Circle((ox, oy), r + SAFETY_MARGIN, color='gold', fill=False, linestyle='--', alpha=0.4)
    ax1.add_patch(safe_zone)

ax1.plot(TARGET[0], TARGET[1], 'ro', markersize=10, zorder=4, label='Target')
ax1.plot(START[0], START[1], 'o', color='purple', markersize=6, zorder=4, label='Start')

drone_dot, = ax1.plot([], [], 'o', color='purple', markersize=12, zorder=5, label='Drone')
trail_line, = ax1.plot([], [], '-', color='purple', linewidth=2, alpha=0.5, zorder=3)
ax1.legend(loc='upper left')

# --- [오른쪽 그래프] ---
ax2.set_title("Flight Distance Analysis")
ax2.set_xlim(0, len(smooth_path))
ax2.set_ylim(0, 600)
ax2.set_xlabel("Time (Frames)")
ax2.set_ylabel("Distance")
ax2.grid(True)

line_traveled, = ax2.plot([], [], 'b-', linewidth=2, label='Distance Traveled (↑)')
line_remaining, = ax2.plot([], [], 'r-', linewidth=2, label='Distance to Target (↓)')
ax2.legend(loc='center right')

# 전역 변수 초기화 (로컬 창 리사이즈 시 오류 방지용)
traveled_data = []
remaining_data = []
current_distance = 0.0

def init():
    global current_distance, traveled_data, remaining_data
    current_distance = 0.0
    traveled_data = []
    remaining_data = []
    
    drone_dot.set_data([], [])
    trail_line.set_data([], [])
    line_traveled.set_data([], [])
    line_remaining.set_data([], [])
    return drone_dot, trail_line, line_traveled, line_remaining

def update(frame):
    global current_distance, traveled_data, remaining_data
    
    current_pos = smooth_path[frame]
    drone_dot.set_data([current_pos[0]], [current_pos[1]])
    
    visited_x = smooth_x[:frame+1]
    visited_y = smooth_y[:frame+1]
    trail_line.set_data(visited_x, visited_y)
    
    if frame > 0:
        prev_pos = smooth_path[frame-1]
        step_dist = np.hypot(current_pos[0]-prev_pos[0], current_pos[1]-prev_pos[1])
        current_distance += step_dist
        
    dist_to_target = np.hypot(TARGET[0]-current_pos[0], TARGET[1]-current_pos[1])
    
    traveled_data.append(current_distance)
    remaining_data.append(dist_to_target)
    
    frames = range(len(traveled_data))
    line_traveled.set_data(frames, traveled_data)
    line_remaining.set_data(frames, remaining_data)
    
    return drone_dot, trail_line, line_traveled, line_remaining

print("시각화 렌더링 중...")
# repeat=False 추가하여 한 번만 재생
ani = FuncAnimation(fig, update, frames=len(smooth_path), init_func=init, blit=True, interval=10, repeat=False)

# 로컬 창을 띄우는 함수
plt.show()