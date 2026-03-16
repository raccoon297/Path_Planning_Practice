import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev

# ==========================================
# 1. 환경 및 파라미터 설정
# ==========================================
MAP_SIZE = 300
GRID_RES = 10      
START = (20, 20)   
TARGET = (280, 280) 

# 기존에 알고 있던 정적 장애물 (노란색)
OBSTACLES = [
    (80, 150, 20),
    (160, 90, 20),
    (140, 220, 20),
    (230, 160, 20),
    (220, 260, 20)
]
SAFETY_MARGIN = 15 

# [신규 추가] 비행 중 갑자기 나타날 미확인 장애물 (주황색/빨간색)
DYNAMIC_OBSTACLE = (180, 180, 25) 
SENSOR_RANGE = 80  # 드론이 전방 장애물을 감지할 수 있는 센서 거리

# ==========================================
# 2. 탐색 알고리즘 로직
# ==========================================
def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def is_collision(x, y, current_obstacles):
    if x < 0 or x > MAP_SIZE or y < 0 or y > MAP_SIZE:
        return True
    for ox, oy, r in current_obstacles:
        if np.hypot(x - ox, y - oy) < (r + SAFETY_MARGIN):
            return True
    return False

# 경로 탐색 함수 (장애물 리스트를 동적으로 받음)
def compute_path(start, goal, current_obstacles):
    start_g = (int(start[0]//GRID_RES), int(start[1]//GRID_RES))
    goal_g = (int(goal[0]//GRID_RES), int(goal[1]//GRID_RES))
    
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
            
            if is_collision(real_x, real_y, current_obstacles):
                continue
                
            cost = np.hypot(dx, dy)
            tentative_g = g_score[current] + cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + heuristic(neighbor, start_g)
                heapq.heappush(open_set, (f_score, neighbor))
                
    if start_g not in came_from:
        return [] # 경로 없음
        
    path = [(start[0], start[1])]
    current = start_g
    while current in came_from:
        current = came_from[current]
        path.append((current[0]*GRID_RES, current[1]*GRID_RES))
    path[-1] = (goal[0], goal[1]) # 목적지 정확히 맞춤
    return path

# 스무딩 함수
def get_smoothed_path(grid_path):
    if len(grid_path) < 4:
        return grid_path # 점이 너무 적으면 그대로 반환
        
    x_coords = [p[0] for p in grid_path]
    y_coords = [p[1] for p in grid_path]
    
    # 중복점 제거 (스플라인 에러 방지)
    unique_x, unique_y = [x_coords[0]], [y_coords[0]]
    for i in range(1, len(x_coords)):
        if np.hypot(x_coords[i]-unique_x[-1], y_coords[i]-unique_y[-1]) > 0.1:
            unique_x.append(x_coords[i])
            unique_y.append(y_coords[i])
            
    if len(unique_x) < 4:
        return list(zip(unique_x, unique_y))
        
    tck, u = splprep([unique_x, unique_y], s=10.0) 
    u_new = np.linspace(0, 1, int(len(unique_x)*5)) # 촘촘하게 분할
    smooth_x, smooth_y = splev(u_new, tck)
    return list(zip(smooth_x, smooth_y))

# ==========================================
# 3. 비행 시뮬레이션 사전 연산 (Pre-calculation)
# ==========================================
print("비행 시뮬레이션 연산 중...")
current_obstacles = OBSTACLES.copy()
initial_grid_path = compute_path(START, TARGET, current_obstacles)
current_smooth_path = get_smoothed_path(initial_grid_path)

flight_history = []
dynamic_obstacle_detected = False
detected_frame = -1

# 드론이 길을 따라 이동하는 과정을 시뮬레이션합니다.
current_pos_idx = 0
while current_pos_idx < len(current_smooth_path):
    drone_pos = current_smooth_path[current_pos_idx]
    flight_history.append(drone_pos)
    
    # [핵심] 레이더 센서 감지 로직
    # 아직 감지되지 않았고, 미확인 장애물과의 거리가 센서 반경 안으로 들어왔다면?
    if not dynamic_obstacle_detected:
        dist_to_dyn = np.hypot(drone_pos[0] - DYNAMIC_OBSTACLE[0], drone_pos[1] - DYNAMIC_OBSTACLE[1])
        if dist_to_dyn < SENSOR_RANGE:
            print(f"⚠️ 경고: 전방 {SENSOR_RANGE}m 이내 미확인 장애물 감지! 경로를 재탐색합니다.")
            dynamic_obstacle_detected = True
            detected_frame = len(flight_history) - 1
            
            # 장애물 리스트 업데이트
            current_obstacles.append(DYNAMIC_OBSTACLE)
            
            # 현재 위치에서부터 목적지까지 다시 경로 탐색 (Replanning)
            new_grid_path = compute_path(drone_pos, TARGET, current_obstacles)
            if new_grid_path:
                current_smooth_path = get_smoothed_path(new_grid_path)
                current_pos_idx = 0 # 새로운 경로의 처음부터 다시 시작
                continue
            else:
                print("우회 경로가 없습니다!")
                break
                
    current_pos_idx += 1

# ==========================================
# 4. 애니메이션 및 시각화 (VS Code용)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- [왼쪽 맵] ---
ax1.set_title("D* Lite Dynamic Obstacle Avoidance")
ax1.set_xlim(0, MAP_SIZE)
ax1.set_ylim(0, MAP_SIZE)
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':', alpha=0.6)

# 정적 장애물 그리기
for ox, oy, r in OBSTACLES:
    circle = plt.Circle((ox, oy), r, color='gold', alpha=0.7, zorder=2)
    ax1.add_patch(circle)

# 목적지와 시작점
ax1.plot(TARGET[0], TARGET[1], 'ro', markersize=10, zorder=4, label='Target')
ax1.plot(START[0], START[1], 'o', color='purple', markersize=6, zorder=4, label='Start')

# 동적 장애물 (처음엔 안 보이다가 나중에 나타남)
dyn_circle = plt.Circle((DYNAMIC_OBSTACLE[0], DYNAMIC_OBSTACLE[1]), DYNAMIC_OBSTACLE[2], 
                        color='tomato', alpha=0.0, zorder=3, label='Dynamic Obstacle')
ax1.add_patch(dyn_circle)

# 센서 반경 표시용 (드론 주위의 옅은 원)
sensor_circle = plt.Circle((START[0], START[1]), SENSOR_RANGE, color='gray', alpha=0.1, zorder=1)
ax1.add_patch(sensor_circle)

drone_dot, = ax1.plot([], [], 'o', color='purple', markersize=12, zorder=5, label='Drone')
trail_line, = ax1.plot([], [], '-', color='purple', linewidth=2, alpha=0.5, zorder=3)
ax1.legend(loc='upper left')

# --- [오른쪽 그래프] ---
ax2.set_title("Flight Status")
ax2.set_xlim(0, len(flight_history))
ax2.set_ylim(0, 500)
ax2.set_xlabel("Time (Frames)")
ax2.set_ylabel("Distance")
ax2.grid(True)

line_traveled, = ax2.plot([], [], 'b-', linewidth=2, label='Traveled Dist')
line_remaining, = ax2.plot([], [], 'r-', linewidth=2, label='Remaining Dist')
ax2.legend(loc='center right')

# 전역 변수 초기화
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
    return drone_dot, trail_line, line_traveled, line_remaining, dyn_circle, sensor_circle

def update(frame):
    global current_distance, traveled_data, remaining_data
    current_pos = flight_history[frame]
    
    # 드론 위치 및 센서 반경 업데이트
    drone_dot.set_data([current_pos[0]], [current_pos[1]])
    sensor_circle.center = (current_pos[0], current_pos[1])
    
    # 미확인 장애물 등장 타이밍 처리
    if dynamic_obstacle_detected and frame >= detected_frame:
        dyn_circle.set_alpha(0.8) # 투명도 조절로 화면에 나타나게 함
    else:
        dyn_circle.set_alpha(0.0)
    
    # 꼬리 업데이트
    visited_x = [p[0] for p in flight_history[:frame+1]]
    visited_y = [p[1] for p in flight_history[:frame+1]]
    trail_line.set_data(visited_x, visited_y)
    
    # 그래프 업데이트
    if frame > 0:
        prev_pos = flight_history[frame-1]
        step_dist = np.hypot(current_pos[0]-prev_pos[0], current_pos[1]-prev_pos[1])
        current_distance += step_dist
        
    dist_to_target = np.hypot(TARGET[0]-current_pos[0], TARGET[1]-current_pos[1])
    
    # 리스트 크기를 프레임에 맞춤
    if len(traveled_data) > frame:
        traveled_data[frame] = current_distance
        remaining_data[frame] = dist_to_target
    else:
        traveled_data.append(current_distance)
        remaining_data.append(dist_to_target)
        
    line_traveled.set_data(range(frame+1), traveled_data[:frame+1])
    line_remaining.set_data(range(frame+1), remaining_data[:frame+1])
    
    return drone_dot, trail_line, line_traveled, line_remaining, dyn_circle, sensor_circle

print("시각화 창을 띄웁니다...")
# repeat=False를 적용하여 목적지 도착 시 애니메이션 종료
ani = FuncAnimation(fig, update, frames=len(flight_history), init_func=init, blit=True, interval=25, repeat=False)

# 로컬 환경에서 창을 띄움
plt.show()