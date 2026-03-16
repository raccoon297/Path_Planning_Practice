import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. 노드 클래스 ---
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

# --- 2. A* 알고리즘 ---
def astar(maze, start, end):
    start_node = Node(None, start)
    end_node = Node(None, end)
    open_list = []
    closed_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                # 시각화 그래프를 위해 (좌표, G값, H값)을 튜플로 저장
                path.append((current.position, current.g, current.h))
                current = current.parent
            return path[::-1]

        children = []
        for new_position in movements:
            node_position = (current_node.position[0] + new_position[0], 
                             current_node.position[1] + new_position[1])

            if (node_position[0] > (len(maze) - 1) or 
                node_position[0] < 0 or 
                node_position[1] > (len(maze[0]) -1) or 
                node_position[1] < 0):
                continue
            if maze[node_position[0]][node_position[1]] == 1:
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        for child in children:
            if child in closed_list:
                continue
            
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + \
                      ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            if len([open_node for open_node in open_list if child == open_node[1] and child.g > open_node[1].g]) > 0:
                continue
            heapq.heappush(open_list, (child.f, child))
    return None

# --- 3. 환경 설정 ---
def create_environment():
    width, height = 50, 50
    maze = np.zeros((width, height))
    obstacles = [(15, 15, 6), (30, 10, 7), (25, 35, 8)]
    for ox, oy, r in obstacles:
        for x in range(width):
            for y in range(height):
                if (x - ox)**2 + (y - oy)**2 <= r**2:
                    maze[x][y] = 1
    return maze

# --- 4. 메인 실행 및 그래프 시각화 ---
maze = create_environment()
start = (5, 5)
end = (45, 45)

path_data = astar(maze, start, end)

if path_data:
    # 데이터 분리 (좌표, G값, H값)
    path_coords = [p[0] for p in path_data]
    g_vals = [p[1] for p in path_data]
    h_vals = [p[2] for p in path_data]
    steps = list(range(len(path_data)))

    # [중요] 1행 2열의 그래프 생성 (왼쪽: 맵, 오른쪽: 분석 그래프)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # --- 왼쪽: 드론 맵 ---
    ax1.set_title("UAV Path Simulation")
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 50)
    ax1.set_aspect('equal')
    ax1.grid(True)
    
    # 장애물, 목적지 그리기
    obstacle_x, obstacle_y = np.where(maze == 1)
    ax1.plot(obstacle_x, obstacle_y, 'bs', markersize=4, label='Trees')
    ax1.plot(end[0], end[1], 'ro', markersize=10, label='Goal')
    
    # 드론 점과 경로선 초기화
    drone_dot, = ax1.plot([], [], 'go', markersize=10, label='Drone')
    path_line, = ax1.plot([], [], 'k--', alpha=0.3)
    ax1.legend(loc='upper left')

    # --- 오른쪽: 비용 분석 그래프 ---
    ax2.set_title("Cost Convergence Analysis (G vs H)")
    ax2.set_xlim(0, len(path_data))
    ax2.set_ylim(0, max(max(g_vals), max(h_vals)) + 100)
    ax2.set_xlabel("Steps (Time)")
    ax2.set_ylabel("Cost (Distance)")
    ax2.grid(True)

    # 그래프 선 초기화
    line_g, = ax2.plot([], [], 'b-', linewidth=2, label='G-Cost (Traveled)')
    line_h, = ax2.plot([], [], 'r-', linewidth=2, label='H-Cost (Remaining)')
    ax2.legend()

    # --- 애니메이션 함수 ---
    def init():
        drone_dot.set_data([], [])
        path_line.set_data([], [])
        line_g.set_data([], [])
        line_h.set_data([], [])
        return drone_dot, path_line, line_g, line_h

    def update(frame):
        # 1. 왼쪽 맵 업데이트
        x, y = path_coords[frame]
        drone_dot.set_data([x], [y])
        
        # 지나온 경로 그리기
        visited_x = [p[0] for p in path_coords[:frame+1]]
        visited_y = [p[1] for p in path_coords[:frame+1]]
        path_line.set_data(visited_x, visited_y)

        # 2. 오른쪽 그래프 업데이트
        current_steps = steps[:frame+1]
        current_g = g_vals[:frame+1]
        current_h = h_vals[:frame+1]
        
        line_g.set_data(current_steps, current_g)
        line_h.set_data(current_steps, current_h)

        return drone_dot, path_line, line_g, line_h

    # repeat=False를 추가하여 한 번만 재생되도록 설정
    ani = FuncAnimation(fig, update, frames=len(path_data), init_func=init, blit=True, interval=50, repeat=False)

    # 로컬 창을 띄우는 함수 (VS Code 실행용)
    plt.show()

else:
    print("경로를 찾을 수 없습니다.")