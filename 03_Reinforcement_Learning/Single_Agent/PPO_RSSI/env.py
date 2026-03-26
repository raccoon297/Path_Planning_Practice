import numpy as np

# --- 환경 설정 상수 ---
MAP_SIZE = 600
START_POS = np.array([30.0, 30.0])
OBS_COUNT = 4
OBS_RADIUS = 40 
TARGET_SIGNAL_RADIUS = 250 
TARGET_RSSI_THRESHOLD = 0.85 
MAX_STEPS = 400

# ==========================================
# 🎯 타겟 생성 모드 설정 패널
# ==========================================
# 'random' : 랜덤 위치 생성 / 'manual' : 수동 위치 생성
TARGET_MODE = 'random' 

# TARGET_MODE가 'manual'일 때 생성될 타겟의 고정 좌표 [x, y]
MANUAL_TARGET_POS = np.array([500.0, 450.0]) 
# ==========================================

class DroneEnv:
    def __init__(self):
        self.map_size = MAP_SIZE
        self.drone_start = START_POS.copy()
        
        self.state_dim = 22 
        self.action_dim = 2 
        
        self.obstacles = [
            np.array([200.0, 200.0]),
            np.array([400.0, 150.0]),
            np.array([150.0, 450.0]),
            np.array([450.0, 400.0])
        ]
        
    def reset(self):
        self.drone_pos = self.drone_start.copy()
        self.prev_action = np.array([1.0, 1.0]) / np.sqrt(2) 
        self.search_vec = np.array([1.0, 1.0]) / np.sqrt(2)
        self.steps = 0
        
        # [핵심 1] 타겟 생성 로직 분기
        if TARGET_MODE == 'manual':
            self.target_pos = MANUAL_TARGET_POS.copy()
        else:
            # 랜덤 모드일 경우
            while True:
                self.target_pos = np.random.uniform(50, self.map_size - 50, size=2)
                valid = True
                
                # [핵심 2] (0,0) 부터 (200,200) 구역 생성 제외 (시작점 근처 스폰 금지)
                if self.target_pos[0] <= 200.0 and self.target_pos[1] <= 200.0:
                    valid = False
                    
                # 장애물과 겹치는지 확인
                if valid:
                    for obs in self.obstacles:
                        if np.linalg.norm(self.target_pos - obs) < (OBS_RADIUS + 30):
                            valid = False
                            break
                            
                if valid:
                    break
                
        self.prev_dist = np.linalg.norm(self.target_pos - self.drone_pos)
        self.prev_rssi = self.get_rssi(self.drone_pos)
        return self.get_state()

    def get_rssi(self, pos):
        dist = np.linalg.norm(self.target_pos - pos)
        if dist <= TARGET_SIGNAL_RADIUS:
            return 1.0 - (dist / TARGET_SIGNAL_RADIUS)
        return 0.0

    def get_state(self):
        curr_rssi = self.get_rssi(self.drone_pos)
        rssi_diff = curr_rssi - self.prev_rssi
        
        # [핵심 2] 드론에게 제공하는 만능 나침반 (Sensor Direction)
        if curr_rssi > 0:
            # 핑크 구역(추적 모드): 조난자 방향을 알려줌
            vec_to_target = self.target_pos - self.drone_pos
            norm = np.linalg.norm(vec_to_target)
            sensor_dir = vec_to_target / norm if norm > 0.001 else np.array([0.0, 0.0])
        else:
            # 하얀 구역(탐색 모드): 맵 전체를 수색하는 당구공 벡터를 알려줌
            sensor_dir = self.search_vec
        
        state = [
            self.drone_pos[0] / self.map_size, 
            self.drone_pos[1] / self.map_size, 
            (self.map_size - self.drone_pos[0]) / self.map_size, 
            (self.map_size - self.drone_pos[1]) / self.map_size,
            curr_rssi, 
            rssi_diff * 100.0, 
            self.prev_action[0], 
            self.prev_action[1],
            sensor_dir[0], 
            sensor_dir[1]
        ]
        
        for obs in self.obstacles:
            rx = (obs[0] - self.drone_pos[0]) / self.map_size
            ry = (obs[1] - self.drone_pos[1]) / self.map_size
            dist = np.linalg.norm([rx, ry])
            state.extend([rx, ry, dist])
            
        return np.array(state, dtype=np.float32)

    def step(self, action):
        move_step = 10.0
        
        action = np.clip(action, -1.0, 1.0)
        a_norm = np.linalg.norm(action)
        if a_norm > 0.001:
            action = action / a_norm 
        else:
            action = self.prev_action 
            
        inertia = 0.5 
        velocity = (action * (1.0 - inertia)) + (self.prev_action * inertia)
        
        v_norm = np.linalg.norm(velocity)
        if v_norm > 0.001:
            velocity = velocity / v_norm
        else:
            velocity = self.prev_action
            
        self.prev_action = velocity 
        new_pos = self.drone_pos + velocity * move_step
        
        # [핵심 3] 가상 탐색 벡터(당구공)가 벽에 닿으면 반사각으로 튕겨 나오도록 계산
        margin = 30.0
        if new_pos[0] < margin and self.search_vec[0] < 0:
            self.search_vec[0] *= -1
        elif new_pos[0] > self.map_size - margin and self.search_vec[0] > 0:
            self.search_vec[0] *= -1
            
        if new_pos[1] < margin and self.search_vec[1] < 0:
            self.search_vec[1] *= -1
        elif new_pos[1] > self.map_size - margin and self.search_vec[1] > 0:
            self.search_vec[1] *= -1
            
        self.search_vec = self.search_vec / np.linalg.norm(self.search_vec)

        curr_rssi = self.get_rssi(new_pos)
        curr_dist = np.linalg.norm(self.target_pos - new_pos)
        dist_diff = self.prev_dist - curr_dist

        reward = 0
        done = False
        info = ""

        if new_pos[0] < 0 or new_pos[0] > self.map_size or new_pos[1] < 0 or new_pos[1] > self.map_size:
            reward = -500 
            done = True
            info = "Wall Crash"
            self.prev_dist = curr_dist
            self.prev_rssi = curr_rssi
            return self.get_state(), reward, done, info

        obs_penalty = 0
        for obs in self.obstacles:
            dist = np.linalg.norm(new_pos - obs)
            if dist <= OBS_RADIUS:
                reward = -500 
                done = True
                info = "Obstacle Crash"
                self.prev_dist = curr_dist
                self.prev_rssi = curr_rssi
                return self.get_state(), reward, done, info
            elif dist < OBS_RADIUS + 30: 
                obs_penalty -= (OBS_RADIUS + 30 - dist) * 5.0 

        wall_penalty = 0
        w_margin = 20.0
        if new_pos[0] < w_margin: wall_penalty -= (w_margin - new_pos[0]) * 5.0
        if new_pos[0] > self.map_size - w_margin: wall_penalty -= (new_pos[0] - (self.map_size - w_margin)) * 5.0
        if new_pos[1] < w_margin: wall_penalty -= (w_margin - new_pos[1]) * 5.0
        if new_pos[1] > self.map_size - w_margin: wall_penalty -= (new_pos[1] - (self.map_size - w_margin)) * 5.0

        self.drone_pos = new_pos
        self.steps += 1

        if curr_rssi >= TARGET_RSSI_THRESHOLD:
            reward = 3000 
            done = True
            info = "Goal Reached"
        else:
            step_penalty = -1.0 
            
            if curr_rssi > 0:
                # 핑크색 구역: 타겟을 향해 나침반(sensor_dir)대로 직진
                vec_to_target = self.target_pos - self.drone_pos
                norm = np.linalg.norm(vec_to_target)
                target_dir = vec_to_target / norm if norm > 0.001 else np.array([0.0, 0.0])
                alignment = np.dot(velocity, target_dir)
                
                reward = step_penalty + (dist_diff * 15.0) + (alignment * 2.0) + obs_penalty + wall_penalty
            else:
                # 하얀색 구역: 탐색 벡터(search_vec)를 찰떡같이 따라가면 강력한 칭찬
                alignment = np.dot(velocity, self.search_vec)
                reward = step_penalty + (alignment * 5.0) + obs_penalty + wall_penalty

        if self.steps >= MAX_STEPS:
            done = True
            info = "Timeout"

        self.prev_dist = curr_dist
        self.prev_rssi = curr_rssi
        return self.get_state(), reward, done, info