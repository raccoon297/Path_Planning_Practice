import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        self.map_size = 300.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 11차원 관측 공간 유지
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.obstacles = [
            {'pos': np.array([100.0, 100.0]), 'radius': 30.0},
            {'pos': np.array([200.0, 230.0]), 'radius': 30.0},
            {'pos': np.array([75.0, 220.0]), 'radius': 30.0}
        ]
        
        self.max_dist = self.map_size * 1.414
        self.max_steps = 500
        self.current_step = 0
        self.prev_min_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = np.array([20.0, 20.0], dtype=np.float32)
        self.current_step = 0
        
        if options and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'], dtype=np.float32)
        else:
            self.target_pos = self._get_random_target()
            
        self.prev_distance = np.linalg.norm(self.target_pos - self.drone_pos)
        self.prev_action = np.zeros(2, dtype=np.float32)
        
        self.prev_min_dist, _, _ = self._get_min_threat()
        
        return self._get_obs(), {}

    def _get_random_target(self):
        while True:
            pos = np.random.uniform(20.0, 280.0, size=(2,))
            collision = False
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs['pos']) <= obs['radius'] + 10.0:
                    collision = True
                    break
            if not collision and np.linalg.norm(pos - np.array([20.0, 20.0])) > 100.0:
                return pos

    def _get_min_threat(self):
        """💡 NEW: 원형 장애물과 4면의 벽을 모두 포함하여 가장 가까운 위협을 계산합니다."""
        min_dist = float('inf')
        threat_dx = 0.0
        threat_dy = 0.0

        # 1. 원형 장애물 검사
        for obs in self.obstacles:
            dist_vec = obs['pos'] - self.drone_pos 
            dist = np.linalg.norm(dist_vec) - obs['radius']
            if dist < min_dist:
                min_dist = dist
                norm_dist = np.linalg.norm(dist_vec)
                if norm_dist > 0:
                    threat_dx = dist_vec[0] / norm_dist
                    threat_dy = dist_vec[1] / norm_dist

        # 2. 4면의 벽 검사 (벽으로 향하는 벡터)
        walls = [
            (self.drone_pos[0], -1.0, 0.0),                     # 왼쪽 벽
            (self.map_size - self.drone_pos[0], 1.0, 0.0),      # 오른쪽 벽
            (self.drone_pos[1], 0.0, -1.0),                     # 아래쪽 벽
            (self.map_size - self.drone_pos[1], 0.0, 1.0)       # 위쪽 벽
        ]
        
        for w_dist, w_dx, w_dy in walls:
            if w_dist < min_dist:
                min_dist = w_dist
                threat_dx = w_dx
                threat_dy = w_dy

        return min_dist, threat_dx, threat_dy

    def _get_obs(self):
        min_dist, threat_dx, threat_dy = self._get_min_threat()
        
        tangent_dx = 0.0
        tangent_dy = 0.0
        # 벽이든 장애물이든 60픽셀 이내면 회피 접선 방향을 활성화
        if min_dist < 60.0:
            tangent_dx = -threat_dy
            tangent_dy = threat_dx

        safe_dist = min_dist / self.map_size

        target_vec = self.target_pos - self.drone_pos
        target_dist = np.linalg.norm(target_vec)
        
        target_dx = 0.0
        target_dy = 0.0
        if target_dist > 0:
            target_dx = target_vec[0] / target_dist
            target_dy = target_vec[1] / target_dist

        obs_list = [
            self.drone_pos[0] / self.map_size,
            self.drone_pos[1] / self.map_size,
            self.prev_action[0],
            self.prev_action[1],
            threat_dx,     # 가장 가까운 위협의 X 방향
            threat_dy,     # 가장 가까운 위협의 Y 방향
            target_dx,  
            target_dy,   
            safe_dist,
            tangent_dx,
            tangent_dy
        ]
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        action_diff = np.linalg.norm(action - self.prev_action)
        smoothness_penalty = action_diff * 0.05  
        
        self.prev_action = action
        move_vector = action * 5.0
        self.drone_pos += move_vector
        
        terminated = False
        truncated = False
        reward = 0.0
        
        # 장애물 및 벽 최단 거리 계산
        min_dist, _, _ = self._get_min_threat()
        
        # 💡 벽이든 원이든 부딪히면 무조건 사망 (코드 대폭 단순화)
        if min_dist <= 0: 
            reward = -500.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        distance_to_target = np.linalg.norm(self.target_pos - self.drone_pos)
        
        if distance_to_target < 5.0:  
            reward = 2000.0
            terminated = True
        else:
            reward = (self.prev_distance - distance_to_target) * 5.0 
            
            # 💡 NEW: 스마트 자기장 척력
            # 타겟과의 거리가 60 미만으로 좁혀지면 장애물 척력을 서서히 무시하기 시작합니다!
            if min_dist < 60.0 or self.prev_min_dist < 60.0:
                threat_diff = min_dist - self.prev_min_dist
                
                # 목표와 가까울수록 척력 배율이 0에 가까워짐
                repulsion_scale = min(1.0, distance_to_target / 60.0) 
                reward += threat_diff * 15.0 * repulsion_scale
                
            reward -= 0.1 
            reward -= smoothness_penalty

        self.prev_distance = distance_to_target
        self.prev_min_dist = min_dist 
        
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}