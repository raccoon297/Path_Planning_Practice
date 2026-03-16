import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        self.map_size = 300.0
        # 행동 공간: 드론의 x, y 이동 속도 (정규화된 값 -1 ~ 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 관측 공간: 드론의 현재 위치 (x, y) - 0~1 사이로 정규화
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 환경 요소 세팅
        self.target_pos = np.array([280.0, 280.0])
        self.obstacles = [
            {'pos': np.array([100.0, 100.0]), 'radius': 30.0},
            {'pos': np.array([200.0, 230.0]), 'radius': 30.0},
            {'pos': np.array([75.0, 220.0]), 'radius': 50.0},
            {'pos': np.array([250.0, 100.0]), 'radius': 30.0}
        ]
        
        self.drone_pos = None
        self.max_steps = 500
        self.current_step = 0
        self.prev_distance = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 드론을 왼쪽 아래에서 시작
        self.drone_pos = np.array([20.0, 20.0], dtype=np.float32)
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.target_pos - self.drone_pos)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # PPO 학습 효율을 위해 좌표를 0~1로 정규화하여 반환
        return self.drone_pos / self.map_size

    def step(self, action):
        self.current_step += 1
        
        # 액션 적용 (최대 이동 속도를 5로 설정)
        move_vector = action * 5.0
        self.drone_pos += move_vector
        
        terminated = False
        truncated = False
        reward = 0.0
        
        # 1. 벽 충돌 검사 (맵 밖으로 나가는 경우)
        if (self.drone_pos[0] < 0 or self.drone_pos[0] > self.map_size or 
            self.drone_pos[1] < 0 or self.drone_pos[1] > self.map_size):
            reward = -1000.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        # 2. 장애물 충돌 검사
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.drone_pos - obs['pos'] + 10)
            if dist_to_obs <= obs['radius']: # 드론 자체 크기는 점으로 가정
                reward = -1000.0
                terminated = True
                return self._get_obs(), reward, terminated, truncated, {}

        # 3. 목표 도달 검사 및 거리 보상
        distance_to_target = np.linalg.norm(self.target_pos - self.drone_pos)
        
        if distance_to_target < 5.0:  # 목표물 반경 5 이내 도달 시 성공
            reward = 2000.0
            terminated = True
        else:
            # 거리 기반 Dense Reward: 타겟에 가까워지면 보상, 멀어지면 페널티
            reward = (self.prev_distance - distance_to_target) * 2.0
            reward -= 0.0 # 시간 지연 페널티 (빨리 도착하도록 유도)

        self.prev_distance = distance_to_target
        
        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}