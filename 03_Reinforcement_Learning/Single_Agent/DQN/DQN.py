import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import matplotlib.pyplot as plt
from matplotlib import animation

# 1. 환경 설정 (Environment)
class DroneEnv:
    def __init__(self):
        self.map_size = 300
        self.goal = np.array([280.0, 280.0])
        self.obstacles = [
            np.array([100, 100]), np.array([200, 200]), 
            np.array([150, 50]), np.array([50, 250]), np.array([250, 100])
        ]
        self.obs_radius = 27
        self.reset()

    def reset(self):
        self.pos = np.array([20.0, 20.0])
        self.angle = 0.0 
        self.path = [self.pos.copy()]
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.pos[0]/300, 
            self.pos[1]/300, 
            (self.goal[0]-self.pos[0])/300, 
            (self.goal[1]-self.pos[1])/300,
            np.cos(self.angle),
            np.sin(self.angle)
        ])

    def step(self, action):
        if action == 0: self.angle += 0.1
        elif action == 2: self.angle -= 0.1
        
        speed = 6.0
        new_pos = self.pos + np.array([np.cos(self.angle), np.sin(self.angle)]) * speed
        
        done = False
        reward = -0.1 
        
        # [수정] 목표물 거리 체크 및 스냅 (자석 효과)
        dist_to_goal = np.linalg.norm(new_pos - self.goal)
        
        # 벽 충돌
        if new_pos[0] < 0 or new_pos[0] > 300 or new_pos[1] < 0 or new_pos[1] > 300:
            reward = -40
            done = True
        # 장애물 충돌
        for obs in self.obstacles:
            if np.linalg.norm(new_pos - obs) < self.obs_radius:
                reward = -40
                done = True

        # [수정] 목적지 도착 로직: 속도 범위 내에 들어오면 "딱" 붙여줌
        if not done and dist_to_goal < 8.0: # 속도(6)보다 살짝 크게 설정
            self.pos = self.goal.copy() # 위치를 목표 지점으로 강제 고정
            self.path.append(self.pos.copy()) # 마지막 목표 좌표를 경로에 추가
            reward = 100 # 높은 도착 보상
            done = True
            return self._get_state(), reward, done
        
        # 일반적인 이동
        if not done:
            dist_old = np.linalg.norm(self.pos - self.goal)
            dist_new = np.linalg.norm(new_pos - self.goal)
            reward += (dist_old - dist_new) * 0.5
            self.pos = new_pos
            self.path.append(self.pos.copy())
            
        return self._get_state(), reward, done

# 2. DQN 모델
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3) 
        )
    def forward(self, x):
        return self.fc(x)

# 3. 학습 설정 (이전과 동일)
env = DroneEnv()
model = QNet()
target_model = QNet()
target_model.load_state_dict(model.state_dict()) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
memory = collections.deque(maxlen=10000)
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01

# 학습 실행
scores = []
episodes = 1000 

print("DQN 모델 학습을 시작합니다. 잠시만 기다려주세요...")
for epi in range(episodes):
    s = env.reset()
    score = 0
    for t in range(300): # 스텝 한도를 300으로 넉넉히 상향
        if random.random() < epsilon:
            a = random.randint(0, 2)
        else:
            s_tensor = torch.from_numpy(s).float()
            with torch.no_grad():
                a = model(s_tensor).argmax().item()
            
        ns, r, done = env.step(a)
        memory.append((s, a, r, ns, done))
        s = ns
        score += r
        if done: break
        
        if len(memory) > 500:
            batch = random.sample(memory, 64)
            s_b, a_b, r_b, ns_b, d_b = zip(*batch)
            s_b, r_b, ns_b, d_b = map(lambda x: torch.tensor(np.array(x)).float(), [s_b, r_b, ns_b, d_b])
            a_b = torch.tensor(a_b).unsqueeze(1)
            
            q_val = model(s_b).gather(1, a_b)
            with torch.no_grad():
                next_q = target_model(ns_b).max(1)[0]
            target = r_b + (1 - d_b) * gamma * next_q
            
            loss = nn.MSELoss()(q_val.squeeze(), target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
    epsilon = max(min_epsilon, epsilon * 0.995)
    scores.append(score)
    if epi % 10 == 0: target_model.load_state_dict(model.state_dict())
    if epi % 100 == 0: print(f"Episode {epi} score: {score:.2f}")

print("학습 완료! 시각화 창을 띄웁니다.")

# 4. 결과 시각화 (VS Code 용)
def visualize_result(env, model):
    s = env.reset()
    done = False
    while not done and len(env.path) < 400:
        with torch.no_grad():
            a = model(torch.from_numpy(s).float()).argmax().item()
        s, _, done = env.step(a)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(scores); ax1.set_title("Training Progress")
    
    ax2.set_xlim(0, 300); ax2.set_ylim(0, 300)
    for obs in env.obstacles:
        ax2.add_artist(plt.Circle(obs, env.obs_radius, color='yellow', alpha=0.6))
    ax2.plot(env.goal[0], env.goal[1], 'ro', markersize=10, label="Goal")
    
    line, = ax2.plot([], [], 'm-', lw=2)
    drone_dot, = ax2.plot([], [], 'mo', markersize=8)

    def animate(i):
        path_arr = np.array(env.path)
        # 프레임 인덱스 오류 수정 (i 대신 i+1, i-1 대신 i 사용)
        line.set_data(path_arr[:i+1, 0], path_arr[:i+1, 1])
        drone_dot.set_data([path_arr[i, 0]], [path_arr[i, 1]])
        return line, drone_dot

    # repeat=False 추가
    ani = animation.FuncAnimation(fig, animate, frames=len(env.path), interval=50, blit=True, repeat=False)
    
    return ani

ani = visualize_result(env, model)
# 로컬 GUI 창 실행
plt.show()