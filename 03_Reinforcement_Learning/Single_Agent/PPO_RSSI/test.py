import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env import DroneEnv, MAP_SIZE, OBS_RADIUS, TARGET_SIGNAL_RADIUS
from ppo import PPO

def evaluate_and_animate():
    env = DroneEnv()
    agent = PPO(env.state_dim, env.action_dim)
    
    try:
        agent.policy.load_state_dict(torch.load('best_ppo_drone.pth'))
        rewards_history = np.load('rewards_history.npy')
    except FileNotFoundError:
        print("에러: 'best_ppo_drone.pth' 또는 'rewards_history.npy'를 찾을 수 없습니다. 먼저 train.py를 실행하세요.")
        return

    # 1. 시뮬레이션 데이터 수집
    state = env.reset()
    trajectory = [env.drone_pos.copy()]
    
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu = agent.policy.actor(state_tensor)
        action = mu.numpy()[0] # 평가 시에는 탐험(노이즈) 없이 평균값으로 최적 비행
        
        state, reward, done, info = env.step(action)
        trajectory.append(env.drone_pos.copy())
        
        if done:
            print(f"테스트 비행 종료: {info} (총 스텝: {len(trajectory)})")
            break

    trajectory = np.array(trajectory)

    # 2. 화면 구성 (1행 2열: 왼쪽 애니메이션, 오른쪽 그래프)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # [왼쪽] 애니메이션 환경
    ax1.set_xlim(0, MAP_SIZE)
    ax1.set_ylim(0, MAP_SIZE)
    ax1.set_title("Drone Signal Tracking Animation")
    ax1.set_aspect('equal')
    
    # 목표물 신호 반경 (반투명 붉은 구역)
    signal_circle = plt.Circle(env.target_pos, TARGET_SIGNAL_RADIUS, color='red', alpha=0.2, label='Signal Radius (150)')
    ax1.add_patch(signal_circle)
    
    # 목표물 중앙 (빨간색 점)
    ax1.plot(env.target_pos[0], env.target_pos[1], 'ro', markersize=6, label='Target')
    
    # 장애물 (노란색)
    for obs in env.obstacles:
        obs_circle = plt.Circle(obs, OBS_RADIUS, color='yellow', ec='black')
        ax1.add_patch(obs_circle)
        
    # 드론 (보라색)
    drone_dot, = ax1.plot([], [], 'o', color='purple', markersize=8, label='Drone')
    path_line, = ax1.plot([], [], '-', color='purple', alpha=0.4)
    ax1.legend(loc='upper right')

    # [오른쪽] 학습 상태 그래프
    # 1. 원본 데이터 (회색, 반투명)
    ax2.plot(rewards_history, color='gray', alpha=0.3, label='Episode Reward')
    
    # 2. 이동 평균 계산 (최근 50개 에피소드 기준)
    window_size = 50
    if len(rewards_history) >= window_size:
        # np.convolve를 이용해 이동 평균 계산
        moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        # 이동 평균은 윈도우 크기만큼 뒤에서 시작하므로 인덱스 조정 필요
        ax2.plot(range(window_size-1, len(rewards_history)), moving_avg, color='orange', linewidth=2, label=f'{window_size}-ep Moving Avg')
    
    ax2.set_title("Training Progress (Reward History)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.legend(loc='lower right') # 범례 추가
    ax2.grid(True, linestyle='--', alpha=0.6)

    def update(frame):
        # 점 하나의 좌표라도 반드시 리스트 [] 로 감싸주어야 합니다.
        drone_dot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        path_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
        return drone_dot, path_line

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=40, blit=True, repeat=False)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_and_animate()