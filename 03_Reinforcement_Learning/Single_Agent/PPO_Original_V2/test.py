import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from drone_env import DroneEnv
from ppo import ActorCritic 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_custom_log(log_path):
    try:
        rewards = np.loadtxt(log_path + "custom_rewards.csv", delimiter=",")
        return rewards
    except FileNotFoundError:
        print("로그 파일을 찾을 수 없습니다.")
        return []

def main():
    env = DroneEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(state_dim, action_dim).to(device)
    try:
        policy.load_state_dict(torch.load('ppo_drone_model_custom.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("에러: 'ppo_drone_model_custom.pth' 파일을 찾을 수 없습니다. train.py를 먼저 실행해주세요.")
        return
        
    policy.eval()
    
    # 🎯 여기에 원하는 조난자(목표) 좌표를 입력합니다. (예: [280.0, 50.0])
    # (x 60-280, y 280)(125,150) 에서는 모두 성공
    target_coordinate = np.array([140.0, 220.0])
    print(f"테스트 시작: 조난자 위치는 {target_coordinate} 입니다. 드론은 신호만으로 이곳을 찾아갑니다.")
    
    # 드론 환경에 테스트용 타겟 위치 주입
    obs, _ = env.reset(options={'target_pos': target_coordinate})
    path_x, path_y = [env.drone_pos[0]], [env.drone_pos[1]]
    
    done = False
    while not done:
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mean = policy.actor_mean(state)
            action = action_mean.cpu().numpy().flatten()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        path_x.append(env.drone_pos[0])
        path_y.append(env.drone_pos[1])
        done = terminated or truncated

    # 시각화 로직
    fig, (ax_map, ax_graph) = plt.subplots(1, 2, figsize=(14, 6))
    
    rewards = load_custom_log("./ppo_logs/")
    if len(rewards) > 0:
        window = min(100, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax_graph.plot(rewards, alpha=0.3, color='gray', label='Episode Reward')
        ax_graph.plot(np.arange(window-1, len(rewards)), moving_avg, color='blue', label=f'Moving Average ({window} ep)')
    
    ax_graph.set_title("Training Evaluation (Custom PPO)")
    ax_graph.set_xlabel("Episode")
    ax_graph.set_ylabel("Reward")
    ax_graph.legend()
    ax_graph.grid(True)

    ax_map.set_xlim(0, 300)
    ax_map.set_ylim(0, 300)
    ax_map.set_title("Drone Search Path (Signal Based)")
    ax_map.set_aspect('equal')
    
    for obs_data in env.obstacles:
        circle = plt.Circle(obs_data['pos'], obs_data['radius'], color='yellow', zorder=2)
        ax_map.add_patch(circle)
        
    # 목표 위치 표시 (빨간 점)
    target = plt.Circle(env.target_pos, 5, color='red', zorder=2)
    ax_map.add_patch(target)
    
    drone_dot, = ax_map.plot([], [], 'o', color='purple', markersize=8, zorder=3)
    path_line, = ax_map.plot([], [], '-', color='purple', alpha=0.5, zorder=1)

    def init():
        drone_dot.set_data([], [])
        path_line.set_data([], [])
        return drone_dot, path_line

    def update(frame):
        drone_dot.set_data([path_x[frame]], [path_y[frame]])
        path_line.set_data(path_x[:frame+1], path_y[:frame+1])
        return drone_dot, path_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(path_x),
        init_func=init, blit=True, interval=50, repeat=False
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()