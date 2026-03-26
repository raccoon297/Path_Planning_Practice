import numpy as np
import torch
from env import DroneEnv
from ppo import PPO

def train():
    env = DroneEnv()
    agent = PPO(env.state_dim, env.action_dim)
    
    max_episodes = 3000
    update_timestep = 2000
    
    timestep = 0
    episode_rewards = []
    
    memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}
    best_reward = -np.inf

    print("=== 드론 자율비행 PPO 학습 시작 ===")
    
    for ep in range(1, max_episodes + 1):
        state = env.reset()
        ep_reward = 0
        
        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['dones'].append(done)
            memory['values'].append(value)
            
            state = next_state
            ep_reward += reward
            timestep += 1
            
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory = {k: [] for k in memory.keys()}
                
            if done:
                break
                
        episode_rewards.append(ep_reward)
        
        # 조건: 출력하는 에피소드를 100씩 출력하도록 변경
        if ep % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {ep} \t Avg Reward (Last 100): {avg_reward:.2f} \t Last Result: {info}")
            
            # 최고 성능일 때만 pth 파일 덮어쓰기 (단일 파일 유지)
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.policy.state_dict(), 'best_ppo_drone.pth')
                
    np.save('rewards_history.npy', episode_rewards)
    print("=== 학습 완료 (best_ppo_drone.pth 생성됨) ===")

if __name__ == '__main__':
    train()