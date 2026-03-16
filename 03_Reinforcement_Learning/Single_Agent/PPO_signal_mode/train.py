import os
import torch
import numpy as np
from drone_env import DroneEnv
from ppo import PPOAgent

def main():
    env = DroneEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 랜덤 타겟을 찾기 위해 에피소드 수를 늘립니다.
    max_episodes = 5000 
    max_steps = 500
    update_timestep = 1000 
    
    ppo_agent = PPOAgent(
        state_dim, action_dim,
        lr_actor=0.0001, lr_critic=0.0003,
        gamma=0.99, K_epochs=10, eps_clip=0.2
    )
    
    time_step = 0
    print_freq = 100
    running_reward = 0
    episode_rewards = []
    
    print("PPO 모델 학습을 시작합니다. (랜덤 타겟 모드)")
    
    for ep in range(1, max_episodes + 1):
        state, _ = env.reset() # 매 에피소드마다 타겟이 랜덤한 위치에 생성됨
        ep_reward = 0
        
        for t in range(1, max_steps + 1):
            time_step += 1
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            ep_reward += reward
            
            if time_step % update_timestep == 0:
                ppo_agent.update()
                
            if done:
                break
                
        running_reward += ep_reward
        episode_rewards.append(ep_reward)
        
        if ep % print_freq == 0:
            avg_reward = running_reward / print_freq
            print(f"Episode: {ep} \t Average Reward: {avg_reward:.2f}")
            running_reward = 0
            
    os.makedirs("./ppo_logs/", exist_ok=True)
    torch.save(ppo_agent.policy_old.state_dict(), 'ppo_drone_model_custom.pth')
    np.savetxt("./ppo_logs/custom_rewards.csv", episode_rewards, delimiter=",")
    print("학습이 완료되었습니다. 모델과 보상 로그가 저장되었습니다.")

if __name__ == '__main__':
    main()