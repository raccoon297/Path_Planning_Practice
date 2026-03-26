import os
import torch
import numpy as np
from drone_env import DroneEnv
from ppo import PPOAgent

def main():
    # 환경 초기화
    env = DroneEnv()
    
    # 하이퍼파라미터 설정
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_episodes = 2000
    max_steps = 500
    update_timestep = 1000 # 2000 스텝마다 신경망 업데이트
    
    ppo_agent = PPOAgent(
        state_dim, action_dim,
        lr_actor=0.0003, lr_critic=0.001,
        gamma=0.99, K_epochs=40, eps_clip=0.2
    )
    
    time_step = 0
    print_freq = 50
    running_reward = 0
    
    # 평가용 그래프를 위한 데이터 저장 (pandas 대신 순수 로직 사용)
    episode_rewards = []
    
    print("PPO 모델 학습을 시작합니다...")
    
    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        
        for t in range(1, max_steps + 1):
            time_step += 1
            
            # 행동 선택 및 환경과 상호작용
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 버퍼에 보상 및 종료 여부 저장
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            ep_reward += reward
            
            # 일정 스텝마다 PPO 알고리즘 업데이트
            if time_step % update_timestep == 0:
                ppo_agent.update()
                
            if done:
                break
                
        running_reward += ep_reward
        episode_rewards.append(ep_reward)
        
        # 로그 출력
        if ep % print_freq == 0:
            avg_reward = running_reward / print_freq
            print(f"Episode: {ep} \t Average Reward: {avg_reward:.2f}")
            running_reward = 0
            
    # 학습된 모델 저장 및 평가용 데이터 저장
    os.makedirs("./ppo_logs/", exist_ok=True)
    torch.save(ppo_agent.policy_old.state_dict(), 'ppo_drone_model_custom.pth')
    np.savetxt("./ppo_logs/custom_rewards.csv", episode_rewards, delimiter=",")
    print("학습이 완료되었습니다. 모델과 보상 로그가 저장되었습니다.")

if __name__ == '__main__':
    main()