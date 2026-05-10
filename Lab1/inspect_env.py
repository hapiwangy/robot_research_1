import gymnasium as gym

env = gym.make("HalfCheetah-v5")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

obs, info = env.reset(seed=42)

print("Initial observation shape:", obs.shape)
print("Initial observation:", obs)
'''
Observation space: Box(-inf, inf, (17,), float64)
Action space: Box(-1.0, 1.0, (6,), float32)
Initial observation shape: (17,)
Initial observation: [-0.01222431  0.07171958  0.03947361 -0.08116453  0.09512447  0.05222794
  0.05721286 -0.07437727 -0.08530439  0.0879398   0.07777919  0.00660307
  0.11272412  0.04675093 -0.08592925  0.03687508 -0.09588826]
代表輸出的observation以及action space
其中observation 代表環境
Action space代表agent可以動的範圍，6代表有6個可動的力矩，每個是-1~1
以這個範例為例: 
obs[0:8]
= 機器人的姿態與各關節角度
obs[8:17]
= 機器人的速度與各關節角速度
action[0] = 後大腿關節 torque
action[1] = 後小腿關節 torque
action[2] = 後腳掌關節 torque
action[3] = 前大腿關節 torque
action[4] = 前小腿關節 torque
action[5] = 前腳掌關節 torque
observation = 機器人現在的姿勢和速度
action      = 你要控制 6 個馬達怎麼出力
reward      = 這一步跑得好不好
'''



for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {step + 1}")
    print("Action:", action)
    print("Reward:", reward)
    print("Observation shape:", obs.shape)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    if terminated or truncated:
        obs, info = env.reset()

'''
Step 1
Action: [ 0.67060983  0.86370736 -0.9081     -0.9642207   0.08712655 -0.70219946]
Reward: -0.6968924142781227
Observation shape: (17,)
Terminated: False
Truncated: False
這個代表每一步action之後，整個狀態的變化
'''
env.close()