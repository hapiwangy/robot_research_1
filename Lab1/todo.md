# In this Lab, try to let robot run in the HalfCheetah-v5

# environment
- using python version = 3.12/ 3.13 (3.14 are likely error)
- python -m pip install "stable-baselines3[extra]"

# file explained
- inspect_env.py: check the environment we will be used
- train_half_cheetah: running a small PPO on training the cheetah
- enjoy_half_cheetah: check the output result

# notes:
## MuJoCo
```
response for physical engine
關節角度
關節速度
碰撞
接觸力
重力
摩擦
剛體動力學
> in this example, MuJoCo simulate a 2D 多關節robot，where actions are the signal for the motor control signal
```
## Gymnasium
```
pack the MuJoCo environment ito RL interface, so that we can run different RL algorithm with same unify format
observation_space: agent 能看到什麼
action_space: agent 能做什麼
reward: agent 做得好不好
```
## Stable-Baselines3
```
response for the Algorithm, so we don not have to 寫 PPO 的 loss、rollout buffer、optimizer、advantage estimation
only need to call the function
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100_000)
what it will do 
收集 trajectory
計算 reward return
估計 advantage
更新 policy network
更新 value network
儲存與載入模型
```
```
without stable-baselines3 
import gymnasium as gym

env = gym.make("HalfCheetah-v5")

obs, info = env.reset()

for _ in range(1000):
    # action here is chosed randomly, 
    ## stable-baselines3 change here to action, _ = model.predict(obs)
    ## and train the model to make it choose better option
    action = env.action_space.sample()
    

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
## overall flow chart
```
Gymnasium Env: HalfCheetah-v5
        ↑
        │ observation, reward
        │
Stable-Baselines3 PPO Agent
        │
        │ action
        ↓
MuJoCo Physics Simulation
```
```
1. env.reset()
2. agent 根據 obs 產生 action
3. env.step(action)
4. MuJoCo 模擬下一個物理狀態
5. Gymnasium 回傳 obs, reward, terminated, truncated
6. SB3 收集資料
7. PPO 更新神經網路
8. 重複很多次
```
```
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("HalfCheetah-v5")

model = PPO("MlpPolicy", env)

model.learn(100_000)

model.save("agent")
MuJoCo = 物理世界
Gymnasium = 世界的標準操作介面
Stable-Baselines3 = 學習怎麼在世界中行動的大腦
```
## parameters meaning
```
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

model.learn(total_timesteps=100_000)
> learning_rates
learning_rate=3e-4   # PPO 常用起點
learning_rate=1e-4   # 更保守
learning_rate=1e-3   # 較激進，可能不穩

> n_steps
先用目前 policy 跑環境，收集 n_steps 筆資料
↓
用這批資料更新神經網路
↓
再回去環境收集下一批資料
env = make_vec_env("HalfCheetah-v5", n_envs=4)
n_steps = 2048
our rollout will be 4 * 2048 = 8192
n_steps 大
更新頻率較低
每次更新比較慢

n_steps 小
更新比較頻繁
資料較少，估計可能比較吵
訓練可能不穩

use 1024 or 2048 as start point


> batch_size
使用多少資料來更新網路
把剛蒐集到的資料 / batch_size來當作一次的iteration，用來更新網路
小 
更新比較頻繁、比較 noisy
有時探索效果好，但訓練曲線較抖
大
更新比較穩
但可能比較慢，也可能泛化差一點
tips: batch_size 最好能整除 n_steps × n_envs

> gamma
折扣因子，決定 agent 有多重視未來 reward
reward = current reward + future reward 
gamma 越接近 1：越重視長期結果
gamma 越小：越重視眼前結果
example gamma = 0.99, means next step is 0.99 and the step after nextstep is 0.99 * 0.99
gamma too small > focus on the current reward

> total_timesteps
how mant action can agent do 
10,000 steps    只是測試程式能不能跑
100,000 steps   可以看到一點學習
1,000,000 steps 比較可能有像樣結果
3,000,000+      MuJoCo 常見訓練量
```
```
trouble shot
1. reward not improve at all
> total_timesteps increase
> learning_rate increase or dcecrease

2. reward increase and decrease abruptly
> decrease decrease
> bigger batch size

3. reward曲線很陡
> increase n_steps
> increase batch_size

4. training too slow
> n_envs increase
> total_timesteps decrease first for fast expeiments
```
```
learning_rate     = 每次教練修正動作的幅度
n_steps           = 看了多少次跑步影片後才給建議
batch_size        = 每次分析幾段影片
gamma             = 選手重視眼前一步，還是整段跑步表現
total_timesteps   = 總共練習多少步
means we are modifiy below
學習速度
穩定性
資料量
長期規劃能力
總訓練時間
```
## try other robot
| 環境                    | 難度 | 說明       |
| --------------------- | -: | -------- |
| `InvertedPendulum-v5` |  低 | 適合入門     |
| `Reacher-v5`          | 中低 | 機械臂到達目標  |
| `HalfCheetah-v5`      |  中 | 學會向前跑    |
| `Hopper-v5`           | 中高 | 單腳跳躍，容易倒 |
| `Ant-v5`              |  高 | 四足機器人    |
| `Humanoid-v5`         | 很高 | 高維度人形控制  |
