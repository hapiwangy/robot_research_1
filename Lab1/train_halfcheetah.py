import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


ENV_ID = "HalfCheetah-v5"
MODEL_PATH = "ppo_halfcheetah"


def main():
    # 1. 建立向量化環境
    # Stable-Baselines3 通常建議用 VecEnv 介面訓練
    env = make_vec_env(ENV_ID, n_envs=4, seed=42)

    # 2. 建立 PPO 模型
    # MlpPolicy 表示 observation 是向量，使用多層感知器神經網路
    ## PPO: 為了讓new policy不要和 old policy差太多 > 透過clip過大的policy update來達到效果
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./tensorboard_logs/",
    )

    # 3. 訓練
    # 這裡 100_000 只是快速熟悉用，真正要好表現通常需要更多 steps
    model.learn(total_timesteps=100_000)

    # 4. 儲存模型
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 5. 評估模型
    eval_env = gym.make(ENV_ID)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()