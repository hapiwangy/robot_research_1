import gymnasium as gym
from stable_baselines3 import PPO


ENV_ID = "HalfCheetah-v5"
MODEL_PATH = "ppo_halfcheetah"


def main():
    env = gym.make(ENV_ID, render_mode="human")

    model = PPO.load(MODEL_PATH)

    obs, info = env.reset(seed=123)

    for _ in range(3000):
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()