from drone.py import CustomEnv
import gym

env = CustomEnv()

# Create PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

# Train PPO
model.learn(total_timesteps=20000)