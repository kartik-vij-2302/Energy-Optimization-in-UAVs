import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Custom UAV environment for reinforcement learning
class UAVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0):
        super().__init__()
        # Environment parameters
        self.dt = 0.1  # time step
        self.drone_mass = 1.5
        self.payload_min = 0.0
        self.payload_max = 1.5
        self.drag_coeff = 1.1
        self.fluid_density = 1.0
        self.cross_section_area = 0.08
        self.max_acc = 6.0
        self.max_dacc = 4.0
        self.max_speed = 12.0
        self.max_dist = 200.0
        self.max_steps = 800
        self.wind_max = 6.0
        self.goal_radius = 1.0
        self.alpha_energy = 0.05
        self.beta_smooth = 0.001
        self.success_bonus = 100.0
        self.step_penalty = 0.01
        self.progress_scale = 1.0

        # Observation and action space definitions
        low = np.array([-self.max_dist, -self.max_dist, -self.max_speed, -self.max_speed,
                        -self.max_acc, -self.max_acc, -self.wind_max, -self.wind_max, 0.0], dtype=np.float32)
        high = np.array([ self.max_dist,  self.max_dist,  self.max_speed,  self.max_speed,
                          self.max_acc,  self.max_acc,  self.wind_max,  self.wind_max, self.payload_max], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_dacc, high=self.max_dacc, shape=(2,), dtype=np.float32)

        # Random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.distances = []
        self.energies = []
        self.reset(seed=seed)

    # Calculate drag force based on relative velocity
    def _drag(self, v_rel):
        s = np.linalg.norm(v_rel)
        if s == 0.0:
            return np.zeros(2, dtype=np.float32), 0.0
        mag = 0.5 * self.fluid_density * (s**2) * self.drag_coeff * self.cross_section_area
        vec = -mag * (v_rel / s)
        return vec.astype(np.float32), mag

    # Clip vector to a maximum norm
    def _clip_vec(self, vec, max_norm):
        n = np.linalg.norm(vec)
        if n > max_norm:
            return vec * (max_norm / (n + 1e-8))
        return vec

    # Reset environment to initial state
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.payload = float(self.np_random.uniform(self.payload_min, self.payload_max))
        self.pos = self.np_random.uniform(-50.0, 50.0, size=(2,)).astype(np.float32)
        gdir = np.sign(self.np_random.uniform(-1, 1, size=(2,)))
        self.goal = self.np_random.uniform(80.0, 120.0, size=(2,)).astype(np.float32) * gdir
        self.vel = np.zeros(2, dtype=np.float32)
        self.acc = np.zeros(2, dtype=np.float32)
        self.wind = self.np_random.uniform(-self.wind_max, self.wind_max, size=(2,)).astype(np.float32)
        self.steps = 0
        self.prev_dist = float(np.linalg.norm(self.goal - self.pos))
        self.distances = [self.prev_dist]
        self.energies = [0.0]  # start with zero energy
        return self._get_obs(), {}

    # Get current observation
    def _get_obs(self):
        rel = (self.goal - self.pos).astype(np.float32)
        return np.array([rel[0], rel[1], self.vel[0], self.vel[1],
                         self.acc[0], self.acc[1], self.wind[0], self.wind[1], self.payload], dtype=np.float32)

    # Step the environment by one timestep
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = self._clip_vec(action, self.max_dacc)
        self.acc = self.acc + action
        self.acc = np.clip(self.acc, -self.max_acc, self.max_acc)

        m = self.drone_mass + self.payload
        v_rel = self.vel - self.wind
        drag_vec, _ = self._drag(v_rel)
        thrust_force = m * self.acc
        power = float(np.dot(thrust_force, self.vel))
        if power < 0.0:
            power = 0.0
        energy = power * self.dt
        self.energies.append(energy)

        acc_eff = self.acc + drag_vec / m
        self.vel = self.vel + acc_eff * self.dt
        self.vel = self._clip_vec(self.vel, self.max_speed)
        self.pos = self.pos + self.vel * self.dt
        self.steps += 1

        dist = float(np.linalg.norm(self.goal - self.pos))
        self.distances.append(dist)
        progress = self.prev_dist - dist
        self.prev_dist = dist

        # Reward calculation
        reward = self.progress_scale * progress \
             - self.alpha_energy * energy \
             - self.step_penalty \
             - self.beta_smooth * float(np.dot(action, action))

        terminated = False
        truncated = False
        if dist <= self.goal_radius:
            reward += self.success_bonus
            terminated = True
        if np.linalg.norm(self.pos) > self.max_dist:
            truncated = True
            reward -= 10.0
        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {"energy": energy, "dist": dist}
        return obs, reward, terminated, truncated, info


# Helper function to create environment with a given seed
def make_env(seed=0):
    def _f():
        return UAVEnv(seed=seed)
    return _f


if __name__ == "__main__":
    # Create vectorized environment for training
    env = DummyVecEnv([make_env(0)])
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=256,
                gae_lambda=0.95, gamma=0.995, n_epochs=30, learning_rate=3e-4, clip_range=0.2)

    # Train the agent
    model.learn(total_timesteps=200000)
    env.close()

    # Test the trained agent
    test_env = UAVEnv(seed=42)
    obs, _ = test_env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)

    # Plot distance & energy
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Distance to Goal (m)', color=color)
    ax1.plot(test_env.distances, color=color, label='Distance')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Energy per Step (J)', color=color)
    ax2.plot(test_env.energies, color=color, label='Energy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('UAV Distance & Energy Consumption Over Time')
    plt.show()
