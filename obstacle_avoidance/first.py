import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Custom UAV environment with obstacle
class UAVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0):
        super().__init__()
        # Environment parameters
        self.dt = 0.1
        self.drone_mass = 65
        self.payload_min = 0.0
        self.payload_max = 30
        self.drag_coeff = 1.1
        self.fluid_density = 1.0
        self.cross_section_area = 0.08
        self.max_acc = 6.0
        self.max_speed = 10.0
        self.max_dist = 1000.0
        self.max_steps = 500000
        self.wind_max = 6.0
        self.goal_radius = 1.0
        self.alpha_energy = 0.05
        self.beta_smooth = 0.001
        self.success_bonus = 1000.0
        self.step_penalty = 0.1
        self.avoidance_weight = 2.0
        self.gamma_obstacle = 1.0

        # Observation and action space
        low = np.array([-self.max_dist, -self.max_dist, -self.max_speed, -self.max_speed,
                        -self.max_acc, -self.max_acc, -self.wind_max, -self.wind_max, 0.0], dtype=np.float32)
        high = np.array([ self.max_dist,  self.max_dist,  self.max_speed,  self.max_speed,
                          self.max_acc,  self.max_acc,  self.wind_max,  self.wind_max, self.payload_max], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_acc, high=self.max_acc, shape=(2,), dtype=np.float32)

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.reset(seed=seed)

    def _drag(self, v_rel):
        s = np.linalg.norm(v_rel)
        if s == 0.0:
            return np.zeros(2, dtype=np.float32), 0.0
        mag = 0.5 * self.fluid_density * (s**2) * self.drag_coeff * self.cross_section_area
        vec = -mag * (v_rel / s)
        return vec.astype(np.float32), mag

    def _clip_vec(self, vec, max_norm):
        n = np.linalg.norm(vec)
        if n > max_norm:
            return vec * (max_norm / (n + 1e-8))
        return vec

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
        self.distances = [np.linalg.norm(self.goal - self.pos)]
        self.energies = []
        self.obs_distances = []

        # Place obstacle randomly
        self.obstacle_pos = self.np_random.uniform(-40, 40, size=(2,)).astype(np.float32)
        self.obstacle_radius = 10.0

        return self._get_obs(), {}

    def _get_obs(self):
        rel = (self.goal - self.pos).astype(np.float32)
        return np.array([rel[0], rel[1], self.vel[0], self.vel[1],
                         self.acc[0], self.acc[1], self.wind[0], self.wind[1], self.payload], dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = self._clip_vec(action, self.max_acc)
        self.acc = action

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
        self.vel = self._clip_vec(self.vel + acc_eff * self.dt, self.max_speed)
        self.pos = self.pos + self.vel * self.dt
        self.steps += 1

        dist = float(np.linalg.norm(self.goal - self.pos))
        self.distances.append(dist)

        # Distance to obstacle
        obs_dist = float(np.linalg.norm(self.obstacle_pos - self.pos)) - self.obstacle_radius
        self.obs_distances.append(obs_dist)

        # Reward: closer to goal, farther from obstacle, low energy
        reward = (
            -dist
            + self.avoidance_weight * max(obs_dist, 0)  # reward distance from obstacle
            - self.alpha_energy * energy
            - self.step_penalty
            - self.beta_smooth * float(np.dot(action, action))
        )

        terminated = False
        truncated = False

        if obs_dist < 0:  # collision
            reward -= 1000.0
            terminated = True

        if dist <= self.goal_radius:  # reached goal
            reward += self.success_bonus
            terminated = True

        if np.linalg.norm(self.pos) > self.max_dist:
            truncated = True
            reward -= 10.0

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {"energy": energy, "dist": dist, "obs_dist": obs_dist}
        return obs, reward, terminated, truncated, info


# === Training & Testing ===
def make_env_with_alpha(alpha, seed=0):
    env = UAVEnv(seed=seed)
    env.alpha_energy = alpha
    return env

if __name__ == "__main__":
    alpha_values = [0.01, 0.02,0.03,0.04, 0.05]  # Different alpha_energy values
    results = {}

    for alpha in alpha_values:
        print(f"\n=== Testing alpha_energy = {alpha} ===")
        env = DummyVecEnv([lambda: make_env_with_alpha(alpha, 0)])
        model = PPO("MlpPolicy", env, verbose=0, n_steps=1024, batch_size=256,
                gae_lambda=0.95, gamma=0.995, n_epochs=20,
                learning_rate=5e-4, clip_range=0.2, device="cpu")
        model.learn(total_timesteps=500000)
        env.close()

        test_env = make_env_with_alpha(alpha, seed=42)
        obs, _ = test_env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)

        results[alpha] = {
            "distances": test_env.distances,
            "energies": test_env.energies,
            "obs_distances": test_env.obs_distances,
            "total_energy": sum(test_env.energies)
        }

    # Separate plots
    # Distance to goal
    plt.figure()
    for alpha in alpha_values:
        plt.plot(results[alpha]["distances"], label=f"Goal Dist α={alpha}")
    plt.xlabel("Step")
    plt.ylabel("Distance to Goal (m)")
    plt.title("UAV Distance to Goal")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_distance.png")

    # Energy consumption per step
    plt.figure()
    for alpha in alpha_values:
        plt.plot(results[alpha]["energies"], label=f"Energy α={alpha}")
    plt.xlabel("Step")
    plt.ylabel("Energy per Step (J)")
    plt.title("UAV Energy Consumption per Step")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_energy_per_step.png")

    # Total energy consumption (per episode)
    plt.figure()
    totals = [results[a]["total_energy"] for a in alpha_values]
    plt.bar([str(a) for a in alpha_values], totals)
    plt.xlabel("Alpha values")
    plt.ylabel("Total Energy (J)")
    plt.title("Total Energy Consumption per Episode")
    plt.savefig("plot_total_energy.png")

    # Obstacle distances
    plt.figure()
    for alpha in alpha_values:
        plt.plot(results[alpha]["obs_distances"], label=f"Obs Dist α={alpha}")
    plt.xlabel("Step")
    plt.ylabel("Distance to Obstacle (m)")
    plt.title("UAV Distance to Obstacle")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_obstacle.png")
