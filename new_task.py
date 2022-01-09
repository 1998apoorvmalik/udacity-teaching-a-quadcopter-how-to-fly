import numpy as np

from physics_sim import PhysicsSim


class Task:
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5.0, target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.target_distance = pow(sum(np.square(target_pos - init_pose[:3])), 1 / 2)

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0.0, 0.0, 10.0])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        nearness = (
            self.target_distance - pow(sum(np.square(self.sim.pose[:3] - self.target_pos)), 1 / 2)
        ) / self.target_distance
        score_1 = nearness * 10 if nearness > 0.8 else nearness
        score_2 = -0.1 * pow(sum(np.square(self.sim.pose[3:])), 1 / 2)
        score_3 = 0.1 if self.sim.done and self.sim.time < self.sim.runtime else -0.1

        score_4 = -0.0003 * np.square(self.target_pos[0] - self.sim.pose[0])
        score_5 = -0.0003 * np.square(self.target_pos[1] - self.sim.pose[1])
        score_6 = 0.1 if self.sim.pose[2] > 0.1 else -0.1
        score_7 = -1 * self.sim.pose[2] if self.sim.pose[2] > self.target_pos[2] + 10 else 0

        reward = score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7
        reward = np.tanh(reward * 10e-3)
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
