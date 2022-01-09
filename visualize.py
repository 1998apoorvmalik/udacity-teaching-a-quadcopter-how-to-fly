"""
Class for plotting a quadrotor
Original code by: Daniel Ingram (https://github.com/AtsushiSakai/PythonRobotics)
Edited by: Apoorv Malik (https://github.com/1998apoorvmalik)
"""
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np


class Quadrotor:
    def __init__(
        self,
        x=0,
        y=0,
        z=0,
        roll=0,
        pitch=0,
        yaw=0,
        target_pos=np.array([0.0, 0.0, 10.0]),
        x_lim=(-50, 50),
        y_lim=(-50, 50),
        z_lim=(0, 250),
        figsize=(3, 3),
        size=10,
    ):

        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim
        self.figsize = figsize

        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.rewards = []
        self.time_steps = []
        self.target_pos = target_pos
        self.update_pose(x, y, z, roll, pitch, yaw, reward=0, time=0)

    def update_pose(self, x, y, z, roll, pitch, yaw, reward, time):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.reward = reward
        self.time = time
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)
        self.rewards.append(reward)
        self.time_steps.append(self.time)

        t_image = self.plot_path()
        r_image = self.plot_reward()

        return t_image, r_image

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [
                [
                    cos(yaw) * cos(pitch),
                    -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll),
                    sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll),
                    x,
                ],
                [
                    sin(yaw) * cos(pitch),
                    cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll),
                    -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll),
                    y,
                ],
                [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z],
            ]
        )

    def plot_path(self):  # pragma: no cover
        plt.ion()
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        plt.cla()

        ax.plot(
            [p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
            [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
            [p1_t[2], p2_t[2], p3_t[2], p4_t[2]],
            "k.",
        )

        ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]], [p1_t[2], p2_t[2]], "r-")
        ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]], [p3_t[2], p4_t[2]], "r-")

        ax.plot(self.x_data, self.y_data, self.z_data, "b:")

        ax.scatter(self.target_pos[0], self.target_pos[1], self.target_pos[2], label="Target position", color="green")
        ax.set_title("Live Drone Trajectory")
        ax.legend()

        plt.xlim(self.x_lim)
        plt.ylim(self.y_lim)
        ax.set_zlim(self.z_lim)

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image

    def plot_reward(self):
        x = self.rewards
        y = self.time_steps

        plt.ion()
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel = "Time"
        ax.set_ylabel = "Reward"
        ax.set_title("Reward vs Time")

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image
