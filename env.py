#!/usr/bin/env python3

import numpy as np
import cv2
import palettable
import matplotlib

class ArmEnv:
    def __init__(self):
        self.W = 500
        # self.goal = np.array([self.W//4, self.W//4], dtype=np.float32)
        # self.goal = np.array([self.W - self.W//4, self.W - self.W//4], dtype=np.float32)
        # self.goal = np.array([self.W//2, self.W//2], dtype=np.float32)
        self.goal = np.random.randint(0,self.W, (2,))

        self.l1 = 100
        self.l2 = 100
        self.l3 = 100
        self.tool = np.zeros(2)
        self.l1_point = np.zeros(2)
        self.l2_point = np.zeros(2)

        self.obs = [None]*3
        self.obs_size = 2
        self.act_size = 2

    def _cost(self, obs):
        l1_point, l2_point, tool = self.kinematics(obs)
        cost = np.sqrt(np.sum((tool - self.goal)**2))/self.W
        return cost

    def kinematics(self, obs):
        tool = np.zeros(2)
        l1_point = np.zeros(2)
        l2_point = np.zeros(2)

        l1_point[0] = self.l1*np.cos(obs[0]) + self.W/2
        l1_point[1] = self.l1*np.sin(obs[0]) + self.W/2

        l2_point[0] = self.l1*np.cos(obs[0]) + self.l2*np.cos(obs[0] + obs[1]) + self.W/2
        l2_point[1] = self.l1*np.sin(obs[0]) + self.l2*np.sin(obs[0] + obs[1]) + self.W/2

        tool[0] = self.l1*np.cos(obs[0]) + self.l2*np.cos(obs[0] + obs[1]) + self.l3*np.cos(obs[0] + obs[1] + obs[2]) + self.W/2
        tool[1] = self.l1*np.sin(obs[0]) + self.l2*np.sin(obs[0] + obs[1]) + self.l3*np.sin(obs[0] + obs[1] + obs[2]) + self.W/2


        return l1_point, l2_point, tool

    def step(self, act):
        act = np.clip(act, -1, 1)
        act = act * 0.3
        self.obs[0] += act[0]
        self.obs[1] += act[1]
        self.obs[2] += act[2]
        self.obs[0] = np.arctan2(np.sin(self.obs[0]), np.cos(self.obs[0]))
        self.obs[1] = np.arctan2(np.sin(self.obs[1]), np.cos(self.obs[1]))
        self.obs[2] = np.arctan2(np.sin(self.obs[2]), np.cos(self.obs[2]))

        self.l1_point, self.l2_point, self.tool = self.kinematics(self.obs)

        rew = -self._cost(self.obs)
        done = False

        # return self.obs, rew, done
        return self._obs(), rew, done
        # return (self.goal[0] - self.kinematics(self.obs)[0])/self.W, rew, done

    def render(self, iteration=0):
        img = np.zeros((self.W, self.W, 3), np.uint8)
        img[:] = (200, 200, 190)

        # Goal
        cv2.circle(img, (int(self.goal[1]), int(self.goal[0])), 20, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(self.goal[1]), int(self.goal[0])), 18, (40, 40, 110), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(self.goal[1]-2), int(self.goal[0]-4)), 20, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(self.goal[1]-2), int(self.goal[0]-4)), 18, (80, 80, 220), -1, lineType=cv2.LINE_AA)


        cv2.circle(img, (self.W//2, self.W//2), 11, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        # Link 1
        cv2.line(img, (self.W//2,self.W//2), (int(self.l1_point[1]), int(self.l1_point[0])), (40,40,40), 20, lineType=cv2.LINE_AA)
        cv2.line(img, (self.W//2,self.W//2), (int(self.l1_point[1]), int(self.l1_point[0])), (120, 120, 120), 15, lineType=cv2.LINE_AA)
        cv2.circle(img, (self.W//2, self.W//2), 4, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        # Link 2
        cv2.line(img, (int(self.l1_point[1]), int(self.l1_point[0])), (int(self.l2_point[1]), int(self.l2_point[0])), (40,40,40), 20, lineType=cv2.LINE_AA)
        cv2.line(img, (int(self.l1_point[1]), int(self.l1_point[0])), (int(self.l2_point[1]), int(self.l2_point[0])), (120,120,120), 15, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(self.l1_point[1]), int(self.l1_point[0])), 4, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(self.l2_point[1]), int(self.l2_point[0])), 4, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        #Link 3
        cv2.line(img, (int(self.l2_point[1]), int(self.l2_point[0])), (int(self.tool[1]), int(self.tool[0])), (40,40,40), 20, lineType=cv2.LINE_AA)
        cv2.line(img, (int(self.l2_point[1]), int(self.l2_point[0])), (int(self.tool[1]), int(self.tool[0])), (120,120,120), 15, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(self.l2_point[1]), int(self.l2_point[0])), 4, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(self.tool[1]), int(self.tool[0])), 4, (40, 40, 40), -1, lineType=cv2.LINE_AA)


        cv2.putText(img, "Iteration: " + str(iteration), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), lineType=cv2.LINE_AA)
        cv2.namedWindow("Environment")
        cv2.imshow("Environment", img)
        cv2.waitKey(1)
        return img

    def reset(self):
        # Obs -> theta1, theta2, d_theta1, d_theta2
        # self.obs[0] = 0 + 0.5 * np.random.normal(0, 1, 1)[0]
        # self.obs[1] = np.pi/4 + 0.5 * np.random.normal(0, 1, 1)[0]
        self.obs[0] = 1
        self.obs[1] = 1.5
        self.obs[2] = 1.5
        # self.goal = np.random.randint(0,self.W, (2,))

        return self._obs()

        # return (self.goal[0] - self.kinematics(self.obs)[0])/self.W

    def _obs(self):
        l1_point, l2_point, tool = self.kinematics(self.obs)
        return (tool - self.goal)/self.W
        # # return self.obs
        # x = (self.goal[0] - self.kinematics(self.obs)[1][0])/self.W
        # y = (self.goal[1] - self.kinematics(self.obs)[1][1])/self.W
        # return [x, y]
        # # return x**2 + y**2




#
#
# env = ArmEnv3()
# obs = env.reset()
#
# done = False;
# while not done:
#     act = [0.1, 0.02, 0.08]
#     nobs, rew, done = env.step(act)
#     env.render()
#     obs = nobs
#
#
