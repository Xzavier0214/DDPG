# -*- coding: utf-8 -*-
"""
Created on Sat Aug  11 20:30:00 2018

@author: 曾鑫
DDPG 算法，Continuous control with deep reinforcement learning
环境采用 gym Pendulum-v0
"""

import gym
import numpy as np

from ddpg import DDPG

# region 常量定义

MAX_EPISODES = 1000          # 一共进行试验次数
MAX_EP_STEPS = 200          # 每次试验总步数（Pendulum-v0最大200，之后done会自动取真）
GAMMA = 0.9                 # 折合因子gamma
ALPHA_A = 0.001             # actor的学习因子
ALPHA_C = 0.002             # critic的学习因子
TAO = 0.01                  # soft replacement 因子tao（越大表示替换越快）
MEMORY_CAPACITY = 10000     # 记忆储量
BATCH_SIZE = 32             # 批量学习规模

VAR = 3                     # 随机策略随机部分方差
KESI = .99995               # 随机策略随机部分方差衰减因子

RENDER = True               # 是否展示
# endregion

env = gym.make('Pendulum-v0')
env.seed(1)

s_dim = env.observation_space.shape[0]                  # 状态空间维度
a_dim = env.action_space.shape[0]                       # 动作空间维度
a_bound = env.action_space.low, env.action_space.high   # 动作取值上下界

ddpg = DDPG(s_dim, a_dim, a_bound,
            MEMORY_CAPACITY, BATCH_SIZE,
            GAMMA, ALPHA_A, ALPHA_C, TAO)
ddpg.initail_net()

var = VAR
for each_episode in range(MAX_EPISODES):

    ep_reward = 0
    s = env.reset()
    for each_step in range(MAX_EP_STEPS):

        if RENDER:

            env.render()

        # 根据状态选择动作并加上随机部分
        # 这里必须加上[0]索引，因为env.step一次只能接受一个动作
        a = ddpg.choose_action(s[np.newaxis, :])[0]
        a = np.clip(np.random.normal(a, var), *a_bound)

        s_, r, done, _ = env.step(a)

        ddpg.store_transition(s, a, r, s_)

        if ddpg.counter > MEMORY_CAPACITY:

            ddpg.learn()
            var *= KESI

        s = s_
        ep_reward += r

        if each_step == MAX_EP_STEPS - 1:

            print('Episode:', each_episode, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var)
            break

# ddpg.save_net()
