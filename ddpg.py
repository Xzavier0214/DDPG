# -*- coding: utf-8 -*-
"""
Created on Sat Aug  11 20:30:00 2018

@author: 曾鑫
DDPG 类，实现DDPG算法
"""

import tensorflow as tf
import numpy as np


class DDPG:

    def __init__(self,
                 s_dim, a_dim, a_bound,     # 状态、动作空间维数，动作上下界
                 capability, batch,         # 记忆容量、批量学习规模
                 gamma,                     # MDP衰减系数gamma
                 alpha_a, alpha_c,          # Actor、Critic 学习率
                 tao,                       # soft-replacement 系数
                 ):

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound

        self.capability = capability
        self.batch = batch

        self.gamma = gamma
        self.alpha_a = alpha_a
        self.alpha_c = alpha_c
        self.tao = tao

        self.memory = np.zeros((capability, 2*s_dim + a_dim + 1), np.float32)
        self.counter = 0

        self._build_net()

    def _build_net(self):
        w_initializer, b_initializer = \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        self.s = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')

        # actor和critic都需要分别建立两个相同结构的网络，evaluation-net，fixed target-net
        with tf.variable_scope('Actor'):

            self.a = self._build_a_net(self.s, 'eval', w_initializer, b_initializer)
            self.a_ = self._build_a_net(self.s_, 'target', w_initializer, b_initializer, False)

        with tf.variable_scope('Critic'):

            self.c = self._build_c_net(self.s, self.a, 'eval', w_initializer, b_initializer)
            self.c_ = self._build_c_net(self.s_, self.a_, 'target', w_initializer, b_initializer, False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Critic/target')

        self.soft_replacement = [[tf.assign(at, self.tao*ae + (1 - self.tao)*at),
                                  tf.assign(ct, self.tao*ce + (1 - self.tao)*ct)]
                                 for ae, at, ce, ct in
                                 zip(self.ae_params, self.at_params, self.ce_params, self.ct_params)]

        c_loss = tf.losses.mean_squared_error(self.c, self.r + self.gamma*self.c_)
        self.ctrain = tf.train.AdamOptimizer(self.alpha_c).minimize(c_loss, var_list=self.ce_params)

        a_loss = -tf.reduce_mean(self.c)
        self.atrain = tf.train.AdamOptimizer(self.alpha_a).minimize(a_loss, var_list=self.ae_params)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def _build_a_net(self, s, scope,
                     kernel_initializer=None, bias_initializer=None, trainable=True):

        with tf.variable_scope(scope):

            hidden_layer = tf.layers.dense(s, 30, tf.nn.relu,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           trainable=trainable)
            output_layer = tf.layers.dense(hidden_layer, self.a_dim, tf.nn.tanh,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           trainable=trainable)
        return output_layer

    def _build_c_net(self, s, a, scope,
                     kernel_initializer=None, bias_initializer=None, trainable=True):

        with tf.variable_scope(scope):

            hidden_layer = tf.layers.dense(tf.concat([s, a], 1), 30, tf.nn.relu,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           trainable=trainable)
            output_layer = tf.layers.dense(hidden_layer, 1, None,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           trainable=trainable)
        return output_layer

    def initail_net(self, path=None):

        if path is None:

            self.sess.run(tf.global_variables_initializer())
        else:

            self.saver.restore(self.sess, path)

    def choose_action(self, s):

        a = self.sess.run(self.a, {self.s: s})
        a = a*(self.a_bound[1])
        return a

    def store_transition(self, s, a, r, s_):

        transition = np.concatenate((s, a, [r], s_))
        index = self.counter % self.capability
        self.memory[index, :] = transition
        self.counter += 1

    def learn(self):

        indices = np.random.choice(self.capability, self.batch)
        memories = self.memory[indices, :]

        s = memories[:, :self.s_dim]
        a = memories[:, self.s_dim:self.s_dim + self.a_dim]
        r = memories[:, (self.s_dim + self.a_dim + 1):(self.s_dim + self.a_dim + 2)]
        s_ = memories[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.s: s})
        self.sess.run(self.ctrain, {self.s: s, self.a: a, self.r: r, self.s_: s_})

        self.sess.run(self.soft_replacement)

    def save_net(self):

        self.saver.save(self.sess, './result.ckpt')
