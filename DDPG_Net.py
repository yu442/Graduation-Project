"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym
import time
import random
from Car_env3 import Car

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 500
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = False

###############################  Actor  ####################################

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net1 = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            # net2 = tf.layers.dense(net1, 64, activation=tf.nn.relu,
            #                        kernel_initializer=init_w, bias_initializer=init_b, name='l2',
            #                        trainable=trainable)
            #
            # net3 = tf.layers.dense(net2, 32, activation=tf.nn.relu,
            #                        kernel_initializer=init_w, bias_initializer=init_b, name='l4',
            #                        trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net1, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

                # net1 = tf.layers.dense(net, 256, activation=tf.nn.relu,
                #                        kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                #                        trainable=trainable)
                # net2 = tf.layers.dense(net1, 64, activation=tf.nn.relu,
                #                        kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                #                        trainable=trainable)
                # net3 = tf.layers.dense(net2, 32, activation=tf.nn.relu,
                #                        kernel_initializer=init_w, bias_initializer=init_b, name='l4',
                #                        trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


Env1 = Car()
Env2 = Car()
Env3 = Car()
Env4 = Car()

state_dim = 2
action_dim = 1
action_bound = [2 ]

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)
var = 3
sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

  # control exploration

t1 = time.time()
def PreCar(a,s,Env):
    s_, r, done = Env.step(a)
    s_ = np.array(s_)
    M.store_transition(s, a, r / 10, s_)
    if M.pointer > MEMORY_CAPACITY:
        global var
        var *= .9999
        b_M = M.sample(BATCH_SIZE)
        b_s = b_M[:, :state_dim]
        b_a = b_M[:, state_dim: state_dim + action_dim]
        b_r = b_M[:, -state_dim - 1: -state_dim]
        b_s_ = b_M[:, -state_dim:]
        critic.learn(b_s, b_a, b_r, b_s_)
        actor.learn(b_s)
    return s_,r,done
reward = []
reward_step = []
speed_step = [[],[],[],[],[]]
position_step = [[],[],[],[],[]]
for i in range(MAX_EPISODES):
    s1 = Env1.reset(80, 60, 8)
    s1 = np.array(s1)
    s2 = Env2.reset(60, 40, 8)
    s2 = np.array(s2)
    s3 = Env3.reset(40, 20, 8)
    s3 = np.array(s3)
    s4 = Env4.reset(20 ,0 ,8)
    s4 = np.array(s4)
    ep_reward = [0, 0, 0, 0]
    for j in range(MAX_EP_STEPS):
        # Add exploration noise

        # 第一队辆车
        Env1.vl = random.uniform(9,12)
        a1 = actor.choose_action(s1)
        a1 = np.clip(np.random.normal(a1, var), -2, 2)    # add randomness to action selection for exploration
        s1, r1, done1 = PreCar(a1,s1,Env1)
        ep_reward[0] += r1
        if done1:
        # if j == MAX_EP_STEPS-1 or done1:
            # print("ENV1"'Episode:', i, ' Reward: %i' % int(ep_reward[0]/j), 'Explore: %.2f' % var, 'step: ',j,'speed: ',Env1.vm,'observation: ',Env1.observation)
            break
        if i == 499:
            position_step[0].append(Env1.pl)
            speed_step[0].append(Env1.vl)
            position_step[1].append(Env1.pm)
            speed_step[1].append(Env1.vm)
        if i == 499:
            position_step[2].append(Env2.pm)
            speed_step[2].append(Env2.vm)
        # 第二队辆车
        s2 = Env2.render(Env1.vm,Env1.pm)
        s2 = np.array(s2)
        a2 = actor.choose_action(s2)
        a2 = np.clip(np.random.normal(a2, var), -2, 2)
        s2, r2, done2 = PreCar(a2,s2,Env2)
        ep_reward[1] += r2
        if done2:
        # if j == MAX_EP_STEPS-1 or done2:
            # print('Episode:', i, ' Reward: %i' % int(ep_reward[1]/j), 'Explore: %.2f' % var, 'step: ',j,'speed: ',
            #       Env2.vm,'observation: ',Env2.observation)
            break
        if i == 499:
            position_step[3].append(Env3.pm)
            speed_step[3].append(Env3.vm)
        # 第三队辆车
        s3 = Env3.render(Env2.vm, Env2.pm)
        s3 = np.array(s3)
        a3 = actor.choose_action(s3)
        a3 = np.clip(np.random.normal(a3, var), -2, 2)
        s3, r3, done3 = PreCar(a3, s3, Env3)
        ep_reward[2] += r3
        if done3:
        # if j == MAX_EP_STEPS - 1 or done3:
        #     print('Episode:', i, ' Reward: %i' % int(ep_reward[2] / j), 'Explore: %.2f' % var, 'step: ', j, 'speed: ',
        #           Env3.vm, 'observation: ', Env3.observation)
            break
        if i == 499:
            position_step[4].append(Env4.pm)
            speed_step[4].append(Env4.vm)
        # 第四队辆车
        s4 = Env4.render(Env3.vm, Env3.pm)
        s4 = np.array(s4)
        a4 = actor.choose_action(s4)
        a4 = np.clip(np.random.normal(a4, var), -2, 2)
        s4, r4, done4 = PreCar(a4, s4, Env4)
        ep_reward[3] += r4
        if j == MAX_EP_STEPS - 1 or done4:
            print('Episode:', i, ' Reward: %i' % int(ep_reward[3] / j), 'Explore: %.2f' % var, 'step: ', j,
                  'speed: ',
                  Env4.vm, 'observation: ', Env4.observation)
            break

    # if i % 10 == 0:
    #     reward.append(ep_reward/j)


plt.plot(np.arange(len(position_step[0])), position_step[0])
plt.plot(np.arange(len(position_step[1])), position_step[1])
plt.plot(np.arange(len(position_step[2])), position_step[2])
plt.plot(np.arange(len(position_step[3])), position_step[3])
plt.plot(np.arange(len(position_step[4])), position_step[4])
plt.legend(['PL','PM1','PM2','PM3','PM4'], loc = 'upper left')
plt.ylabel("Position")
plt.xlabel("Episode")
plt.show()


plt.plot(np.arange(len(speed_step[0])), speed_step[0])
plt.plot(np.arange(len(speed_step[1])), speed_step[1])
plt.plot(np.arange(len(speed_step[2])), speed_step[2])
plt.plot(np.arange(len(speed_step[3])), speed_step[3])
plt.plot(np.arange(len(speed_step[4])), speed_step[4])
plt.legend(['PL','PM1','PM2','PM3','PM4'], loc = 'upper left')
plt.ylabel("Speed")
plt.xlabel("Episode")
plt.show()

# plt.plot(np.arange(len(reward)),reward)
# plt.ylabel("Reward")
# plt.xlabel("Episode")
# plt.show()

# plt.plot(np.arange(len(reward_step)),reward_step)
# plt.ylabel("Reward For Each Step")
# plt.xlabel("Step")
# plt.show()
#
# plt.plot(np.arange(len(speed_step)),speed_step)
# plt.ylabel("Speed For Each Step")
# plt.xlabel("Step")
# plt.show()

print('Running time: ', time.time()-t1)