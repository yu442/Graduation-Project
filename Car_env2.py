import copy
import random
import numpy as np
import random
'''
定义State，Action，Reward
State为[X,V],Action为刹车和踩油门，Reward为 J = -et Q e
Xt+1=Xt+Vt*🔺t
Vt+1=Vt+At*🔺t
'''
class Car():
    def __init__(self):
        # self.action_space = [[1], [-1], [0]]
        # self.n_actions = len(self.action_space)
        self.n_features = 2
        self.observation = [0, 0]
        # 期望间距
        self.target = 20
        # 前车状态
        self.pl = 0
        self.vl = 0
        # 当前车状态
        self.pm = 0
        self.vm = 0
        # 位置参数和速度参数
        self.pq = 10
        self.vq = 10
        # 仿真步长
        self.t = 1
        # 设置最大速度和最小速度
        self.max = 25
        self.min = 0
        # 步数
        self.st = 0
    # 创建环境
    # 初始化每辆车的速度
    # vm = [10, random.randint(5, 15), random.randint(5, 15), random.randint(5, 15)]
    # 获取状态S
    def reset(self):
        self.st = 0
        # 设置每辆车的位置,每辆车间距100m
        self.pl = 20
        self.pm = 0
        # 初始化每辆车的速度
        self.vl = 20
        # self.vm = random.randint(5, 15)
        self.vm = 8
        self.max = self.max * 0.9995
        self.observation[0] = self.pm
        self.observation[1] = self.vm
        return self.observation

    # 判断是否发生碰撞
    def iscollsion(self,s):

        if(s[0] > self.pl):
            return True
        return False

    # 动作的执行效果
    def step(self,action):
        a = action[0]
        self.st += 1
        reward = 0
        s = copy.deepcopy(self.observation)

        # 速度和位置变化
        if s[1] + a > self.max:
            s[1] = self.max
            s[0] += s[1]
            self.pl += self.vl
        elif s[1] + a < self.min:
            s[1] = s[1]
            s[0] += s[1]
            self.pl += self.vl
        else:
            s[1] += a
            s[0] += s[1]
            self.pl += self.vl

        # 奖励函数
        if self.iscollsion(s) == True:
            reward += -100000
            done = True
        else:
            reward_tem = (-(self.pq * ((self.pl - s[0] -self.target)** 2) + self.vq * ((self.vl - s[1]) ** 2)) / 100)
            reward = reward_tem
            done = False
        return s,reward,done
    # 刷新环境
    def render(self,s):
        self.observation = s
        self.pm = s[0]
        self.vm = s[1]





