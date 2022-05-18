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
        self.target = 10
        # 前车状态
        self.pl = 0
        self.vl = 0
        # 当前车状态
        self.pm = 0
        self.vm = 0
        # 位置参数和速度参数
        self.pq = 5
        self.vq = 10
        # 仿真步长
        self.t = 1
        # 设置最大速度和最小速度
        self.max = 15
        self.min = 5
        # 步数
        self.st = 0
    # 创建环境
    # 初始化每辆车的速度
    # vm = [10, random.randint(5, 15), random.randint(5, 15), random.randint(5, 15)]
    # 获取状态S
    def reset(self,
              pl,
              pm,
              vl,
    ):
        self.st = 0
        # 设置每辆车的位置,每辆车间距100m
        self.pl = pl
        self.pm = pm
        # 初始化每辆车的速度
        self.vl = vl
        # self.vm = random.randint(5, 15)
        self.vm = 6
        self.max = 15
        self.observation[0] = self.pl - self.pm
        self.observation[1] = self.vl - self.vm
        return self.observation

    # 刷新环境
    def render(self,v,p):
        self.vl = v
        self.pl = p
        self.observation[0] = self.pl - self.pm
        self.observation[1] = self.vl - self.vm
        return self.observation

    # 判断是否发生碰撞
    def iscollsion(self,s):

        if(s[0] <= 0):
            return True
        return False

    # 动作的执行效果
    def step(self,action):
        a = action[0]
        self.st += 1
        reward = 0
        # s = copy.deepcopy(self.observation)
        s = self.observation
        if self.vm + a > self.max:
            self.vm = self.max
            self.pm += self.vm
            self.pl += self.vl
        elif self.vm + a < self.min:
            self.vm = self.min
            self.pm += self.vm
            self.pl += self.vl
        else:
            self.vm += a
            self.pm += self.vm
            self.pl += self.vl
        s[0] = self.pl - self.pm
        s[1] = self.vl - self.vm

        if self.iscollsion(s) == True:
            reward += -1000
            done = True
        else:

            reward += -(self.pq * (s[0]) ** 2 + self.vq * (s[1]) ** 2)/100
            done = False
        return s,reward,done





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
        self.target = 10
        # 前车状态
        self.pl = 0
        self.vl = 0
        # 当前车状态
        self.pm = 0
        self.vm = 0
        # 位置参数和速度参数
        self.pq = 5
        self.vq = 10
        # 仿真步长
        self.t = 1
        # 设置最大速度和最小速度
        self.max = 15
        self.min = 5
        # 步数
        self.st = 0
    # 创建环境
    # 初始化每辆车的速度
    # vm = [10, random.randint(5, 15), random.randint(5, 15), random.randint(5, 15)]
    # 获取状态S
    def reset(self,
              pl,
              pm,
              vl,
    ):
        self.st = 0
        # 设置每辆车的位置,每辆车间距100m
        self.pl = pl
        self.pm = pm
        # 初始化每辆车的速度
        self.vl = vl
        # self.vm = random.randint(5, 15)
        self.vm = 6
        self.max = 15
        self.observation[0] = self.pl - self.pm
        self.observation[1] = self.vl - self.vm
        return self.observation

    # 刷新环境
    def render(self,v,p):
        self.vl = v
        self.pl = p
        self.observation[0] = self.pl - self.pm
        self.observation[1] = self.vl - self.vm
        return self.observation

    # 判断是否发生碰撞
    def iscollsion(self,s):

        if(s[0] <= 0):
            return True
        return False

    # 动作的执行效果
    def step(self,action):
        a = action[0]
        self.st += 1
        reward = 0
        # s = copy.deepcopy(self.observation)
        s = self.observation
        if self.vm + a > self.max:
            self.vm = self.max
            self.pm += self.vm
            self.pl += self.vl
        elif self.vm + a < self.min:
            self.vm = self.min
            self.pm += self.vm
            self.pl += self.vl
        else:
            self.vm += a
            self.pm += self.vm
            self.pl += self.vl
        s[0] = self.pl - self.pm
        s[1] = self.vl - self.vm

        if self.iscollsion(s) == True:
            reward += -1000
            done = True
        else:

            reward += -(self.pq * (s[0]) ** 2 + self.vq * (s[1]) ** 2)/100
            done = False
        return s,reward,done





