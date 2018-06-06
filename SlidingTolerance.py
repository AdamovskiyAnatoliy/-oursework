import numpy as np
import matplotlib.pyplot as plt
from NelderMead import NelderMead


U = lambda g, x: 1 if g(x) < 0 else 0

def T(h, g, x):
    h_sum = sum([hi(x)**2 for hi in h])
    g_sum = sum([U(gi, x) * gi(x)**2 for gi in g])
    return np.sqrt(h_sum + g_sum)


num = 0

def f(x):
    global num
    num += 1
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2



#f = lambda x: 4* x[0] - x[1]**2 -12
h = [
    lambda x: 25 - x[0]**2 - x[1]**2
]
g = [
    lambda x: 10 * x[0] - x[0]**2 + 18 * x[1] - x[1]**2 - 34,
    lambda x: x[0],
    lambda x: x[1]
]
x_0 = np.array([2312, 231])
t = 0.3


class SlidingTolerance:
    def __init__(self, 
                 func, 
                 x_0,
                 h=[],
                 g=[],
                 t=1, 
                 epsilon=10**-6,
                 max_iter=500):
        self.func = func
        self.x_k = [x_0]
        self.h = h
        self.g = g
        self.t = t
        self.epsilon = epsilon
        self.max_iter = max_iter
        
        self.m = len(h)
        self.n = len(x_0)
        self.r = self.n - self.m
        self.F = [2 * (self.m + 1) * self.t]
    
    def init_curent(self):
        d_1 = (np.sqrt(self.n + 1) + self.n - 1) / (self.n * np.sqrt(2))
        d_2 = (np.sqrt(self.n + 1) -1 ) / (self.n * np.sqrt(2)) 
        d_1_ = self.x_k[-1] + self.t * d_1
        d_2_ = self.x_k[-1] + self.t * d_2
        
        self.x_l = self.x_k[-1] 
        self.x_g = np.array([d_1_[0], d_2_[1]])
        self.x_h = np.array([d_2_[0], d_1_[1]])
        
        self.f_l = self.func(self.x_l)
        self.f_g = self.func(self.x_g)
        self.f_h = self.func(self.x_h)
                            
    def sort_current(self):
        x_array = np.array([self.x_l, self.x_g, self.x_h])
        f_array = np.array([self.f_l, self.f_g, self.f_h])
        sort_index = f_array.argsort()
        self.x_l, self.x_g, self.x_h = x_array[sort_index]
        self.f_l, self.f_g, self.f_h = f_array[sort_index]

    def loss(self):
        center = np.mean([self.x_l, self.x_g, self.x_h], axis=0)
        center_val = self.func(center)
        return np.sqrt(np.mean(([self.f_l, self.f_g, self.f_h] - center_val)**2))
    
    def optimal_t(self):
        self.t = 0.05 * self.F[-1]
        
    def run(self):
        self.init_curent()
        self.sort_current()
        print(self.x_l, self.x_g, self.x_h)
        for i in range(3000):

            if self.F[-1] < T(self.h, self.g, self.x_l):
                self.optimal_t()
                self.x_l = NelderMead(func=lambda x: T(self.h, self.g, x),
                                      x_0=self.x_l,
                                      epsilon=self.F[-1],
                                      t=self.t,
                                      need_init=True).run()[0]
                self.sort_current()
            else:
                self.x_l, self.x_g, self.x_h = NelderMead(
                        func=self.func,
                        x_0=self.x_l,
                        epsilon=self.F[-1],
                        need_init=True,
                        t = self.t*5,
                        one_iter=True).run()
                self.sort_current()
                
            
            self.sort_current()       
            x_mean = np.mean([self.x_l, self.x_g, self.x_h], axis=0)
            theta = np.sqrt(sum((self.x_l - x_mean)**2 +\
                     (self.x_g - x_mean)**2 +\
                     (self.x_h - x_mean)**2)) * (self.m+1) / (self.r + 1)
            print('___________')
            print(i, self.x_l, self.x_g, self.x_h)
            print(theta) 
            self.F.append(min(self.F[-1], theta))
            
            if self.F[-1]<self.epsilon and T(self.h, self.g, self.x_l)<self.F[-1]:
                break
            
    
            
            
            
symplex = SlidingTolerance(x_0=x_0, 
                           func=f, 
                           h=h, 
                           g=g, 
                           t=t)

symplex.run()