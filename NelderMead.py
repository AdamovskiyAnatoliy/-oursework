import numpy as np


class NelderMead:
    def __init__(self,
                 func,
                 x_0,
                 t=1,
                 alpha=1,
                 beta=0.5,
                 gamma=2,
                 epsilon=10**-6, 
                 max_iter=500,
                 need_init=True, 
                 x_l=None,
                 x_g=None,
                 x_h=None,
                 f_l=None,
                 f_g=None,
                 f_h=None):
        self.func = func
        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.need_init = need_init
        
        if self.need_init:
        
            d_1 = (np.sqrt(len(x_0) + 1) + len(x_0) - 1) / (len(x_0) * np.sqrt(2))
            d_2 = (np.sqrt(len(x_0) + 1) -1 ) / (len(x_0) * np.sqrt(2)) 
            d_1_ = x_0 + self.t * d_1
            d_2_ = x_0 + self.t * d_2
    
            self.x_l = x_0 
            self.x_g = np.array([d_1_[0], d_2_[1]])
            self.x_h = np.array([d_2_[0], d_1_[1]])
    
            self.f_l = self.func(self.x_l)
            self.f_g = self.func(self.x_g)
            self.f_h = self.func(self.x_h)
        else:
            self.x_l = x_l
            self.x_g = x_g
            self.x_h = x_h
            self.f_l = f_l
            self.f_g = f_g
            self.f_h = f_h
        
    def sort_current(self):
        x_array = np.array([self.x_l, self.x_g, self.x_h])
        f_array = np.array([self.f_l, self.f_g, self.f_h])
        sort_index = f_array.argsort()
        self.x_l, self.x_g, self.x_h = x_array[sort_index]
        self.f_l, self.f_g, self.f_h = f_array[sort_index]

    def center_gravity(self):
        self.x_c = (self.x_l + self.x_g) / 2
        self.f_c = self.func(self.x_c)

    def reflection(self):
        self.x_r = (1 + self.alpha) * self.x_c - self.alpha * self.x_h
        self.f_r = self.func(self.x_r)

    def stretching(self):
        self.x_e = (1 - self.gamma) * self.x_c + self.gamma * self.x_r
        self.f_e = self.func(self.x_e)

    def compression(self):
        self.x_s = self.beta * self.x_h + (1 - self.beta) * self.x_c
        self.f_s = self.func(self.x_s)

    def reduction(self):
        x_g_new = self.x_l + (self.x_l - self.x_g) / 2
        x_h_new = self.x_l + (self.x_l - self.x_h) / 2
        self.x_g, self.f_g = x_g_new, self.func(x_g_new)
        self.x_h, self.f_h = x_h_new, self.func(x_h_new)

    def replacement_h_to_r(self):
        self.x_h, self.f_h = self.x_r, self.f_r

    def replacement_h_to_e(self):
        self.x_h, self.f_h = self.x_e, self.f_e

    def replacement_h_to_s(self):
        self.x_h, self.f_h = self.x_s, self.f_s
        
    def fit(self):
        self.center_gravity()
        self.reflection()
        if self.f_r < self.f_l:
            self.stretching()
            if self.f_e < self.f_r:
                self.replacement_h_to_e()
            else:
                self.replacement_h_to_r()
        else:
            if self.f_l < self.f_r < self.f_g:
                self.replacement_h_to_r()
            else:
                if self.f_g < self.f_r < self.f_h:
                    self.replacement_h_to_r()
                else:
                    pass
                self.compression()
                if self.f_s < self.f_h:
                    self.replacement_h_to_s()
                else:
                    self.reduction()
         
    def loss2(self):
        return self.func(self.x_l)        
            
    def loss(self):
        center = np.mean([self.x_l, self.x_g, self.x_h], axis=0)
        center_val = self.func(center)
        return np.sqrt(np.mean(([self.f_l, self.f_g, self.f_h] - center_val)**2))

    def run(self):
        self.sort_current()
        for i in range(self.max_iter):
            self.fit()
            self.sort_current()
            if self.need_init:
                if self.loss2() < self.epsilon:
                    return self.x_l, self.x_g, self.x_h
            else:               
                return self.x_l, self.x_g, self.x_h