import math


class CosLrWarmupScheduler():
    def __init__(self, optimizer, total_iter):
        assert type(total_iter) is int
        self.optimizer = optimizer
        self.total_iter = total_iter
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.current_iter = 0

    def step(self):
        self.current_iter += 1
        if self.current_iter > self.total_iter:
            return
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = (math.cos(math.pi + self.current_iter/self.total_iter * math.pi) + 1.000001) * 0.5 * self.base_lr[i]


class CosLr():
    def __init__(self, optimizer, period_iter):
        assert type(period_iter) is int
        self.optimizer = optimizer
        self.total_iter = period_iter
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.current_iter = 0

    def step(self):
        self.current_iter += 1
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = (math.cos(self.current_iter/self.total_iter * math.pi) + 1.000001) * 0.5 * self.base_lr[i]