import numpy as np

class Step_Scheduler():
    def __init__(self, num_batch, warm_up_epoch, step_lr):
        warm_up_iters = warm_up_epoch * num_batch
        if warm_up_iters > 0:
            self.lr_lambda = lambda iter: iter / warm_up_iters \
                                     if iter < warm_up_iters else \
                                     0.1 ** np.sum(np.array(step_lr) <= iter // num_batch)
        else:
            self.lr_lambda = lambda iter : 0.1 ** np.sum(np.array(step_lr) <= iter // num_batch)
            
    def get_lambda(self):
        return self.lr_lambda

class Cosine_Scheduler():
    def __init__(self, num_batch, max_epoch, warm_up_epoch, warm_up_iters=None):
        if warm_up_iters is None:
            warm_up_iters = num_batch * warm_up_epoch
        max_iters = num_batch * max_epoch
        annealing_iters = max_iters - warm_up_iters
        self.lr_lambda = lambda iter : iter / warm_up_iters \
                        if iter < warm_up_iters else \
                        0.5 * (1 + np.cos((iter - warm_up_iters) / annealing_iters * np.pi))
        
    def get_lambda(self):
        return self.lr_lambda