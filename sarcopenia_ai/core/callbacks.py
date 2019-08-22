import logging
import math
import os

import keras.backend as K
import numpy as np
from keras.callbacks import Callback

logger = logging.getLogger(__name__)


class LRFinder(Callback):
    """

    see fastai library and https://arxiv.org/abs/1506.01186

    """

    def __init__(self, max_iterations=5000, base_lr=10e-8, max_lr=1000., alpha=0.99, log_path='.'):
        super().__init__()
        self.max_iterations = max_iterations
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.losses = []
        self.lrs = []
        self.lr = self.base_lr
        self.alpha = alpha
        self.factor = (max_lr / base_lr) ** (1 / max_iterations)
        self.weighted_loss = 0.
        self.weighted_count = 0.
        self.smoothed_loss = 0.
        self.best_loss = 0.
        self.log_path = log_path

    def on_batch_end(self, batch, logs={}):

        if batch >= self.max_iterations or self.lr >= self.max_lr:
            self.model.stop_training = True

        # Stop if the loss is exploding
        # elif batch > 1 and self.smoothed_loss > 4 * self.best_loss:
        #     self.model.stop_training = True

        loss = logs.get('loss')

        # Record the best loss
        if self.smoothed_loss < self.best_loss:
            self.best_loss = self.smoothed_loss
            logger.info('current best loss {} for lr {}'.format(self.best_loss, self.lr))



        self.weighted_loss = loss + (1 - self.alpha) * self.weighted_loss
        self.weighted_count = 1 + (1 - self.alpha) * self.weighted_count
        self.smoothed_loss = self.weighted_loss / self.weighted_count

        # Update the lr for the next step
        self.lr *= self.factor
        K.set_value(self.model.optimizer.lr, self.lr)
        self.losses.append(self.smoothed_loss)
        self.lrs.append(self.lr)

        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        logs.update({'smoothed_loss': self.smoothed_loss})
        logs.update({'loss': loss})

    def on_train_end(self, logs=None):
        path = os.path.join(self.log_path, 'lr_finder.csv')
        logger.info('Saving LRFinder results to {}'.format(path))
        with open(path, 'w') as file:
            for loss, lr in zip(self.losses, self.lrs):
                file.write('{}, {}\n'.format(lr, loss))





def step_decay(epoch, initial_lrate=0.05, drop=0.5, epochs_drop=10):
    """
    This drops the learning rate every epochs_drop
    """
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


# copied from https://github.com/bckenstler/CLR
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        logs.update({'iteration': self.trn_iterations})
        #
        # for k, v in logs.items():
        #     self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
