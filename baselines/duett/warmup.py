import pytorch_lightning as pl


class WarmUpCallback(pl.callbacks.Callback):
    def __init__(self, steps=1000, base_lr=None, invsqrt=True, decay=None):
        print('warmup_steps {}, base_lr {}, invsqrt {}, decay {}'.format(steps, base_lr, invsqrt, decay))
        self.warmup_steps = steps
        if decay is None:
            self.decay = steps
        else:
            self.decay = decay

        if base_lr is None:
            self.state = {'steps': 0, 'base_lr': base_lr}
        else:
            self.state = {'steps': 0, 'base_lr': float(base_lr)}

        self.invsqrt = invsqrt

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        optimizers = model.optimizers()

        if self.state['steps'] < self.warmup_steps:
            if type(optimizers) == 'list':
                if self.state['base_lr'] is None:
                    self.state['base_lr'] = [o.param_groups[0]['lr'] for o in optimizers]
                for opt, base in zip(optimizers, self.state['base_lr']):
                    self.set_lr(opt, self.state['steps'] / self.warmup_steps * base)
            else:
                if self.state['base_lr'] is None:
                    self.state['base_lr'] = optimizers.param_groups[0]['lr']
                self.set_lr(optimizers, self.state['steps'] / self.warmup_steps * self.state['base_lr'])
            self.state['steps'] += 1
        elif self.invsqrt:
            if type(optimizers) == 'list':
                if self.state['base_lr'] is None:
                    self.state['base_lr'] = [o.param_groups[0]['lr'] for o in optimizers]
                for opt, base in zip(optimizers, self.state['base_lr']):
                    self.set_lr(opt, base * (self.decay / (self.state['steps'] - self.warmup_steps + self.decay)) ** 0.5)
            else:
                if self.state['base_lr'] is None:
                    self.state['base_lr'] = optimizers.param_groups[0]['lr']
                self.set_lr(optimizers, self.state['base_lr'] * (
                    self.decay / (self.state['steps'] - self.warmup_steps + self.decay)) ** 0.5)
            self.state['steps'] += 1

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()
