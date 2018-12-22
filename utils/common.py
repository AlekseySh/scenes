def args_to_text(args):
    text = ''
    for arg in vars(args):
        string = f'{arg}: {getattr(args, arg)} \n'
        text += string
    return text


class OnlineAvg:

    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, new_x):
        self.n += 1
        self.avg = (self.avg * (self.n - 1) + new_x) / self.n


class Stopper:

    def __init__(self, n_obs, delta):
        self.n_obs = n_obs
        self.delta = delta

        self.cur_val = None
        self.max_val = 0
        self.num_fails = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self._count_fails()
        self._update_max()

    def _count_fails(self):
        if self.cur_val - self.max_val <= self.delta:
            self.num_fails += 1
        else:
            self.num_fails = 0

    def check_criterion(self):
        is_stop = self.num_fails == self.n_obs
        return is_stop

    def _update_max(self):
        if self.max_val < self.cur_val:
            self.max_val = self.cur_val
