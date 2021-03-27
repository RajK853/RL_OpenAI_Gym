from .base_scheduler import BaseScheduler


class FunctionScheduler(BaseScheduler):
    def __init__(self, func, val_offset=0.0, val_scale=1.0, auto_scale=False, **kwargs):
        super(FunctionScheduler, self).__init__(**kwargs)
        self.func = func
        self.val_offset = self.clip_range[0] if auto_scale else val_offset
        self.val_scale = (self.clip_range[1] - self.clip_range[0]) if auto_scale else val_scale
        self.update_value()

    def calculate_value(self):
        val = self.val_offset + self.val_scale * self.func(self.steps)
        return val
