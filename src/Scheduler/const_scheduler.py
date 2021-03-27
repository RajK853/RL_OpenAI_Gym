from .base_scheduler import BaseScheduler


class ConstantScheduler(BaseScheduler):
    def __init__(self, value, **kwargs):
        kwargs["clip_range"] = (value, value)    # Override the clip_range as it is useless in this scheduler
        super(ConstantScheduler, self).__init__(**kwargs)
        self.value = value

    def calculate_value(self):
        return self.value

    def update_value(self):
        pass

    def increment(self, step=1):
        self.steps += step
