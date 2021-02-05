from time import time
from collections import deque


class ProgressBar:
    """
    Progress bar
    """
    DEFAULT_DONE_CHAR = "█"
    DEFAULT_UNDONE_CHAR = "░"

    def __init__(self, total_iter, title="Progress", max_bar_size=50, display_interval=1, auto_reset=True, change_line_at_reset=True, 
        done_char=DEFAULT_DONE_CHAR, undone_char=DEFAULT_UNDONE_CHAR, info_text=""):
        """
        Constructor function
        :param total_iter: (int) Max iteration size for 100%
        :param title: (str) Title for the progress bar
        :param max_bar_size: (int) Maximum character size for 100%
        :param display_interval: (int) Display every given iterations
        :param change_line_at_reset: (bool) Change line when reset
        :param done_char: (str) Character for completed progress
        :param undone_char: (str) Character for left progress
        """
        self.total_iter = int(total_iter)
        self.title = title
        self.max_bar_size = max_bar_size
        self.display_interval = display_interval
        self.change_line = change_line_at_reset
        self.auto_reset = auto_reset
        self.done_char = done_char
        self.undone_char = undone_char
        self.info_text = info_text
        self._current_iter = 0
        self._progress_ratio = 0.0
        self.dt_queue = deque(maxlen=50)
        self._last_time = time()
        self._time_left_min = 0
        self._time_left_sec = 0
        self._avg_time_per_iter = 0

    def reset(self):
        self._current_iter = 0
        if self.change_line:
            print()

    @property
    def done_bar(self):
        bar_size = round(self._progress_ratio * self.max_bar_size)
        current_bar = bar_size * self.done_char
        return current_bar

    def calculate_progress(self):
        self._progress_ratio = self._current_iter / self.total_iter

    def calculate_time_info(self):
        self._avg_time_per_iter = sum(self.dt_queue)/len(self.dt_queue)
        time_left = (self.total_iter - self._current_iter)*self._avg_time_per_iter
        self._time_left_min, self._time_left_sec = divmod(time_left, 60)

    def display(self, **info_dict):
        """
        Display progress bar with additional text at the right of the progress bar
        :param info_dict: (dict) Dictionary of additional fields to display
        :return: None
        """
        self.calculate_progress()
        self.calculate_time_info()
        done_percent = self._progress_ratio*100
        info_text = self.info_text.format(**info_dict)
        # display_text = [title] [progress_bar] [progress_percent] [time left (MM:SS)] [additional text]
        display_text = (f"\r{self.title} {self.done_bar:{self.undone_char}<{self.max_bar_size}} {done_percent:>5.2f}%"
                        f" | mean time per iter: {self._avg_time_per_iter:0>.2f} s | time left: {self._time_left_min:0>2.0f}:{self._time_left_sec:0>2.0f}"
                        f" | {info_text} ")
        print(display_text, flush=True, end="")

    def step(self, num=1, **info_dict):
        """
        Take given number of steps
        :param num: (int) Step number
        :param info_dict: (dict) Dictionary of additional fields to display
        :return: None
        """
        current_time = time()
        self.dt_queue.append(current_time - self._last_time)
        self._last_time = current_time
        self._current_iter = min(self._current_iter + num, self.total_iter)
        if (self._current_iter == self.total_iter) or ((self._current_iter % self.display_interval) == 0):
            self.display(**info_dict)
        if self._current_iter == self.total_iter and self.auto_reset:
            self.reset()
