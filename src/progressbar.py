class ProgressBar:
    """
    Progress bar
    """
    def __init__(self, total_iter, display_text="Progress", max_bar_size=50, display_interval=1,
                 change_line_at_reset=True, done_char="█", undone_char="░"):
        """
        Constructor function
        :param total_iter: (int) Max iteration size for 100%
        :param display_text: (str) Display text at left of progress bar
        :param max_bar_size: (int) Maximum character size for 100%
        :param display_interval: (int) Display every given iterations
        :param change_line_at_reset: (bool) Change line when reset
        :param done_char: (str) Character for completed progress
        :param undone_char: (str) Character for left progress
        """
        self.total_iter = total_iter
        self.display_text = display_text
        self.max_bar_size = max_bar_size
        self.display_interval = display_interval
        self.change_line = change_line_at_reset
        self.done_char = done_char
        self.undone_char = undone_char
        self._current_iter = 0

    def reset(self):
        self._current_iter = 0
        if self.change_line:
            print()

    def display(self, add_text):
        """
        Display progress bar with additional text at the right of the progress bar
        :param add_text: (str) Additional text
        :return: None
        """
        progress_ratio = self._current_iter / self.total_iter
        bar_size = int(progress_ratio * self.max_bar_size)
        current_bar = bar_size * self.done_char
        print(f"\r{self.display_text} {current_bar:{self.undone_char}<{self.max_bar_size}}"
              f" {100 * progress_ratio:>6.2f}% ({self._current_iter:>3}/{self.total_iter:<3}) {add_text}",
              flush=True, end="")

    def step(self, num=1, add_text=""):
        """
        Take given number of steps
        :param num: (int) Step number
        :param add_text: (str) Additional text
        :return: None
        """
        self._current_iter = min(self._current_iter + num, self.total_iter)
        if (self._current_iter == self.total_iter) or (not self._current_iter % self.display_interval):
            self.display(add_text)
        if self._current_iter == self.total_iter:
            self.reset()
