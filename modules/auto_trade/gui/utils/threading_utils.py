import threading
import time
from typing import Callable


class PeriodicUpdater:
    def __init__(self, callback: Callable, interval: int = 30):
        self.callback = callback
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            try:
                self.callback()
            except Exception as e:
                print(f"Error in periodic update: {e}")
            time.sleep(self.interval)
