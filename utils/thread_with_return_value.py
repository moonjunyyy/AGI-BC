import threading
import traceback
import sys

class Thread_With_Return_Value(threading.Thread):
    def __init__(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._return = None

    def run(self):
        try:
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)
        except Exception as e:
            print(f"The thread raised an exception: {e}")
            print(traceback.format_exc())
            print(sys.exc_info()[2])
            
    def join(self, *args):
        super().join(*args)
        return self._return