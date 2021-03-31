from functools import wraps
import time


class Monitor(object):

    def __init__(self):
        self.data = {}

    def record(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def reset(self, msg_type=None):
        result = self.data.copy()
        if msg_type is not None:
            result['msg_type'] = msg_type
        self.data = {}
        return result

    @classmethod
    def dealy(cls, moni=None):
        def decorate(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                a = func(*args, **kwargs)
                if moni is None:
                    if getattr(args[0], 'moni', None) is None:
                        args[0].moni = Monitor()
                    args[0].moni.record(action_dealy=time.time() - start_time)
                else:
                    moni.record(action_dealy=time.time() - start_time)
                return a

            return wrapper

        return decorate


if __name__ == '__main__':
    moni = Monitor()


    class A():
        def __init__(self):
            pass

        @Monitor.dealy()
        def haha(self, x):
            time.sleep(1)
            return x


    a = A()
    x = a.haha(5)
    print(x)

    print(a.moni.data)
    print(a.moni.reset("halo"))

    x = a.haha(6)
    print(x)

    print(a.moni.data)
    print(a.moni.reset("halo"))
