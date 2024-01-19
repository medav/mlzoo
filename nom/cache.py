import os
import functools
import pickle

from hashlib import md5

def stable_hash(s): return md5(s.encode('utf-8')).hexdigest()


def cache_pickle(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        argss = '_'.join(str(arg) for arg in args)
        filename = f'.cache/{func.__name__}.{stable_hash(argss)}.cache.pkl'
        if os.path.exists(filename):
            # print(f'Loading {func.__name__} from cache')
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            print(f'Regenerating {func.__name__}...')
            ret = func(*args, **kwargs)

            os.makedirs('.cache', exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(ret, f)

            return ret

    return wrapper
