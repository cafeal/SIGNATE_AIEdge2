import functools
import tensorflow as tf

def func_name_scope(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tf.name_scope(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator