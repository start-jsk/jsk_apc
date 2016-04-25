def separator(wrapped):
    def inner(*args, **kwargs):
        print '\n\n\n\n\n\n=======BEGIN==========='
        ret = wrapped(*args, **kwargs)
        print '=========END=========\n\n\n\n\n\n'
        return ret
    return inner
