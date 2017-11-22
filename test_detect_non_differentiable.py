import numpy as np
from chainer import gradient_check



def func(x):
    # return abs(x)
    # return -3 * x ** 2 + 2 * x + 1
    return np.sign(x), np.sign(-x), np.sign(x)


def check_positive(inputs, eps):
    # Should be non-differentiable
    grad_outputs = [
        np.random.uniform(-1, 1, _.shape).astype(_.dtype) for _ in inputs]

    def f():
        out = func(*inputs)
        # print('out', out.shape)
        return out

    try:
        gradient_check.numerical_grad(
            f, inputs, grad_outputs, eps=eps,
            detect_nondifferentiable=True)
    except gradient_check.NondifferentiableError:
        pass
    else:
        raise AssertionError(
            'Function is expected to be non-differentiable, '
            'but determined to be differentiable.\n\n'
            'eps: {}\n'
            'inputs: {}\n'
            'xp: {}\n'
            ''.format(
                eps, inputs, np.__name__))

def check_negative(self, inputs, eps):
    # Should be differentiable
    grad_outputs = [
        np.random.uniform(-1, 1, _.shape).astype(_.dtype) for _ in inputs]

    def f():
        return func(*inputs)

    try:
        gradient_check.numerical_grad(
            f, inputs, grad_outputs, eps=eps,
            detect_nondifferentiable=True)
    except gradient_check.NondifferentiableError as e:
        raise AssertionError(
            'Function `{}` is expected to be differentiable, '
            'but determined to be non-differentiable.\n\n'
            'eps: {}\n'
            'inputs: {}\n'
            'xp: {}\n\n'
            '{}: {}'
            ''.format(
                func_name, eps, inputs, np.__name__,
                e.__class__.__name__, e))


# inputs = np.array([1], dtype='f'),
# inputs = np.random.rand(100,).astype('f'),
inputs = [[1, 2], [-3, 5]]
# inputs = [0]
inputs = [np.asarray(inputs).astype(np.float32)]
print('IN', inputs)
check_positive(inputs, eps=1e-3)
