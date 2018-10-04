import math
import warnings

import numpy
import six

from chainer import backend
from chainer.backends import cuda
from chainer import configuration
from chainer import FunctionNode
from chainer import testing
from chainer import variable


class NondifferentiableError(Exception):
    pass


class _GradientSetter(FunctionNode):
    def __init__(self, grad):
        self.grad = grad

    def forward(self, inputs):
        xp = backend.get_array_module(inputs[0])

        if self.grad is None:
            y0, = inputs
            gy0 = xp.ones_like(y0)
            assert gy0.size == 1

            self.grad = (gy0,)

        # output a 0-sized 1-dim array like inputs
        return xp.empty((0,), dtype=inputs[0].dtype),

    def backward(self, inputs, grad_outputs):
        grad = self.grad

        return tuple(
            None if g is None else variable.as_variable(g)
            for g in grad)


def _copy_arrays(xs):
    xp = backend.get_array_module(*xs)
    return [xp.array(x, order='C', dtype=numpy.float64, copy=True) for x in xs]


def _as_tuple(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _filter_list(lst, ignore_list):
    return [x for x, ignore in six.moves.zip(lst, ignore_list) if not ignore]


def _set_y_grad(ys_var, y_grad_array):
    if y_grad_array is not None:
        if len(ys_var) != len(y_grad_array):
            raise ValueError(
                'Upstream gradients must contain equally many elements as '
                'number of output elements.\n'
                'Actual: {} != {}'.format(len(ys_var), len(y_grad_array)))
        y_var, = _GradientSetter(y_grad_array).apply(ys_var)
    else:
        if len(y) != 1:
            raise ValueError(
                'Function must return a zero-dimensional array of length 1 '
                'if the upstream gradient is `None`.\n'
                'Actual: {} != 1'.format(len(ys_var)))
        y_var, = _GradientSetter(None).apply(ys_var)
        y_grad_array = (1,)
    return y_var, y_grad_array


def _clear_grads(xs):
    for x in xs:
        x.grad_var = None

def _numerical_grad(
        func, inputs, grad_outputs, eps=1e-3,
        detect_nondifferentiable=False, diff_atol=0, diff_rtol=1e-2,
        center_outputs=None):
    assert eps > 0
    gpu = any(isinstance(x, cuda.ndarray) for x in grad_outputs)
    cpu = any(isinstance(x, numpy.ndarray) for x in grad_outputs)

    if gpu and cpu:
        raise RuntimeError('Do not mix GPU and CPU arrays in `numerical_grad`')
    elif gpu:
        xp = cuda.cupy
        numerical_grad_kernel_1 = cuda.reduce(
            'T y1, T y2, U gy, T eps', 'V gxi',
            '(y1 - y2) * gy', 'a + b', 'gxi += a / (eps * 2)', '0',
            'numerical_grad_kernel_1'
        )
        numerical_grad_kernel_3 = cuda.reduce(
            'T y1, T y2, T y3, T y4, U gy, T eps', 'V gxi',
            '(-y1 + 8 * y2 - 8 * y3 + y4) * gy',
            'a + b', 'gxi += a / (eps * 6)', '0',
            'numerical_grad_kernel_3'
        )
    else:
        xp = numpy

    grads = [xp.zeros(x.shape, numpy.float64) for x in inputs]

    if detect_nondifferentiable:
        ys = center_outputs
        nout = len(ys)
        ys_shapes = [y.shape for y in ys]
        ys_sizes = numpy.array([y.size for y in ys])
        ys_cumsizes = numpy.cumsum(ys_sizes)

    # Evaluate func at a single input
    def eval_func(x, i, delta, orig):
        return _copy_arrays(func(orig + delta))

    # An iteration on a single input displacement
    def iterate_single_input(i_in, x, orig_x, i):
        orig = orig_x[i]
        # `yss` holds a list of output arrays for each of 2 or 5 sampling
        # points.
        if detect_nondifferentiable:
            yss = [
                eval_func(x, i, -eps * 1., orig),
                eval_func(x, i, -eps * .5, orig),
                ys0,
                eval_func(x, i, +eps * .5, orig),
                eval_func(x, i, +eps * 1., orig),
            ]
        else:
            yss = [
                eval_func(x, i, -eps * 1, orig),
                eval_func(x, i, +eps * 1, orig),
            ]

        if detect_nondifferentiable:
            # Detect non-differentiable point by quadratic fitting

            # Check for non-finite output.
            # If any single element in the output arrays has different
            # finiteness among sampled points, that means this is a
            # non-differentiable point.
            # If the function consistently generates non-finite values
            # around the point, we do not treat the point as
            # non-differentiable.
            # (Example: x<0 region for the logarithm function)
            any_nonfinite = False
            for i_out in range(nout):
                isfinites = [xp.isfinite(ys[i_out]) for ys in yss]
                if any((isfinites[0] != isfinites[i]).any()
                       for i in range(1, len(yss))):
                    s = six.StringIO()
                    s.write(
                        'Tried to compute the numeric gradient on a '
                        'non-differentiable point.\n\n')
                    s.write('i_in: {}\n'.format(i_in))
                    s.write('i_out: {}\n'.format(i_out))
                    s.write('x: {}\n'.format(inputs[i_in]))
                    s.write('index on x: {}\n'.format(i))
                    s.write('eps: {}\n'.format(eps))
                    s.write('y[x-eps  ]: {}\n'.format(yss[0][i_out]))
                    s.write('y[x-eps/2]: {}\n'.format(yss[1][i_out]))
                    s.write('y[x      ]: {}\n'.format(yss[2][i_out]))
                    s.write('y[x+eps/2]: {}\n'.format(yss[3][i_out]))
                    s.write('y[x+eps  ]: {}\n'.format(yss[4][i_out]))
                    raise NondifferentiableError(s.getvalue())

                any_nonfinite |= not all(_.all() for _ in isfinites)

            if not any_nonfinite:
                # Stack flattenend outputs to make (5, *)-shaped 2D array
                ystack = xp.vstack(
                    [xp.hstack([y.ravel() for y in ys]) for ys in yss])
                assert ystack.ndim == 2 and ystack.shape[0] == len(yss)
                # Fit to quadratic
                if gpu:
                    ystack = ystack.get()
                polyfit = numpy.polynomial.polynomial.polyfit
                _, (residuals, _, _, _) = polyfit(
                    range(len(yss)), ystack, deg=2, full=True)
                if gpu:
                    residuals = xp.array(residuals)
                residuals = xp.sqrt(residuals / len(yss))

                # Check for error for each output array
                for i_out in range(nout):
                    size = sizes[i_out]
                    cumsize = cumsizes[i_out]
                    shape = shapes[i_out]
                    # TODO(niboshi): The following two lines could be
                    # rewritten using xp.stack, which is supported in
                    # NumPy>=1.10
                    ymax = xp.concatenate(
                        [ys[i_out][None] for ys in yss]).max(axis=0)
                    ymin = xp.concatenate(
                        [ys[i_out][None] for ys in yss]).min(axis=0)
                    # Restore the shape of flattened residual
                    res = residuals[cumsize - size:cumsize]
                    res = res.reshape(shape)
                    det = xp.asarray(
                        diff_atol + diff_rtol * (ymax - ymin) < res)
                    # Constant output = not nondifferentiable
                    det[ymax == ymin] = False
                    if det.any():
                        s = six.StringIO()
                        s.write(
                            'Tried to compute the numeric gradient on a '
                            'non-differentiable point.\n\n')
                        s.write('i_in: {}\n'.format(i_in))
                        s.write('i_out: {}\n'.format(i_out))
                        s.write('x: {}\n'.format(inputs[i_in]))
                        s.write('index on x: {}\n'.format(i))
                        s.write('eps: {}\n'.format(eps))
                        s.write('diff_rtol: {}\n'.format(diff_rtol))
                        s.write('diff_atol: {}\n'.format(diff_atol))
                        s.write('ymax: {}\n'.format(ymax))
                        s.write('ymin: {}\n'.format(ymin))
                        s.write(
                            'diff_atol + diff_rtol * (ymax-ymin): {}\n'.format(
                                diff_atol + diff_rtol * (ymax - ymin)))
                        s.write('fitting errors: {}\n'.format(res))
                        s.write('y[x-eps  ]: {}\n'.format(yss[0][i_out]))
                        s.write('y[x-eps/2]: {}\n'.format(yss[1][i_out]))
                        s.write('y[x      ]: {}\n'.format(yss[2][i_out]))
                        s.write('y[x+eps/2]: {}\n'.format(yss[3][i_out]))
                        s.write('y[x+eps  ]: {}\n'.format(yss[4][i_out]))
                        raise NondifferentiableError(s.getvalue())

        # Calculate numerical gradient
        for i_out, gy in enumerate(grad_outputs):
            if gy is None:
                continue
            gpu_ = (gpu and
                    all(isinstance(ys[i_out], cuda.ndarray)
                        for ys in yss))
            if len(yss) == 2:  # 1st order
                y0 = yss[0][i_out]
                y1 = yss[1][i_out]
                if gpu_:
                    numerical_grad_kernel_1(
                        y1, y0, xp.asarray(gy), eps, gx[i])
                else:
                    dot = ((y1 - y0) * gy).sum()
                    gx[i] += dot / (2 * eps)
            elif len(yss) == 5:  # 3rd order
                y0 = yss[0][i_out]
                y1 = yss[1][i_out]
                y2 = yss[3][i_out]
                y3 = yss[4][i_out]
                if gpu_:
                    numerical_grad_kernel_3(
                        y3, y2, y1, y0, gy, eps, gx[i])
                else:
                    num = -y3 + 8 * y2 - 8 * y1 + y0
                    dot = (num * gy).sum()
                    gx[i] += dot / (6 * eps)
            else:
                assert False

    # Calculate numeric gradient
    with configuration.using_config('type_check', False):
        for i_in, (x, gx) in enumerate(six.moves.zip(inputs, grads)):
            orig_x = x.copy()  # hold original value
            for i in numpy.ndindex(x.shape):
                iterate_single_input(i_in, x, orig_x, i)

    return [g.astype(x.dtype, copy=False)
            for g, x in six.moves.zip(grads, inputs)]

def numerical_grad(
        f, inputs, grad_outputs, eps=1e-3,
        detect_nondifferentiable=False, diff_atol=0, diff_rtol=1e-2,
        center_outputs=None):

    assert eps > 0
    for x in inputs:
        if x.dtype.kind != 'f':
            raise RuntimeError(
                'The dtype of input arrays must be kind of float')

    inputs = tuple(inputs)
    grad_outputs = tuple(grad_outputs)
    gpu = any(isinstance(x, cuda.ndarray) for x in inputs + grad_outputs)
    cpu = any(isinstance(x, numpy.ndarray) for x in inputs + grad_outputs)

    if gpu and cpu:
        raise RuntimeError('Do not mix GPU and CPU arrays in `numerical_grad`')

    if gpu:
        xp = cuda.cupy
        numerical_grad_kernel_1 = cuda.reduce(
            'T y1, T y2, U gy, T eps', 'V gxi',
            '(y1 - y2) * gy', 'a + b', 'gxi += a / (eps * 2)', '0',
            'numerical_grad_kernel_1'
        )
        numerical_grad_kernel_3 = cuda.reduce(
            'T y1, T y2, T y3, T y4, U gy, T eps', 'V gxi',
            '(-y1 + 8 * y2 - 8 * y3 + y4) * gy',
            'a + b', 'gxi += a / (eps * 6)', '0',
            'numerical_grad_kernel_3'
        )
    else:
        xp = numpy
    grads = [xp.zeros(x.shape, numpy.float64) for x in inputs]

    if detect_nondifferentiable:
        if center_outputs is None:
            ys0 = _copy_arrays(f())
        else:
            ys0 = center_outputs
        nout = len(ys0)
        shapes = [_.shape for _ in ys0]
        sizes = numpy.array([_.size for _ in ys0])
        cumsizes = numpy.cumsum(sizes)

    # Evaluate func at a single input
    def eval_func(x, i, delta, orig):
        x[i] = orig + delta
        y = _copy_arrays(f())
        x[i] = orig
        return y

    # An iteration on a single input displacement
    def iterate_single_input(i_in, x, orig_x, i):
        orig = orig_x[i]
        # `yss` holds a list of output arrays for each of 2 or 5 sampling
        # points.
        if detect_nondifferentiable:
            yss = [
                eval_func(x, i, -eps * 1., orig),
                eval_func(x, i, -eps * .5, orig),
                ys0,
                eval_func(x, i, +eps * .5, orig),
                eval_func(x, i, +eps * 1., orig),
            ]
        else:
            yss = [
                eval_func(x, i, -eps * 1, orig),
                eval_func(x, i, +eps * 1, orig),
            ]

        if detect_nondifferentiable:
            # Detect non-differentiable point by quadratic fitting

            # Check for non-finite output.
            # If any single element in the output arrays has different
            # finiteness among sampled points, that means this is a
            # non-differentiable point.
            # If the function consistently generates non-finite values
            # around the point, we do not treat the point as
            # non-differentiable.
            # (Example: x<0 region for the logarithm function)
            any_nonfinite = False
            for i_out in range(nout):
                isfinites = [xp.isfinite(ys[i_out]) for ys in yss]
                if any((isfinites[0] != isfinites[i]).any()
                       for i in range(1, len(yss))):
                    s = six.StringIO()
                    s.write(
                        'Tried to compute the numeric gradient on a '
                        'non-differentiable point.\n\n')
                    s.write('i_in: {}\n'.format(i_in))
                    s.write('i_out: {}\n'.format(i_out))
                    s.write('x: {}\n'.format(inputs[i_in]))
                    s.write('index on x: {}\n'.format(i))
                    s.write('eps: {}\n'.format(eps))
                    s.write('y[x-eps  ]: {}\n'.format(yss[0][i_out]))
                    s.write('y[x-eps/2]: {}\n'.format(yss[1][i_out]))
                    s.write('y[x      ]: {}\n'.format(yss[2][i_out]))
                    s.write('y[x+eps/2]: {}\n'.format(yss[3][i_out]))
                    s.write('y[x+eps  ]: {}\n'.format(yss[4][i_out]))
                    raise NondifferentiableError(s.getvalue())

                any_nonfinite |= not all(_.all() for _ in isfinites)

            if not any_nonfinite:
                # Stack flattenend outputs to make (5, *)-shaped 2D array
                ystack = xp.vstack(
                    [xp.hstack([y.ravel() for y in ys]) for ys in yss])
                assert ystack.ndim == 2 and ystack.shape[0] == len(yss)
                # Fit to quadratic
                if gpu:
                    ystack = ystack.get()
                polyfit = numpy.polynomial.polynomial.polyfit
                _, (residuals, _, _, _) = polyfit(
                    range(len(yss)), ystack, deg=2, full=True)
                if gpu:
                    residuals = xp.array(residuals)
                residuals = xp.sqrt(residuals / len(yss))

                # Check for error for each output array
                for i_out in range(nout):
                    size = sizes[i_out]
                    cumsize = cumsizes[i_out]
                    shape = shapes[i_out]
                    # TODO(niboshi): The following two lines could be
                    # rewritten using xp.stack, which is supported in
                    # NumPy>=1.10
                    ymax = xp.concatenate(
                        [ys[i_out][None] for ys in yss]).max(axis=0)
                    ymin = xp.concatenate(
                        [ys[i_out][None] for ys in yss]).min(axis=0)
                    # Restore the shape of flattened residual
                    res = residuals[cumsize - size:cumsize]
                    res = res.reshape(shape)
                    det = xp.asarray(
                        diff_atol + diff_rtol * (ymax - ymin) < res)
                    # Constant output = not nondifferentiable
                    det[ymax == ymin] = False
                    if det.any():
                        s = six.StringIO()
                        s.write(
                            'Tried to compute the numeric gradient on a '
                            'non-differentiable point.\n\n')
                        s.write('i_in: {}\n'.format(i_in))
                        s.write('i_out: {}\n'.format(i_out))
                        s.write('x: {}\n'.format(inputs[i_in]))
                        s.write('index on x: {}\n'.format(i))
                        s.write('eps: {}\n'.format(eps))
                        s.write('diff_rtol: {}\n'.format(diff_rtol))
                        s.write('diff_atol: {}\n'.format(diff_atol))
                        s.write('ymax: {}\n'.format(ymax))
                        s.write('ymin: {}\n'.format(ymin))
                        s.write(
                            'diff_atol + diff_rtol * (ymax-ymin): {}\n'.format(
                                diff_atol + diff_rtol * (ymax - ymin)))
                        s.write('fitting errors: {}\n'.format(res))
                        s.write('y[x-eps  ]: {}\n'.format(yss[0][i_out]))
                        s.write('y[x-eps/2]: {}\n'.format(yss[1][i_out]))
                        s.write('y[x      ]: {}\n'.format(yss[2][i_out]))
                        s.write('y[x+eps/2]: {}\n'.format(yss[3][i_out]))
                        s.write('y[x+eps  ]: {}\n'.format(yss[4][i_out]))
                        raise NondifferentiableError(s.getvalue())

        # Calculate numerical gradient
        for i_out, gy in enumerate(grad_outputs):
            if gy is None:
                continue
            gpu_ = (gpu and
                    all(isinstance(ys[i_out], cuda.ndarray)
                        for ys in yss))
            if len(yss) == 2:  # 1st order
                y0 = yss[0][i_out]
                y1 = yss[1][i_out]
                if gpu_:
                    numerical_grad_kernel_1(
                        y1, y0, xp.asarray(gy), eps, gx[i])
                else:
                    dot = ((y1 - y0) * gy).sum()
                    gx[i] += dot / (2 * eps)
            elif len(yss) == 5:  # 3rd order
                y0 = yss[0][i_out]
                y1 = yss[1][i_out]
                y2 = yss[3][i_out]
                y3 = yss[4][i_out]
                if gpu_:
                    numerical_grad_kernel_3(
                        y3, y2, y1, y0, gy, eps, gx[i])
                else:
                    num = -y3 + 8 * y2 - 8 * y1 + y0
                    dot = (num * gy).sum()
                    gx[i] += dot / (6 * eps)
            else:
                assert False

    # Calculate numeric gradient
    with configuration.using_config('type_check', False):
        for i_in, (x, gx) in enumerate(six.moves.zip(inputs, grads)):
            orig_x = x.copy()  # hold original value
            for i in numpy.ndindex(x.shape):
                iterate_single_input(i_in, x, orig_x, i)

    return [g.astype(x.dtype, copy=False)
            for g, x in six.moves.zip(grads, inputs)]


def assert_allclose(x, y, atol=1e-5, rtol=1e-4, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both CPU and GPU arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    warnings.warn(
        'chainer.gradient_check.assert_allclose is deprecated.'
        'Use chainer.testing.assert_allclose instead.',
        DeprecationWarning)
    testing.assert_allclose(x, y, atol, rtol, verbose)



# TODO(hvy): Temp
def _compute_backward_grad(
        func, x_arrays, y_grad_arrays, params):

    # Input variables.
    x_vars = [variable.Variable(x) for x in x_arrays]

    # Output variables and ndarrays.
    y_vars = func(*x_vars)
    y_vars = _as_tuple(y_vars)

    # All creators of `y` need to be the same because we only call
    # `y[0].backward` to call `backward` method of the creator.
    # To do so we need to insert a dummy function `_GradientSetter` to the
    # computational graph.
    # Note that `func` may not be a `Function` object.
    y0_var, y_grad_arrays = _set_y_grad(y_vars, y_grad_arrays)

    # Clear gradients which may exist if func calls backward inside of itself.
    _clear_grads(x_vars)
    _clear_grads(params)

    # We only need to call `backward` for one result `Variable`.
    # `Variable.backward` method calls `Function.backward` of its creator.
    y0_var.backward()

    return [x.grad for x in x_vars], [p.grad for p in params], y0_var



def check_backward(
        func, x_arrays, y_grad_arrays, params=(),
        eps=1e-3, atol=1e-5, rtol=1e-4, no_grads=None, dtype=None,
        detect_nondifferentiable=False):
    x_arrays = _as_tuple(x_arrays)
    y_grad_arrays = _as_tuple(y_grad_arrays)
    params = _as_tuple(params)

    # Check inputs.
    if dtype is not None and numpy.dtype(dtype).kind != 'f':
        raise ValueError('`dtype` is allowed only float type')
    if no_grads is None:
        no_grads = [x.dtype.kind != 'f' for x in x_arrays]
    else:
        if len(no_grads) != len(x_arrays):
            raise ValueError(
                'Length of no_grads param and xs should be same.\n'
                'Actual: {0} != {1}'.format(len(no_grads), len(x_arrays)))


    # Compute and keep the gradient arrays of params which may be
    # overwritten by func
    grads, param_grads, y_arrays = _compute_backward_grad(
        func, x_arrays, y_grad_arrays, params)

    # Remove None grads corresponding to no_grads.
    assert len(grads) == len(no_grads)
    for no_grad, grad in six.moves.zip(no_grads, grads):
        if no_grad and grad is not None:
            raise RuntimeError(
                'gradient of int variable must be None')

    # Check for possible early return.
    if len(x_arrays) - no_grads.count(True) + len(params) == 0:
        # When there is no float variables, we need not to check gradient
        # values
        return

    grads = [g for g, ng in six.moves.zip(grads, no_grads) if not ng]
    grads += list(param_grads)

    arrays = tuple([x for x, ng in six.moves.zip(x_arrays, no_grads) if not ng]) + tuple([p.array for p in params])
    if dtype is None:
        casted_arrays = arrays
    else:
        casted_arrays = [a.astype(dtype, copy=False) for a in arrays]

    # Create perturbation directions.
    # The direction vector is normalized in order to keep the scale of
    # differentiation error invariant with respect to the number of input
    # dimensions. Ideally, the scale of the curvature with respect to each
    # input dimension should be taken into account, but we ignore the
    # differences and assume that the curvature is uniform with respect to all
    # the input dimentions.
    xp = backend.get_array_module(*x_arrays)
    directions = [xp.random.normal(size=a.shape) for a in arrays]
    norm = math.sqrt(sum([xp.square(d).sum() for d in directions]))
    if norm != 0:
        # norm could be zero if input arrays are 0-sized.
        scale = 1. / norm
        directions = [d * scale for d in directions]

    assert len(arrays) == len(grads)
    assert len(arrays) == len(casted_arrays)
    assert len(arrays) == len(directions)

    def func_delta(delta):
        # This functions is called twice in `numerical_grad`.
        # `delta` is `epsilon` or `-epsilon` in these calls.
        # See the document of `numerical_grad`.
        idx = 0
        inputs = []
        for i in range(len(x_arrays)):
            if no_grads[i]:
                a = x_arrays[i]
            else:
                a = casted_arrays[idx]
                a_dtype = a.dtype
                a = a.astype('d') + delta * directions[idx]
                a = a.astype(a_dtype)
                idx += 1
            if numpy.isscalar(a):
                a = xp.array(a)
            inputs.append(variable.Variable(a))

        #y_vars = func(*tuple([variable.Variable((a.astype('d') + delta * d).astype(a.dtype)) for a, d in six.moves.zip(casted_arrays, directions)]))
        y_vars = func(*inputs)
        y_vars = _as_tuple(y_vars)
        y_arrays = tuple(y.array for y in y_vars)
        return y_arrays

    delta = xp.array(0., 'd')
    gx, = _numerical_grad(
        func_delta, (delta,), y_grad_arrays, eps=eps,
        detect_nondifferentiable=detect_nondifferentiable,
        center_outputs=y_arrays)

    gx_accum = 0
    for grad, direction in six.moves.zip(grads, directions):
        if grad is not None:
            gx_accum += (grad.astype('d') * direction).sum()

    try:
        testing.assert_allclose(gx, gx_accum, atol=atol, rtol=rtol)
    except AssertionError as e:
        f = six.StringIO()
        f.write('check_backward failed (eps={} atol={} rtol={})\n'.format(
            eps, atol, rtol))
        for i, x_ in enumerate(x_arrays):
            f.write('inputs[{}]:\n'.format(i))
            f.write('{}\n'.format(variable.Variable(x_)))
        for i, gy_ in enumerate(y_grad_arrays):
            f.write('grad_outputs[{}]:\n'.format(i))
            f.write('{}\n'.format(variable.Variable(gy_)))
        for i, d_ in enumerate(directions):
            f.write('directions[{}]:\n'.format(i))
            f.write('{}\n'.format(variable.Variable(d_)))
        f.write('gradients (numeric):  {}\n'.format(gx))
        f.write('gradients (backward): {}\n'.format(gx_accum))
        f.write('\n')
        f.write(str(e))
        raise AssertionError(f.getvalue())


def check_double_backward(func, x_array, y_grad_array, x_grad_grad, params=(),
                          params_grad_grad=(), eps=1e-3, atol=1e-4, rtol=1e-3,
                          no_grads=None, dtype=None,
                          detect_nondifferentiable=False):
    """Test twice differentiation of a given procedure.

    This function automatically checks if the backward procedure of ``func``
    is correctly implemented for further differentiation. It first computes the
    gradient of ``func`` w.r.t. its inputs in the same way as
    :func:`~chainer.gradient_check.check_backward`. This function then further
    invokes the backward procedure against the gradient variables, starting
    from the initial gradient given by ``x_grad_grad``. It also computes the
    second gradient using :func:`~chainer.gradient_check.numerical_grad`. The
    resulting gradients are compared to confirm if the second-order gradients
    are approximately correct.

    Note that this function **DOES NOT** check if the first-order
    differentiation is correct; the numerical gradient assumes that the
    first-order gradient given by the usual :meth:`chainer.Variable.backward`
    is correct. The implementation of each differentiable function should be
    tested by :func:`~chainer.gradient_check.check_backward` first, and then
    should be tested by this function if neccessary.

    For the details of the arguments, see
    :func:`~chainer.gradient_check.check_backward`. The additional arguments
    ``x_grad_grad`` and ``params_grad_grad`` are (tuples of)
    :class:`~chainer.Variable` (s) that include the initial gradient
    corresponding to the first-order gradient of each input and parameter. Note
    that the default error tolerance ``atol`` and ``rtol`` are slightly larger
    than those of :func:`~chainer.gradient_check.check_backward` because the
    numerical gradients of the second order differentiation are less accurate
    than those of the first order gradients.

    """
    x_array = _as_tuple(x_array)
    params = _as_tuple(params)
    y_grad_array = _as_tuple(y_grad_array)
    x_grad_grad = _as_tuple(x_grad_grad)
    params_grad_grad = _as_tuple(params_grad_grad)
    n_x = len(x_array)

    first_order_no_grads = [x.dtype.kind != 'f' for x in x_array]

    def first_order_grad(*inputs):
        xs = inputs[:n_x]
        gys = inputs[n_x:]

        y = _as_tuple(func(*xs))

        # Let all elements of y share the same creator.
        # See the comment in check_backward.
        y, _ = _set_y_grad(y, gys)

        y.backward(enable_double_backprop=True)

        gxs = []
        for skip, x in six.moves.zip(first_order_no_grads, xs):
            if skip:
                if x.grad is not None:
                    raise RuntimeError(
                        'gradient of int variable must be None')
            else:
                if x.grad is None:
                    raise RuntimeError(
                        'gradients of some arguments are not calculated')
                gxs.append(x.grad_var)

        return tuple(gxs + [p.grad_var for p in params])

    inputs = x_array + y_grad_array
    grad_grad = x_grad_grad + params_grad_grad
    try:
        check_backward(first_order_grad, inputs, grad_grad, params=params,
                       eps=eps, atol=atol, rtol=rtol, no_grads=no_grads,
                       dtype=dtype,
                       detect_nondifferentiable=detect_nondifferentiable)
    except AssertionError as e:
        f = six.StringIO()
        f.write('check_double_backward failed '
                '(eps={} atol={} rtol={})\n'.format(eps, atol, rtol))
        for i, x_ in enumerate(x_array):
            f.write('input[{}]:\n'.format(i))
            f.write('{}\n'.format(x_))
        for i, gy_ in enumerate(y_grad_array):
            f.write('grad_output[{}]:\n'.format(i))
            f.write('{}\n'.format(gy_))
        for i, ggx_ in enumerate(x_grad_grad):
            f.write('grad_grad_input[{}]:\n'.format(i))
            f.write('{}\n'.format(ggx_))
        for i, ggp_ in enumerate(params_grad_grad):
            f.write('grad_grad_param[{}]:\n'.format(i))
            f.write('{}\n'.format(ggp_))
        f.write('\n')
        f.write(str(e))
        raise AssertionError(f.getvalue())
