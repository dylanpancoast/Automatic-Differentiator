"""This module implements Newton's method for finding zeros of functions."""

import numpy as np
import forwardmode as fd

def jacobian(f, x):
    """Computes the Jacobian of a function from Rn to Rm at a given point.

    Parameters
    ----------
    f : function
        The function whose Jacobian is computed. Must support inputs of type
        forwardmode.DualNumber
    x : int|float|numpy.int64|numpy.float64|list
        The input value at which to compute the Jacobian. If list, elements must
        be of type int|float|numpy.int64|numpy.float64.

    Returns
    -------
    numpy.array
        A two-dimensional array representing the Jacobian vector.
    """

    n = f.__code__.co_argcount
    partials = []
    for i in range(n):
        dir = [0] * n
        dir[i] = 1
        partial = fd.directional_derivative(f, x, dir)

        # Check if partial is iterable, indicating f has more than one output.
        try:
            iterator = iter(partial)
        except TypeError:
            partial = [partial]

        partials.append(partial)

    return np.transpose(np.array(partials))

def find_root(f, x_init, tolerance=1e-10, max_steps=1e4):
    """Uses Newton's method to find roots of a function from Rn to Rn.

    Parameters
    ----------
    f : function
        The function whose roots are found. Must support inputs of type
        forwardmode.DualNumber and have the same number of outputs as inputs.
    x_init : int|float|numpy.int64|numpy.float64|list
        The initial input value to iterate from. If list, elements must
        be of type int|float|numpy.int64|numpy.float64.
    tolerance : int|float
        Algorithm will stop once it finds an input value x such that the
        magnitude of f(x) is less than tolerance.
    max_steps : int
        If the algorithm has not found a root within max_steps iterations,
        the algorithm will give up.

    Returns
    -------
    x : list
        The computed root.

    Raises
    ------
    TimeoutError
        If root is not found within max_steps iterations.
    """

    if type(x_init) in fd._supported_scalars:
        x = np.array([x_init])
    else:
        x = np.array(x_init)
    steps = 0

    if type(f(*x)) in fd._supported_scalars:
        while np.linalg.norm(f(*x)) > tolerance and steps < max_steps:
            x = x - (np.linalg.inv(jacobian(f,x)) @ np.array([f(*x)]))
            steps += 1
        if steps == max_steps:
            raise TimeoutError("Could not find root.")
        return x
    else:
        while np.linalg.norm(f(*x)) > tolerance and steps < max_steps:
            x = x - (np.linalg.inv(jacobian(f,x)) @ np.array(f(*x)))
            steps += 1
        if steps == max_steps:
            raise TimeoutError("Could not find root.")
        return x


if __name__ == '__main__':

    def f(x,y):
        return x**2+y, y-x+1

    print(find_root(f, [100,100]))
