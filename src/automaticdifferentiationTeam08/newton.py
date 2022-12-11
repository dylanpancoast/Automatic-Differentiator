"""This module implements Newton's method for finding zeros of functions."""

import numpy as np
import forwardmode as fd

def _jacobian(f, x):
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
        dir = np.array([0 for i in range(n)])
        dir[i] = 1
        partial = fd.directional_derivative(f, x, dir)

        # Check if partial is iterable, indicating f has more than one output.
        try:
            iterator = iter(partial)
        except TypeError:
            partial = [partial]

        partials.append(partial)

    return np.transpose(np.array(partials))

def find_root(f, x, tolerance=1e-10, max_steps=1e4):
    """Uses Newton's method to find roots of a function from Rn to Rn.

    Parameters
    ----------
    f : function
        The function whose roots are found. Must support inputs of type
        forwardmode.DualNumber and have the same number of outputs as inputs.
    x : int|float|numpy.int64|numpy.float64|list
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

    if type(x) in fd._supported_scalars:
        x = np.array([x])
    if not isinstance(x, np.ndarray):
        raise TypeError("Cannot find root from a value of type {}".format(type(x)))
    steps = 0

    while np.linalg.norm(f(*x)) > tolerance and steps <= max_steps:
        if type(f(*x)) in fd._supported_scalars:
            x = x - (np.linalg.inv(_jacobian(f,x)) @ np.array([f(*x)]))
        else:
            x = x - (np.linalg.inv(_jacobian(f,x)) @ np.array(f(*x)))
        steps += 1
    if steps > max_steps:
        raise TimeoutError("Could not find root.")
    return x


if __name__ == '__main__':
    pass
