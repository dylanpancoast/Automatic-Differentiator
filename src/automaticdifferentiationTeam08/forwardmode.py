"""This module uses dual numbers to implement the forward mode automatic
differentiation alogithm.
"""

import numpy as np

_supported_scalars = (int, float, np.float64, np.int64)

class DualNumber:
    """A class used to represent dual numbers.

    ...

    Attributes
    ----------
    real : int|float|numpy.float64
        the real part of the dual number
    dual : int|float|numpy.float64
        the dual part of the dual number
    """

    _supported_scalars = (int, float, np.float64, np.int64, )

    def __init__(self, real, dual):
        """
        Parameters
        ----------
        real : int|float|numpy.float64
            the real part of the dual number
        dual : int|float|numpy.float64
            the dual part of the dual number

        Raises
        ------
        TypeError
            If real or dual are not of a supported type.
        """
        if type(real) not in self._supported_scalars:
            raise TypeError("Real component must be a number")
        if type(dual) not in self._supported_scalars:
            raise TypeError("Dual component must be a number")

        self.real = real
        self.dual = dual

    def _inverse(self):

        x = self.real
        p = self.dual
        return DualNumber(1 / x, -p / x ** 2)

    def __add__(self, new):
        """Returns self + new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be added

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        if isinstance(new, DualNumber):
            return DualNumber(self.real + new.real, self.dual + new.dual)
        elif type(new) in self._supported_scalars:
            return DualNumber(self.real + new, self.dual)
        else:
            raise TypeError("Cannot add DualNumber to {}".format(type(new)))

    def __sub__(self, new):
        """Returns self - new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be added

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        if isinstance(new, DualNumber):
            return DualNumber(self.real - new.real, self.dual - new.dual)
        elif type(new) in self._supported_scalars:
            return DualNumber(self.real - new, self.dual)
        else:
            raise TypeError("Cannot add DualNumber to {}".format(type(new)))

    def __mul__(self, new):
        """Returns self * new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be multiplied

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        if isinstance(new, DualNumber):
            return DualNumber(self.real * new.real, self.real * new.dual +
                              self.dual * new.real)
        elif type(new) in self._supported_scalars:
            return DualNumber(self.real * new, self.dual * new)
        else:
            raise TypeError("Cannot multiply DualNumber by {}".format(type(new)))

    def __truediv__(self, new):
        """Returns self / new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be divided by.

        Raises
        ------
        TypeError
            If new is not of a supported type.
        ValueError
            If new == 0.
        """

        if new == 0:
            raise ValueError("Cannot divide by 0.")

        if isinstance(new, DualNumber):
            return self * new._inverse()
        elif type(new) in _supported_scalars:
            return self * (1 / new)
        else:
            raise TypeError("Cannot divide DualNumber by {}".format(type(new)))

    def __radd__(self, new):
        """Returns new + self.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be added

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        return self.__add__(new)

    def __rsub__(self, new):
        """Returns new - self.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be added

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        return self.__sub__(new)

    def __rmul__(self, new):
        """Returns new * self.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be multiplied

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        return self.__mul__(new)

    def __rtruediv__(self, new):
        """Returns new / self.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be divided from.

        Raises
        ------
        TypeError
            If new is not of a supported type.
        ValueError
            If new == 0.
        """

        return self.__truediv__(DualNumber(new, 0))

    def __eq__(self, new):
        """Returns self == new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be compared against.

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        if type(new) == DualNumber:
            if (self.real == new.real) & (self.dual == new.dual):
                return True
            return False
        elif type(new) in self._supported_scalars:
            if self.real == new:
                return True
            return False
        else:
            raise TypeError("Cannot compare DualNumber to {}".format(type(new)))


    def __ne__(self, new):
        """Returns self != new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be compared against.
        
        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        return not self.__eq__(new)

    def __iadd__(self, new):
        """Sets the real and dual parts of self to match self + new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be added

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        result = self + new
        self.real = result.real
        self.dual = result.dual
        return self

    def __isub__(self, new):
        """Sets the real and dual parts of self to match self + new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be added

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        result = self - new
        self.real = result.real
        self.dual = result.dual
        return self

    def __imul__(self, new):
        """Sets the real and dual parts of self to match self * new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be multiplied

        Raises
        ------
        TypeError
            If new is not of a supported type.
        """

        result = self * new
        self.real = result.real
        self.dual = result.dual
        return self

    def __itruediv__(self, new):
        """Sets the real and dual parts of self to match self / new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be divided by.

        Raises
        ------
        TypeError
            If new is not of a supported type.
        ValueError
            If new == 0.
        """

        result = self / new
        self.real = result.real
        self.dual = result.dual
        return self


    def __pow__(self, new):
        """Returns self ** new.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the number to be exponentiated by.

        Raises
        ------
        TypeError
            If new is not of a supported type.
        ValueError
            If self.real <= 0.

        Notes
        -----
        Do NOT use this method to compute inverses, i.e.

        >> num = DualNumber(real, dual)
        >> inverse = num ** (-1)

        Instead use:

        >> inverse = 1 / num
        """

        if isinstance(new, DualNumber):
            x1 = self.real
            p1 = self.dual
            x2 = new.real
            p2 = new.dual

            # Positive base
            if x1 > 0:
                real = x1 ** x2
                dual = p1 * x2 * (x1 ** (x2 - 1)) + p2 * np.log(x1) * (x1 ** x2)
                return DualNumber(real, dual)

            # Integer exponent with 0 dual part
            if x2 % 1 == 0 and x2 != 0 and p2 == 0:
                real = x1 ** x2
                dual = p1 * x2 * x1**(x2-1)
                return DualNumber(real,dual)

            # Negative base, reciprocal odd integer exponent with 0 dual part
            if 1/x2 % 2 == 1 and p2 == 0 and x1 < 0:
                return -1 * (-1 * self)**new

        elif type(new) in self._supported_scalars:
            return self ** DualNumber(new, 0)

        else:
            raise TypeError("Cannot raise DualNumber to power of {}".format(type(new)))

    def __rpow__(self, new):
        """Returns new ** self.

        Parameters
        ----------
        new : DualNumber|int|float|numpy.float64
            the base to be exponentiated.

        Raises
        ------
        TypeError
            If new is not of a supported type.
        ValueError
            If new.real <= 0.
        """

        return self.__pow__(DualNumber(new, 0))

    def __neg__(self):
        """Returns the additive inverse of self."""

        return -1 * self

    def __str__(self):
        return f"{self.real} + {self.dual}*epsilon"

    def __repr__(self):

        cls = type(self)
        return f"{cls.__name__}({self.real}, {self.dual})"


def sin(x):
    """Implements the sine function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ------
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.sin(x.real), np.cos(x.real) * x.dual)
    elif type(x) in _supported_scalars:
        return np.sin(x)
    else:
        raise TypeError("Cannot take sin of {}".format(type(x)))


def cos(x):
    """Implements the cosine function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ------
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.cos(x.real), -np.sin(x.real) * x.dual)
    elif type(x) in _supported_scalars:
        return np.cos(x)
    else:
        raise TypeError("Cannot take cos of {}".format(type(x)))

def tan(x):
    """Implements the tangent function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ------
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.tan(x.real), x.dual / np.cos(x.real) ** 2)
    elif type(x) in _supported_scalars:
        return np.tan(x)
    else:
        raise TypeError("Cannot take tan of {}".format(type(x)))


def sec(x):
    """Implements the secant function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ------
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(1 / np.cos(x.real), np.sin(x.real) / np.cos(x.real) ** 2 * x.dual)
    elif type(x) in _supported_scalars:
        return 1 / np.cos(x)
    else:
        raise TypeError("Cannot take sec of {}".format(type(x)))


def csc(x):
    """Implements the cosecant function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ------
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(1 / np.sin(x.real), -np.cos(x.real) / np.sin(x.real) ** 2 * x.dual)
    elif type(x) in _supported_scalars:
        return 1 / np.sin(x)
    else:
        raise TypeError("Cannot take csc of {}".format(type(x)))

def cot(x):
    """Implements the sine function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ------
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(1 / np.tan(x.real), -x.dual / np.sin(x.real) ** 2)
    elif type(x) in _supported_scalars:
        return 1 / np.tan(x)
    else:
        raise TypeError("Cannot take cot of {}".format(type(x)))

def arcsin(x):
    """Implements the arcsine function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.arcsin(x.real), x.dual / np.sqrt(1 - x.real ** 2))
    elif type(x) in _supported_scalars:
        return np.arcsin(x)
    else:
        raise TypeError("Cannot take arcsin of {}".format(type(x)))

def arccos(x):
    """Implements the arccosine function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ------
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.arccos(x.real), -x.dual / np.sqrt(1 - x.real ** 2))
    elif type(x) in _supported_scalars:
        return np.arccos(x)
    else:
        raise TypeError("Cannot take arccos of {}".format(type(x)))

def arctan(x):
    """Implements the arctangent function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.arctan(x.real), x.dual / (1 + x.real ** 2))
    elif type(x) in _supported_scalars:
        return np.arctan(x)
    else:
        raise TypeError("Cannot take arctan of {}".format(type(x)))

def sinh(x):
    """Implements the hyperbolic sine function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.sinh(x.real), np.cosh(x.real) * x.dual)
    elif type(x) in _supported_scalars:
        return np.sinh(x)
    else:
        raise TypeError("Cannot take sinh of {}".format(type(x)))

def cosh(x):
    """Implements the hyperbolic cosine function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.cosh(x.real), np.sinh(x.real) * x.dual)
    elif type(x) in _supported_scalars:
        return np.cosh(x)
    else:
        raise TypeError("Cannot take cosh of {}".format(type(x)))

def tanh(x):
    """Implements the hyperbolic tangent function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(np.tanh(x.real), x.dual / np.cosh(x.real) ** 2)
    elif type(x) in _supported_scalars:
        return np.tanh(x)
    else:
        raise TypeError("Cannot take tanh of {}".format(type(x)))

def coth(x):
    """Implements the hyperbolic cotangent function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(1 / np.tanh(x.real), -x.dual / np.cosh(x.real) ** 2)
    elif type(x) in _supported_scalars:
        return 1 / np.tanh(x)
    else:
        raise TypeError("Cannot take coth of {}".format(type(x)))

def csch(x):
    """Implements the hyperbolic cosecant function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(1 / np.sinh(x.real), -x.dual / np.sinh(x.real) / np.cosh(x.real))
    elif type(x) in _supported_scalars:
        return 1 / np.sinh(x)
    else:
        raise TypeError("Cannot take csch of {}".format(type(x)))
    
def sech(x):
    """Implements the hyperbolic secant function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(1 / np.cosh(x.real), -x.dual / np.sinh(x.real) / np.cosh(x.real))
    elif type(x) in _supported_scalars:
        return 1 / np.cosh(x)
    else:
        raise TypeError("Cannot take sech of {}".format(type(x)))


def exp(x, base = np.e):
    """Implements the exponential function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function
    base : DualNumber|int|float|numpy.float64
        The base of the exponential

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if base == np.e:
        if isinstance(x, DualNumber):
            return DualNumber(np.exp(x.real), np.exp(x.real) * x.dual)
        elif type(x) in _supported_scalars:
            return np.exp(x)
        else:
            raise TypeError("Cannot take exp of {}".format(type(x)))
    else:
        if isinstance(x, DualNumber):
            return DualNumber(base ** x.real, base ** x.real * np.log(base) * x.dual)
        elif type(x) in _supported_scalars:
            return base ** x
        else:
            raise TypeError("Cannot take exp of {}".format(type(x)))

def log(x, base = np.e):
    """Implements the natural logarithm function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ------
    TypeError
        If x is not of a supported type.
    """
    if base == np.e:
        if isinstance(x, DualNumber):
            if x.real <= 0:
                raise ValueError("Cannot take log of a negative number.")
            return DualNumber(np.log(x.real), x.dual / x.real)
        elif type(x) in _supported_scalars:
            return np.log(x)
        else:
            raise TypeError("Cannot take log of {}".format(type(x)))
    else:
        if isinstance(x, DualNumber):
            if x.real <= 0:
                raise ValueError("Cannot take log of a negative number.")
            return DualNumber(np.log(x.real) / np.log(base), x.dual / x.real / np.log(base))
        elif type(x) in _supported_scalars:
            return np.log(x) / np.log(base)
        else:
            raise TypeError("Cannot take log of {}".format(type(x)))

def sqrt(x):
    """Implements the square root function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        if x.real < 0:
            raise ValueError("Cannot take square root of a negative number.")
        return DualNumber(np.sqrt(x.real), x.dual / (2 * np.sqrt(x.real)))
    elif type(x) in _supported_scalars:
        return np.sqrt(x)
    else:
        raise TypeError("Cannot take sqrt of {}".format(type(x)))

def logistic(x):
    """Implements the logistic function for dual numbers.

    Parameters
    ----------
    x : DualNumber|int|float|numpy.float64
        The argument to the function

    Raises
    ______
    TypeError
        If x is not of a supported type.
    """

    if isinstance(x, DualNumber):
        return DualNumber(1 / (1 + np.exp(-x.real)), x.dual * np.exp(-x.real) / (1 + np.exp(-x.real)) ** 2)
    elif type(x) in _supported_scalars:
        return 1 / (1 + np.exp(-x))
    else:
        raise TypeError("Cannot take logistic of {}".format(type(x)))

def directional_derivative(fn, point, direction, normalize=True):
    """Computes the directional derivative of a function at a given point and
    direction.

    Parameters
    ----------
    fn : function
        The function whose derivative is to be evaluated. Arguments to fn must
        be of type DualNumber, and fn must return a single DualNumber.
    point : int|float|numpy.float64|numpy.array
        the point at which the derivative is evaulated.
    dir :  int|float|numpy.float64|numpy.array
        the direction vector with respect to which the derivative is evaluated.
    normalize: bool
        if True, normalizes point to a unit vector.

    Returns
    -------
    result.dual : int|float|numpy.float64
        the directional derivative.

    Raises
    ------
    ValueError
        If the lengths of point, dir, and the number of arguments to fn do not
        match.

    TypeError
        If point and direction type cannot be made into np.ndarrays of supported
        types
    """
    # Ensure right type of inputs are given
    # Ensure fn is of type function
    if not callable(fn):
        raise TypeError("Cannot find derivative of a {}".format(type(fn)))
    # Ensure point is of type array
    if type(point) in _supported_scalars:
        point = np.array([point])
    if not isinstance(point, np.ndarray):
        print(type(point))
        raise TypeError("Cannot evaluate derivative at a value of type {}".format(type(point)))
    # Ensure direction is of type array
    if type(direction) in _supported_scalars:
        direction = np.array([direction])
    if not isinstance(direction, np.ndarray):
        raise TypeError("Cannot evaluate derivative in a direction of type {}".format(type(direction)))

    # Ensure right size of inputs are given
    # Ensure that point and dir have the same number of elements
    if len(point) != len(direction):
        raise ValueError("Point and direction must have the same number of dimensions.")
    # Ensure point and dir have the same number of elements as fn has inputs
    if len(point) != fn.__code__.co_argcount:
        raise ValueError("Point and direction must have the same number of dimensions as the function has inputs.")

    # Normalize dir if requested
    if normalize:
        direction = direction / np.linalg.norm(direction)

    # Create a list of dual numbers, one for each input
    duals = []
    for i in range(len(point)):
        duals.append(DualNumber(point[i], direction[i]))

    # Evaluate the function at the dual numbers
    result = fn(*duals)

    # Return the dual part of the result as the directional derivative
    if type(result) == tuple:
        dual_list = []
        for num in result:
            if type(num) is not DualNumber:
                num = DualNumber(num, 0)
            dual_list.append(num.dual)
        return dual_list

    return result.dual


if __name__ == '__main__':
    pass
