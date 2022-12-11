"""Implementation of reverse mode Automatic Differentiation.

Reverse mode automatic differentiation requires a data structure to store the
graph structure of a function's inputs and outputs. To this end, this package
implements a class GenFunc. An instance of a subclass of GenFunc contains the
mathematical definition of a function and its gradient together with references
to one or more input GenFunc objects. Complicated combinations of elementary
functions are instantiated as GenFunc objects through recursive instantiation of
the constituents.

The fundamental subclass of GenFunc is Input, which represents the individual
variables x0, x1, ..., xn of a function m by their associated projections
pi(x0, ... xn) = xi. Input instances do not store references to input functions.
When defining a multivariate function as a GenFunc, use input_vector(), which
returns a function object x such that x(0), x(1), ..., x(n) are Input objects
corresponding to the respective input variables.

Given a GenFunc instance y and an input point (x0, ..., xn), we can evaluate y
by y(x0, ..., xn) and compute its gradient by y.grad(x0, ..., xn).

Example
-------

We can model, evaluate, and differentiate the function
y(x0, x1) = sin(x0) + cos(x0 / x1) as follows:

    >>> from automaticDifferentiationTeam08.reversemode import *
    >>> x = input_vector(2)
    >>> y = Sin(x(0)) + Cos(x(0) / x(1))
    >>> output = y(3,-2)
    >>> gradient = y.grad(3,-2)
"""

import numpy as np

class GenFunc():
    """The base generalized function class. Do not instantiate this class
    directly; subclass it and instantiate the subclass.

    A subclass of GenFunc represents a single elementary function, while an
    instance of the subclass represents a potentially complicated function whose
    final operation is the elementary function represented by the subclass.
    For example, the Sin subclass represents the sine function, while an
    instance of Sin could represent a function such as f(x0,x1) = sin(x0 + x1).
    We distinguish the number of inputs of a GenFunc subclass instance from the
    number of *shallow* inputs. In the example above, the Sin object has two
    inputs but one shallow input (The sine function itself takes one argument).
    The number of shallow inputs is a class attribute, while the number of
    inputs is an instance attribute that can vary between instances.

    Attributes
    ----------
    num_shallow_inputs : int
        The number of shallow inputs to the function. Must be implemented in all
        subclasses, except for Input and Const.
    inputs : tuple[GenFunc]
        The GenFunc instances whose outputs are inputs to the final function of
        self. len(inputs) must equal num_shallow_inputs.
    num_inputs : int
        The number of inputs to the GenFunc instance.
    indexed_children : list[tuple[int, GenFunc]]
        A list of tuples (index, child_fn), where child_fn is a function whose
        inputs attribute contains self and index is the index of self in
        child_fn.inputs.
    stored_val : float
        Stored output used during the forward pass of the differentiation
        algorithm.
    stored_grad : list[float]
        Stored gradient used during the forward pass of the differentiation
        algorithm.
    stored_adjoint : float
        Stored adjoint (partial of the full function with respect to self) used
        in the reverse pass of the differentiation algorithm.

    Notes
    -----
    When subclassing GenFunc to define a custom function, define
    num_shallow_inputs, _evaluate(), and _grad(). Make sure that _grad() returns
    a list, even if the function has only one shallow output.
    """

    num_shallow_inputs = None

    def __init__(self, *args):
        """Instantiate by passing input GenFunc instances.

        Parameters
        ----------
        *args : GenFunc
            The inputs to self. Order matters, the first parameter to __init__
            must be the first input to the function, etc.

        Raises
        ------
        AttributeError
            If the input GenFunc objects do not all have the same number of
            inputs.
        IndexError
            If too many or too few GenFunc objects are given.
        """

        if len(args) != self.num_shallow_inputs:
            raise IndexError(
                f"{type(self).__name__} takes {self.num_shallow_inputs} "
                f"arguments; {len(args)} arguments given."
            )

        self.inputs = args

        num_input_set = set([input.num_inputs for input in self.inputs])
        if len(num_input_set) > 1:
            raise AttributeError("Functions must take same number of inputs")

        self.num_inputs = self.inputs[0].num_inputs
        self.indexed_children = []
        self.stored_val = None
        self.stored_grad = None
        self.stored_adjoint = None
        for index, input in enumerate(self.inputs):
            input._add_child(index, self)

    def __call__(self, *values):
        """Evaluate the instance as a mathematical function.

        Parameters
        ----------
        *values : int|float
            The values to be inputed to the function, in order.

        Returns
        -------
        int|float
            The output to the function.
        """
  
        if len(values) != self.num_inputs:
            raise IndexError(
                f"{len(values)} arguments were given,"
                f" expected {self.num_inputs}"
            )

        return self._evaluate(*[input(*values) for input in self.inputs])

    def _evaluate(self, *shallow_inputs):
        """Custom implementation of the mathematical definition of the subclass.

        Parameters
        ----------
        *shallow_inputs : int|float
            The inputs to the *final* function.

        Returns
        -------
        float
            The output to the function.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """

        raise NotImplementedError

    def _grad(self, *shallow_inputs):
        """Custom implementation of the gradient function of the subclass.

        Parameters
        ----------
        *shallow_inputs : int|float
            The inputs to the *final* function at which the gradient is
            computed.

        Returns
        -------
        list[float]
            The gradient of the function at the specified point.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """

        raise NotImplementedError

    def _add_child(self, index, child_fn):
        """Adds a child function; for accounting in the differentiation
        algorithm.

        Parameters
        ----------
        index : int
            The index of the input of child_function into which self is passed.
        child_fn : GenFunc
            The child function.
        """

        self.indexed_children.append((index, child_fn))

    def _forward_pass(self, *values):
        """The forward pass of the automatic differentiation algorithm.

        Starting from the inputs to a complicated function, we compute each
        successive subfunction from start to finish, storing the outputed values
        and gradients (with respect to the previous outputs) in the
        corresponding GenFunc objects along the way.

        Parameters
        ----------
        *values : int|float
            The input values to the full function.
        """

        if type(self) in (Input, Const):
            self.stored_val = self(*values)
            self.stored_grad = self._grad()
        else:
            shallow_inputs = []
            for input in self.inputs:
                input._forward_pass(*values)
                shallow_inputs.append(input.stored_val)
            self.stored_val = self._evaluate(*shallow_inputs)
            self.stored_grad = self._grad(*shallow_inputs)


    def _reverse_pass(self):
        """The reverse pass of the automatic differentiation algorithm.

        Starting from the final output, we traverse the graph of the function
        in reverse, computing adjoints of each sub-output using the chain rule.

        Returns
        -------
        list[tuple[int, int|float]]
            Encodes the gradient of self with respect to the inputs, to be
            parsed by grad()
        """

        if len(self.indexed_children) == 0:
            self.stored_adjoint = 1
        else:
            # Check if all children have been handled yet, and abort if not.
            children_adjoints = ([
                child.stored_adjoint
                for index, child in self.indexed_children]
            )
            if None in children_adjoints:
                return []

            self.stored_adjoint = sum([
                    child.stored_adjoint * child.stored_grad[index]
                    for index, child in self.indexed_children
                ]
            )

        if type(self) == Input:
            return [(self.i, self.stored_adjoint)]
        elif type(self) == Const:
            return []
        else:
            adjoints = []
            for input in self.inputs:
                adjoints += input._reverse_pass()
            return adjoints

    def _cleanup(self):
        """Removes all values stored during the differentiation algotihm."""

        self.stored_val = None
        self.stored_grad = None
        self.stored_adjoint = None

        if type(self) not in (Input, Const):
            for input in self.inputs:
                input._cleanup()

    def grad(self, *values):
        """Computes the gradient of the function.

        Parameters
        ----------
        *values : int|float
            The input values (not shallow) at which to compute the gradient.

        Returns
        -------
        list[float]
            The gradient vector.

        Raises
        ------
        IndexError
            If too many or too few input values are given.
        """

        if len(values) != self.num_inputs:
            raise IndexError(
                f"{len(values)} arguments were given,"
                f" expected {self.num_inputs}"
            )

        self._forward_pass(*values)
        grad_list = self._reverse_pass()
        grad = [0] * self.num_inputs
        for i, num in grad_list:
            grad[i] += num
        grad = [float(x) for x in grad]
        return grad

    def __add__(self, other):
        """Returns Add(self, other) if other is a GenFunc; if other is an int or
        float, first instantiates a Const object from other.
        """

        if isinstance(other, GenFunc):
            return Add(self, other)
        elif isinstance(other, (int, float)):
            return Add(self, Const(self.num_inputs, other))
        else:
            raise TypeError(f"Cannot add function to {type(other).__name__}")

    def __radd__(self, other):
        """Returns Add(self, other) if other is a GenFunc; if other is an int or
        float, first instantiates a Const object from other."""

        return self + other

    def __sub__(self, other):
        """Returns Subtract(self, other) if other is a GenFunc; if other is an
        int or float, first instantiates a Const object from other."""

        if isinstance(other, GenFunc):
            return Subtract(self, other)
        elif isinstance(other, (int, float)):
            return Subtract(self, Const(self.num_inputs, other))
        else:
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from function."
            )

    def __rsub__(self, other):
        """Returns Subtract(other, self) if other is a GenFunc; if other is an
        int or float, first instantiates a Const object from other."""

        if isinstance(other, GenFunc):
            return Subtract(other, self)
        elif isinstance(other, (int, float)):
            return Subtract(Const(self.num_inputs, other), self)
        else:
            raise TypeError(
                f"Cannot subtract function from {type(other).__name__}"
            )

    def __mul__(self, other):
        """Returns Multiply(self, other) if other is a GenFunc; if other is an
        int or float, first instantiates a Const object from other."""

        if isinstance(other, GenFunc):
            return Multiply(self, other)
        elif isinstance(other, (int, float)):
            return Multiply(self, Const(self.num_inputs, other))
        else:
            raise TypeError(
                f"Cannot multiply function to {type(other).__name__}"
            )

    def __rmul__(self, other):
        """Returns Multiply(self, other) if other is a GenFunc; if other is an
        int or float, first instantiates a Const object from other."""

        return self * other

    def __truediv__(self, other):
        """Returns Divide(self, other) if other is a GenFunc; if other is an
        int or float, first instantiates a Const object from other."""

        if isinstance(other, GenFunc):
            return Divide(self, other)
        elif isinstance(other, (int, float)):
            return Divide(self, Const(self.num_inputs, other))
        else:
            raise TypeError(f"Cannot divide function by {type(other).__name__}")

    def __rtruediv__(self, other):
        """Returns Divide(other, self) if other is a GenFunc; if other is an
        int or float, first instantiates a Const object from other."""

        if isinstance(other, GenFunc):
            return Divide(other, self)
        elif isinstance(other, (int, float)):
            return Divide(Const(self.num_inputs, other), self)
        else:
            raise TypeError(
                f"Cannot divide function from {type(other).__name__}"
            )

    def __pow__(self, other):
        """Returns Power(self, other) if other is an int or float.

        Notes
        -----
        The ** operator does not support the case where both the base and
        exponent are both instances of GenFunc. To define a function such as
        y = x0 ** x1, use

            >>> import numpy as np
            >>> x = input_vector(2)
            >>> y = np.e ** (x(1) * Log(x(0)))
        """

        if isinstance(other, (int, float)):
            return Power(other, self)
        else:
            raise TypeError(f"Cannot raise function to {type(other).__name__}")

    def __rpow__(self, other):
        """Returns Exp(self, other) if other is an int or float.

        Notes
        -----
        The ** operator does not support the case where both the base and
        exponent are both instances of GenFunc. To define a function such as
        y = x0 ** x1, use

            >>> import numpy as np
            >>> x = input_vector(2)
            >>> y = np.e ** (x(1) * Log(x(0)))
        """

        if isinstance(other, (int, float)):
            return Exp(other, self)
        else:
            raise TypeError(
                f"Cannot exponentiate function by base {type(other).__name__}"
            )

    def __neg__(self):
        """Returns -1 * self."""

        return -1 * self


class Input(GenFunc):
    """Represents the input of a multivariate function. An instance of input
    corresponds to a projection pi from the input vector (x0, ..., xn) to one of
    its component variables xi. For example, Input(4, 2) represents the 2nd
    input variable from a vector of four inputs. Do not instantiate this class
    directly, rather use input_vector().

    Unique Attributes
    -----------------
    i : int
        The index of the input variable.
    """

    def __init__(self, n, i):
        """Initializes an input variable.

        Parameters
        ----------
        n : int
            The number of input variables.
        i : int
            The index of the input variable corresponding to self.
        """

        self.num_inputs = n
        self.i = i
        self.indexed_children = []
        self.stored_val = None
        self.stored_grad = None
        self.stored_adjoint = None

    def __call__(self, *values):

        if len(values) != self.num_inputs:
            raise IndexError(
                f"{len(values)} arguments were given,"
                f" expected {self.num_inputs}"
            )

        return values[self.i]

    def _grad(self):
        grad = [0] * self.num_inputs
        grad[self.i] = 1
        return grad

class Const(GenFunc):
    """Represents a constant function of the input variables. Do not instantiate
    this class directly, rather use the implementations of the operators +, -,
    *, /, and ** for instances of GenFunc.

    Unique Attributes
    -----------------
    const : int|float
        The constant value of the function.
    """

    def __init__(self, n, c):
        """Initializes a Const instance.

        Parameters
        ----------
        n : int
            The number of input variables.
        c : int
            The constant value of the function.
        """

        self.num_inputs = n
        self.const = c
        self.indexed_children = []
        self.stored_val = None
        self.stored_grad = None
        self.stored_adjoint = None

    def __call__(self, *values):

        if len(values) != self.num_inputs:
            raise IndexError(
                f"{len(values)} arguments were given,"
                f" expected {self.num_inputs}"
            )

        return self.const

    def _grad(self):
        grad = [0] * self.num_inputs
        return grad


class Add(GenFunc):
    """Implements the addition function of two variables."""

    num_shallow_inputs = 2

    def _evaluate(self, *shallow_inputs):
        return shallow_inputs[0] + shallow_inputs[1]

    def _grad(self, *shallow_inputs):
        grad = [1,1]
        return grad

class Multiply(GenFunc):
    """Implements the multiplication function of two variables."""

    num_shallow_inputs = 2

    def _evaluate(self, *shallow_inputs):
        return shallow_inputs[0] * shallow_inputs[1]

    def _grad(self, *shallow_inputs):
        grad = [shallow_inputs[1], shallow_inputs[0]]
        return grad

class Subtract(GenFunc):
    """Implements the subtraction function of two variables."""

    num_shallow_inputs = 2

    def _evaluate(self, *shallow_inputs):
        return shallow_inputs[0] - shallow_inputs[1]

    def _grad(self, *shallow_inputs):
        grad = [1,-1]
        return grad

class Divide(GenFunc):
    """Implements the division function of two variables."""

    num_shallow_inputs = 2

    def _evaluate(self, *shallow_inputs):
        return shallow_inputs[0] / shallow_inputs[1]

    def _grad(self, *shallow_inputs):
        partial0 = 1 / shallow_inputs[1]
        partial1 = - shallow_inputs[0] / shallow_inputs[1]**2
        return [partial0, partial1]

class Power(GenFunc):
    """Implements the function that raises a variable to a specified power.

    Unique Attributes
    -----------------
    power : int|float
        the power to which the input value is raised.
    """

    num_shallow_inputs = 1

    def __init__(self, power, *args):
        super().__init__(*args)
        self.power = power

    def _evaluate(self, *shallow_inputs):
        return shallow_inputs[0] ** self.power

    def _grad(self, *shallow_inputs):
        if self.power == 0:
            return [0]
        else:
            return [self.power * shallow_inputs[0] ** (self.power - 1)]

class Exp(GenFunc):
    """Implements the function that raises a specified base to a variable.

    Unique Attributes
    -----------------
    base : int|float
        the base from which the input value is raised.
    """

    num_shallow_inputs = 1

    def __init__(self, base, *args):
        super().__init__(*args)
        if base < 0:
            raise AttributeError(
                "Cannot instantiate exponential function with nonpositive base"
            )
        self.base = base

    def _evaluate(self, *shallow_inputs):
        return self.base ** shallow_inputs[0]

    def _grad(self, *shallow_inputs):
        return [np.log(self.base) * self.base ** shallow_inputs[0]]

class Sin(GenFunc):
    """Implements the sine function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.sin(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [np.cos(shallow_inputs[0])]
        return grad

class Cos(GenFunc):
    """Implements the cosine function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.cos(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [-np.sin(shallow_inputs[0])]
        return grad

class Tan(GenFunc):
    """Implements the tangent function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.tan(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [1 / (np.cos(shallow_inputs[0])**2)]
        return grad

class Sec(GenFunc):
    """Implements the secant function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return 1 / np.cos(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [np.sin(shallow_inputs[0]) / np.cos(shallow_inputs[0])**2]
        return grad

class Csc(GenFunc):
    """Implements the cosecant function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return 1 / np.sin(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [-np.cos(shallow_inputs[0]) / np.sin(shallow_inputs[0])**2]
        return grad

class Cot(GenFunc):
    """Implements the cotangent function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return 1 / np.tan(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [-1 / (np.sin(shallow_inputs[0])**2)]
        return grad

class Log(GenFunc):
    """Implements the natural logarithm function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.log(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        if shallow_inputs[0] <= 0:
            raise ValueError(f"{shallow_inputs[0]} not in domain of log.")
        grad = [1 / shallow_inputs[0]]
        return grad

class ArcSin(GenFunc):
    """Implements the inverse sine function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.arcsin(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [1 / np.sqrt(1 - shallow_inputs[0]**2)]
        return grad

class ArcCos(GenFunc):
    """Implements the inverse cosine function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.arccos(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [-1 / np.sqrt(1 - shallow_inputs[0]**2)]
        return grad

class ArcTan(GenFunc):
    """Implements the inverse tangent function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.arctan(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [1 / (1 + shallow_inputs[0]**2)]
        return grad

class ArcSec(GenFunc):
    """Implements the inverse secant function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.arccos(1/shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        x = shallow_inputs[0]
        grad = [1 / (np.abs(x) * np.sqrt(x**2 - 1))]
        return grad

class ArcCsc(GenFunc):
    """Implements the inverse cosecant function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.arcsin(1/shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        x = shallow_inputs[0]
        grad = [-1 / (np.abs(x) * np.sqrt(x**2 - 1))]
        return grad

class ArcCot(GenFunc):
    """Implements the sine function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.arctan(1/shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        x = shallow_inputs[0]
        grad = [-1 / (1 + x**2)]
        return grad

class Abs(GenFunc):
    """Implements the absolute value function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.abs(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        x = shallow_inputs[0]
        if x > 0:
            return [1]
        elif x < 0:
            return [-1]
        elif x == 0:
            raise ValueError("Absolute value not differentiable at 0")

class Sinh(GenFunc):
    """Implements the hyperbolic sine function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.sinh(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [np.cosh(shallow_inputs[0])]
        return grad

class Cosh(GenFunc):
    """Implements the hyperbolic cosine function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.cosh(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [np.sinh(shallow_inputs[0])]
        return grad

class Tanh(GenFunc):
    """Implements the hyperbolic tangent function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return np.tanh(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [1 / np.cosh(shallow_inputs[0])**2]
        return grad

class Sech(GenFunc):
    """Implements the hyperbolic secant function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return 1/np.cosh(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [-np.sinh(shallow_inputs[0]) / np.cosh(shallow_inputs[0])**2]
        return grad

class Csch(GenFunc):
    """Implements the hyperbolic cosecant function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return 1/np.sinh(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [-np.cosh(shallow_inputs[0]) / np.sinh(shallow_inputs[0])**2]
        return grad

class Coth(GenFunc):
    """Implements the hyperbolic cotangent function."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return 1/np.tanh(shallow_inputs[0])

    def _grad(self, *shallow_inputs):
        grad = [-1 / np.sinh(shallow_inputs[0])**2]
        return grad

class Logistic(GenFunc):
    """Implements the logistic function f(x) = 1 / (1 + e^(-x))."""

    num_shallow_inputs = 1

    def _evaluate(self, *shallow_inputs):
        return 1 / (1 + np.exp(-shallow_inputs[0]))

    def _grad(self, *shallow_inputs):
        x = shallow_inputs[0]
        grad = [np.exp(-x) / (1 + np.exp(-x))**2]
        return grad

def input_vector(n):
    """Returns a factory function that generates Input objects for use in
    GenFunc instantiation when modelling functions.

    Parameters
    ----------
    n : int
        The number of input variables required for the function.

    Returns
    -------
    function
        The factory function.

    Examples
    --------
    If the function to be modelled takes 3 inputs x0, x1, and x2, write

        >>> x = input_vector(3)

    x(0) will be an instance of Input representing the first variable, x(1) the
    second, and x(2) the third. See module documentation for more details.
    """

    def input_generator(i):
        return Input(n,i)

    return input_generator

def jacobian(fn_list, *values):
    """Computes the Jacobian matrix of a function at a specified input point.

    Parameters
    ----------
    fn_list : list[GenFunc]
        The list of component functions to the function whose Jacobian is to be
        computed.
    *values : int|float
        The input values at which the Jacobian is computed.
    """

    return [fn.grad(*values) for fn in fn_list]


if __name__ == '__main__':
    pass
 
