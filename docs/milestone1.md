# Intro
Our package `automaticdifferentiation` solves the problem of differentiating real-valued functions by hand by automating the process and executing that automation with a compatible computer. This could help those who have the need to differentiate real-valued functions in large volumes save time.

# Background
**Auto differentiation** (AD) is the process of using software to compute the derivative of a given function. AD accomplishes this by employing a few key concepts:
**The Chain Rule:** in Calculus, the chain rule is used to calculate the derivative of composite functions. The chain rule for composite function h = f(g(x)) is h’ = f’(g(x)) * g’(x). In other words, the derivative of a composite function is the derivative of the outer function times the derivative of the inner function.
**Modes (accumulation):** refers to the “direction” in which the auto differentiator applies the chain rule. Forward accumulation approaches the function from the inside and works outward and is the mode upon which this package concentrates. A given function is expressed as a composition of elementary functions, which include arithmetic operations and standard transcendental functions such as sin, cos, and exp. The structure by which the functions feed into each other is summarized in a directed computational graph. In forward mode AD, derivatives of the elementary functions are computed from the beginning to the end of the graph, and combined using the chain rule.
**Dual numbers:** dual numbers are an augmentation of the algebra of real numbers. In the dual number paradigm, each number contains an additional component designating the derivative of a function at that number. Arithmetic operators are augmented as well to account for the implications of this. Arithmetic is performed on ordered pairs where regular arithmetic is applied to the first value of the pair and first order derivative arithmetic is applied to the second.

# How to Use
First, the package must be downloaded into the user’s python environment using  
```pip install --index-url https://test.pypi.org/simple/ --no-deps automaticdifferentiation```  

Then, importing the package will be as easy as writing the following in the same environment:  
```import automaticdifferentiation as ad```  

`automaticdifferentiation` contains a class `DualNumber`, instances of which are dual numbers which can be added, subtracted, multiplied, and divided. `automaticdifferentiation` also implements fundamental calculus functions such as $\sin$, $\cos$, $\exp$, and $\log$.

To compute the directional derivative of a function $f : \mathbb{R}^n \to \mathbb{R}$ at a point $x = (x_1,\ldots,x_n)$ in the direction $p = (p_1,\ldots,p_n)$, first define a python function `func(z1,...,zn)` that mimics the function $f$ but takes instances of `DualNumber` as inputs. Elementary operations such as addition, subtraction, multiplication, and division can be written as they would be for real numbers, but for other functions (such as $\sin$), use the functions provided by `automaticdifferentiation` (such as `ad.sin()`). To compute the directional derivative, instiatiate `DualNumber` objects `z1`,...,`zn` by setting the real part of `z1` equal to $x_1$ and the dual part of `z1` equal to $p_1$, and similarly for the other inputs. When these objects are passed into the function, the dual part of the output will be the directional derivative. Here is an example, where we compute a directional derivative of the function $f(x_1,x_2) = \sin(x_1 + x_2)$:

```
import automaticdifferentiation as ad

def func(z1, z2):
  return ad.sin(z1 + z2)

# Set point and direction of differentiation.
x = (1, 2)
p = (2, -1)

# Instantiate DualNumber objects to serve as inputs to func.
z = (ad.DualNumber(x[i], p[i]) for i in range(2))

# Pass into function. Directional derivative
# will be the dual part of the output.
output = func(*z)
derivative = output.dual

```


# Software Organization

The basic file structure is organized as follows:
package/   
├── LICENSE   
├── pyproject.toml   
├── README.md   
├── src/   
│   └── PACKAGE_NAME/   
│       ├── __init__.py   m
│       └── example.py   
└── tests/   

Where the package directory contains the entire contents of our package. LICENSE contains specifications for copyright permissions. pyproject.toml contains metadata and dependencies to allow pip to distribute and build our package on other devices. The package will be uploaded to pypi.org using twine. Then, the package will be distributed using a pip install command in the terminal to pull our package from pypi.org. README.md contains directions for usage and other relevant information. src/PACKAGE_NAME/ contains the functions for running Auto Differentiation and all the python code files that create it. Finally, the tests/ directory will act as the test suite for our project, containing test cases to ensure that the package functions as expected when downloaded.

The only **dependency** for this package is NumPy for mathematical and matrix functionality.


# Implementation

`automaticdifferentiation` is a single Python module containing the following classes and functions:

- `class automaticdifferentiation.DualNumber(real, dual)`

  Implementation of a dual number that can be added, multiplied, divided, exponentiated, or operated upon by standard calculus functions.

  - Parameters:
    - `real`: the real part of the dual number.
    - `dual`: the dual part of the dual number.
  - Attributes:
    - `real`: the real part of the dual number.
    - `dual`: the dual part of the dual number.
  - Methods:
    - `__add__(self, other)`: Returns `self + other`. `other` can be of type `DualNumber`, `int`, or `float`.
    - `__iadd__(self, other)`: Returns `self += other`. `other` can be of type `DualNumber`, `int`, or `float`.
    - `__radd__(self, other)`: Returns `other + self`. `other` can be of type `DualNumber`, `int`, or `float`.
    - `__mult__(self, other)`: Returns `self * other`. `other` can be of type `DualNumber`, `int`, or `float`.
    - `__imul(self, other)`: Returns `self *= other`. `other` can be of type `DualNumber`, `int`, or `float`.
    - `__rmul__(self, other)`: Returns `other * self`. `other` can be of type `DualNumber`, `int`, or `float`.
    - `__pow__(self, other)`: Returns `self ** other`. `other` can be of type `DualNumber`, `int`, or `float`.
    - `__rpow__(self, other)`: Returns `other ** self`. `other` can be of type `DualNumber`, `int`, or `float`.

The following are implementations of basic calculus functions that are compatible with dual numbers. For each of these functions $f$ and each dual number $z = a + p\epsilon$, $f(z)$ will equal $f(a) + pf'(a)\epsilon$.

- `automaticdifferentiation.sin(x)`

  Implementation of the sine function.

  - Parameters:
    - `x`: a number of type `DualNumber`, `int`, or `float`.
  - Returns:
    - `y`: the output to the function, of type `DualNumber`.

- `automaticdifferentiation.cos(x)`

  Implementation of the cosine function.

  - Parameters:
    - `x`: a number of type `DualNumber`, `int`, or `float`.
  - Returns:
    - `y`: the output to the function, of type `DualNumber`.

- `automaticdifferentiation.tan(x)`

  Implementation of the tangent function.

  - Parameters:
    - `x`: a number of type `DualNumber`, `int`, or `float`.
  - Returns:
    - `y`: the output to the function, of type `DualNumber`.

- `automaticdifferentiation.exp(x)`

  Implementation of the exponetial function.

  - Parameters:
    - `x`: a number of type `DualNumber`, `int`, or `float`.
  - Returns:
    - `y`: the output to the function, of type `DualNumber`.

- `automaticdifferentiation.log(x)`

  Implementation of the natural logarithm function.

  - Parameters:
    - `x`: a number of type `DualNumber`, `int`, or `float`.
  - Returns:
    - `y`: the output to the function, of type `DualNumber`.







# Licensing
We will use the copyright, open source MIT License for this project. The motivation for this choice falls on it being a strong general public license for most software (which our AD will be) and it allowing others to easily use our product in their own software - the consumer needs only to cite us, the license holders. A last, but not-so-insignificant addition to our reasoning comes in this license being very readable for first time copyright licensing readers.

## Feedback
- Added feedback section
- Reconciled different package names - all now say - - `automaticdifferentiation`
- Added a sentence regarding distribution in software organization
- Changed references to test.pypi.org to just the real pypi.org
- Added specific reference to automaticdifferentiation package in intro
- Added implementation details: classes and functions defined in the module.
