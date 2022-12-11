import numpy as np
import pytest
import sys
sys.path.append('src/automaticdifferentiationTeam08')

import newton as nt

# Functions used for testing
def fn0(x):
    return x

def fn1(x,y):
    return y, x

def fn2(x,y):
    return x**2+y, y-x+1


class TestFunctions:
    """Test for Newton's Method"""

    def test_find_root(self):
        """Test Newton's find_root"""

        # Find root of single-variable input
        r0 = nt.find_root(fn0, 1)
        assert r0 == [0.]

        # Find root of multi-variable input
        r1 = nt.find_root(fn1, np.array([100,100]))
        assert np.all(r1 == [0., 0.])

        r2 = nt.find_root(fn2, np.array([100,100]))
        assert np.all(np.isclose(r2, [0.61803399, -0.38196601]))

        # Reject unreasonable constraints on application of Newton's method
        try: nt.find_root(fn2, np.array([100,100]), tolerance=1e-1, max_steps=1)
        except TimeoutError: assert True
    
        # Reject unsupported types for x input
        try: nt.find_root(fn0, '1')
        except TypeError: assert True
