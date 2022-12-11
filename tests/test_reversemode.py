import numpy as np
import pytest
import sys
sys.path.append('src/automaticdifferentiationTeam08')

from reversemode import *

x = input_vector(3)
x1 = x(2)
gf1 = Sin(x(0) + x(1)) + np.e ** (x(2) * x(1)) - 4

class TestFunctions:
    """Test Reverse Mode AD Module"""

    def test_init(self):
        """Test GenFunc init method (RM Module)"""
        # Create input generator object
        x = input_vector(3)

        # Use generator to create Input object with i = 1
        x1 = x(2)
        
        # Test instantiation of Input object
        assert x1.__dict__ == {'num_inputs': 3, 'i': 2, 'indexed_children': [], 'stored_val': None, 'stored_grad': None, 'stored_adjoint': None}

        # Create Subtract GenFunc object
        gf1 = Sin(x(0) + x(1)) + np.e ** (x(2) * x(1)) - 4

        # Reject empty input
        try: y = GenFunc()
        except IndexError: assert True

        # Create a const object
        y = Const(None, None)

        # Reject calls with no input
        try: y()
        except IndexError: assert True

        # Reject evaluate with no input
        try: y._evaluate()
        except NotImplementedError: assert True

        # Reject different input lengths
        a = input_vector(1)
        c = input_vector(90)

        try: Sin(a(1) + c(12)) 
        except AttributeError: assert True
          

    def test_add(self):
        """Test GenFunc add method (RM)"""

        # Value of y when x0 = 1, x1=2, x2=3
        assert np.isclose(gf1(1,2,3), 399.5699135007949)

        # Test results of calculation
        r = gf1.grad(1,2,3)
        assert np.isclose(r[0], -0.9899924966004454)
        assert np.isclose(r[1], 1209.2963879816045)
        assert np.isclose(r[2], 806.85758698547)

        # Addn w 2 GF
        a = input_vector(2)
        b = a(1) + a(1)
        assert b(1,4) == 8

        # Addn w GF / int
        c = a(1) + 3
        assert c(1,7) == 10

        # Addn w int / GF
        d = 3 + a(1)
        assert d(1,2) == 5

        # Reject invalid addn
        try: b + "hi"
        except TypeError: assert True

        # Reject calls without enough args
        try: b(1)
        except IndexError: assert True


    def test_sub(self):
        """Test GenFunc sub method (RM)"""
        a = input_vector(2)
        b = input_vector(2)

        # GF w GF
        c = a(1) - b(1)
        assert c(4,7) == 0

        # GF with Const
        d = a(1) - 4
        assert d(1, 10) == 6

        # Const with GF
        e = 4 - a(1)
        assert e(1,1) == 3

        # GF with invalid
        try: a(1) - "hi"
        except TypeError: assert True

        # Rsub assertion should return Subtract object
        assert isinstance(b(1).__rsub__(a(1)), Subtract)

        # Rsub w invalid
        try: b(1).__rsub__("hi")
        except TypeError: assert True


    def test_rmul(self):
        """Test GenFunc rmul method (RM)"""
        a = input_vector(2)
        b = input_vector(2)

        # GF w invalid
        try: b(1) * "hi"
        except TypeError: assert True

        # Rmul returns Multiply object
        assert isinstance(b(1).__rmul__(4), Multiply)


    def test_truediv(self):
        """Test GenFunc truediv method and Divide class methods (RM)"""
        a = input_vector(2)
        b = input_vector(2)

        # GF div GF
        c = a(1) / b(1)
        assert isinstance(c, Divide)
        assert c(12,18) == 1.0

        # GF div Const
        d = a(1) /2
        assert d(1, 10) == 5.0

        # Const div GF
        e = 4 / a(1)
        assert e(1,2) == 2.0

        # GF div invalid
        try: a(1) / "hi"
        except TypeError: assert True

        # Rdiv returns Divide object
        assert isinstance(b(1).__rtruediv__(a(1)), Divide)

        # Rdiv rejects invalid
        try: b(1).__rtruediv__("hi")
        except TypeError: assert True

        # Divide class _grad
        assert c._grad(12, 2) == [0.5, -3.0]

    
    def test_pow(self):
        """Test GenFunc pow method and Power class methods"""
        a = input_vector(2)
        b = input_vector(2)

        # GF pow 2
        c = a(1) ** 2
        assert isinstance(c, Power)
        assert c(2,5)== 25

        # GF pow 0
        d = a(1) ** 0
        assert d._grad(2) == [0]

        # Rpow returns Exp object
        assert isinstance(b(1).__rpow__(2), Exp)

        # GF pow invalid
        try: a(1) ** "hi"
        except TypeError: assert True

        # GF rpow invalid
        try: a(1).__rpow__(b(2))
        except TypeError: assert True

        # Pow class _grad
        assert c._grad(2) == [4]
        assert c._grad(0) == [0]


    def test_Exp(self):
        """Test Exp class methods"""
        a = input_vector(2)
        
        # Reject Negative bases
        try: a(1).__rpow__(-2)
        except AttributeError: assert True

        # Test calculations
        b = a(1).__rpow__(2)
        assert b._evaluate(4) == 16
        assert np.isclose(b._grad(5), 22.18070977791825)


    def test_neg(self):
        """Test neg function (RM)"""
        a = input_vector(2)
        assert isinstance(-a(1), Multiply)


    def test_fwd(self):
        """Test GenFunc fwd mode method (RM)"""
        a = input_vector(2)
        d = 3 + a(1)
        d._forward_pass(1,4)

        # Test value storage
        assert d.stored_val == 7
        assert d.stored_grad == [1,1]
        assert d.grad(3,2) == [0.0, 1.0]

        # Test value cleanup
        d._cleanup()
        assert d.stored_val == None
       
       # Grad index error
        try: d.grad(3)
        except IndexError: assert True


    def test_reverse(self):
        """Test GenFunc reverse mode method (RM)"""
        a = input_vector(2)
        d = 3 + a(1)
        d._forward_pass(1,4)
        assert d._reverse_pass() == [(1,1)]


    def test_Input(self):
        """Test methods of Input class (RM)"""
        a = input_vector(2)

        # Input generator returns Input objects
        assert isinstance(a(1), Input)
        
        # Reject incorrect input length
        try: a(1)(1,2,3)
        except IndexError: assert True

    def test_Cos(self):
        """Test Cos class (RM)"""
        a = input_vector(1)
        b = Cos(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 0.540302305)
        assert np.isclose(b._grad(1), -0.8414709848078965)

    
    def test_Tan(self):
        """Test Tan class (RM)"""
        a = input_vector(1)
        b = Tan(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 1.5574077246549023)
        assert np.isclose(b._grad(1), 3.425518820814759)


    def test_Sec(self):
        """Test Sec class (RM)"""
        a = input_vector(1)
        b = Sec(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 1.850815717680925)
        assert np.isclose(b._grad(1), 2.8824746956289795)

    
    def test_Csc(self):
        """Test Csc class (RM)"""
        a = input_vector(1)
        b = Csc(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 1.1883951057781212)
        assert np.isclose(b._grad(1), -0.7630597222326296)

    def test_Cot(self):
        """Test Cot class (RM)"""
        a = input_vector(1)
        b = Cot(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 0.6420926159343306)
        assert np.isclose(b._grad(1), -1.412282927437392)

    
    def test_Log(self):
        """Test Log class (RM)"""
        a = input_vector(1)
        b = Log(a(1))

        assert b._evaluate(1) ==  0.0
        assert np.isclose(b._evaluate(2), 0.6931471805599453)
        assert b._grad(2) == [0.5]

        # Test calculations
        try: assert b._grad(0) == 0.0
        except ValueError: assert True


    def test_ArcSin(self):
        """Test ArcSin class (RM)"""
        a = input_vector(1)
        b = ArcSin(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(.5), 0.5235987755982989)
        assert np.isclose(b._grad(.5), 1.1547005383792517)

    
    def test_ArcCos(self):
        """Test ArcCos class (RM)"""
        a = input_vector(1)
        b = ArcCos(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(.5), 1.0471975511965979)
        assert np.isclose(b._grad(.5), -1.1547005383792517)

    
    def test_ArcTan(self):
        """Test ArcTan class (RM)"""
        a = input_vector(1)
        b = ArcTan(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(.5), 0.4636476090008061)
        assert b._grad(.5) == [0.8]

    
    def test_ArcSec(self):
        """Test ArcSec class (RM)"""
        a = input_vector(1)
        b = ArcSec(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1.5), 0.8410686705679303)
        assert np.isclose(b._grad(1.5), 0.5962847939999439)

    
    def test_ArcCsc(self):
        """Test ArcCsc class (RM)"""
        a = input_vector(1)
        b = ArcCsc(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1.5), 0.7297276562269663)
        assert np.isclose(b._grad(1.5), -0.5962847939999439)


    def test_ArcCot(self):
        """Test ArcCot class (RM)"""
        a = input_vector(1)
        b = ArcCot(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1.5), 0.5880026035475675)
        assert np.isclose(b._grad(1.5), -0.3076923076923077)


    def test_Abs(self):
        """Test Abs class (RM)"""
        a = input_vector(1)
        b = Abs(a(1))

        # Eval with +/- inputs
        assert b._evaluate(-3) == 3
        assert b._evaluate(3) == 3

        # Grad with +/- inputs
        assert b._grad(3) == [1]
        assert b._grad(-3) == [-1]

        # Reject grad at 0
        try: assert b._grad(0) == 0
        except ValueError: assert True

    
    def test_Sinh(self):
        """Test Sinh class (RM)"""
        a = input_vector(1)
        b = Sinh(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 1.1752011936438014)
        assert np.isclose(b._grad(1), 1.5430806348152437)

    
    def test_Cosh(self):
        """Test Cosh class (RM)"""
        a = input_vector(1)
        b = Cosh(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 1.5430806348152437)
        assert np.isclose(b._grad(1), 1.1752011936438014)

    
    def test_Tanh(self):
        """Test Tanh class (RM)"""
        a = input_vector(1)
        b = Tanh(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 0.7615941559557649)
        assert np.isclose(b._grad(1), 0.4199743416140261)

    
    def test_Sech(self):
        """Test Sech class (RM)"""
        a = input_vector(1)
        b = Sech(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 0.6480542736638855)
        assert np.isclose(b._grad(1), -0.49355434756457306)

    
    def test_Csch(self):
        """Test Csch class (RM)"""
        a = input_vector(1)
        b = Csch(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 0.8509181282393216)
        assert np.isclose(b._grad(1), -1.1172855274492743)

    
    def test_Coth(self):
        """Test Coth class (RM)"""
        a = input_vector(1)
        b = Coth(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 1.3130352854993315)
        assert np.isclose(b._grad(1), -0.7240616609663106)


    def test_Logistic(self):
        """Test Logistic class (RM)"""
        a = input_vector(1)
        b = Logistic(a(1))

        # Test calculations
        assert np.isclose(b._evaluate(1), 0.7310585786300049)
        assert np.isclose(b._grad(1), 0.19661193324148188)