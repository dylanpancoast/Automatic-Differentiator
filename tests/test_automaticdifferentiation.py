import numpy as np
import pytest
import sys
sys.path.append('src/automaticdifferentiationTeam08')

from forwardmode import DualNumber
import forwardmode as fad

# For functions which play nice with zero and for checking division by zero
n0 = DualNumber(0, 0)

# For general-use testing (+, -, *, /, **, etc.)
n1 = DualNumber(107, 1)
n2 = DualNumber(207, 2)
n9 = DualNumber(10, 1)

# For functions that play nice with the natural number
n107 = DualNumber(np.e, 1)
n207 = DualNumber(np.e, 2)

# For trig functions
t1 = DualNumber(np.pi, np.pi / 2)
t2 = DualNumber(0, np.pi)
t3 = DualNumber(np.pi / 2, 0)
t4 = DualNumber(-np.pi / 2, np.pi)
t5 = DualNumber(1, np.pi / 4)
t6 = DualNumber(0.5, np.pi / 3)
t7 = DualNumber(-2, np.pi)
t8 = DualNumber(2, np.pi)

# For hyperbolic functions
ee = np.e ** np.e
ene = np.e ** -np.e


class TestFunctions:
    """Test for Forward Mode AD"""

    def test_init(self):
        n1 = DualNumber(107, 1)
        n2 = DualNumber(207, 2)
        assert n1.dual == 1 and n1.real == 107
        assert n2.dual == 2 and n2.real == 207

        # Reject unsupported types
        try:
            DualNumber('1', 1)
        except TypeError:
            assert True
        else:
            assert False

        # Reject unsupported types
        try:
            DualNumber(1, '1')
        except TypeError:
            assert True
        else:
            assert False

    def test_add(self):
        """Test DualNum add"""

        # Test two different dual nums
        n3 = n1 + n2
        assert n3.real == 314
        assert n3.dual == 3

        # Test two of the same dual nums
        n4 = n1 + n1
        assert n4.real == 214
        assert n4.dual == 2

        # Addn w float
        n5 = n1 + 1.5
        assert n5.real == 108.5
        assert n5.dual == 1

        # Addn w int
        n6 = n1 + 10
        assert n6.real == 117
        assert n6.dual == 1

        # Reject unsupported types
        try:
            n6 + '1'
        except TypeError:
            assert True
        else:
            assert False

    def test_sub(self):
        """Test DualNum sub"""

        # Test two different dual nums
        n3 = n1 - n2
        assert n3.real == -100
        assert n3.dual == -1

        # Test two of the same dual nums
        n4 = n1 - n1
        assert n4.real == 0
        assert n4.dual == 0

        # Subt w float
        n5 = n1 - 1.5
        assert n5.real == 105.5
        assert n5.dual == 1

        # Subt w int
        n6 = n1 - 10
        assert n6.real == 97
        assert n6.dual == 1

        # Reject unsupported types
        try:
            n6 - '1'
        except TypeError:
            assert True
        else:
            assert False

    def test_mul(self):
        """Test DualNum mul"""

        # Two of the same dual nums
        n3 = n1 * n1
        assert n3.real == 11449
        assert n3.dual == 214

        # Two different complex nums
        n4 =  n1 * n2
        assert n4.real == 22149
        assert n4.dual == 421

        # Mult w float
        n5 = n1 * 1.5
        assert n5.real == 160.5
        assert n5.dual == 1.5

        # Mult w int
        n6 = n1 * 2
        assert n6.real == 214
        assert n6.dual == 2

        # Reject unsupported types
        try:
            n6 * '1'
        except TypeError:
            assert True
        else:
            assert False

    def test_truediv(self):
        """Test DualNum truediv"""

        # Two of the same dual nums
        n3 = n1 / n1
        assert np.isclose(n3.real, 1)
        assert np.isclose(n3.dual, 0)

        # Two different complex nums
        n4 =  n207 / n107
        assert np.isclose(n4.real, 1)
        assert np.isclose(n4.dual, 1 / np.e)

        # Div w float
        n5 = n207 / 1.5
        assert np.isclose(n5.real, 2 * np.e / 3)
        assert np.isclose(n5.dual, 4 / 3)

        # Div w int
        n6 = n1 / 2
        assert np.isclose(n6.real, 53.5)
        assert np.isclose(n6.dual, 0.5)

        # Reject unsupported types
        try:
            n6 / '1'
        except TypeError:
            assert True
        else:
            assert False

        # Reject division by 0
        try:
            n1 / 0 and n1 / n0
        except ValueError:
            assert True
        else:
            assert False

    def test_radd(self):
        """Test DualNum radd"""

        # Addn w float
        n5 = 1.5 + n1
        assert n5.real == 108.5
        assert n5.dual == 1

        # Addn w int
        n6 = 10 + n1
        assert n6.real == 117
        assert n6.dual == 1

        # Reject unsupported types
        try:
            '1' + n6
        except TypeError:
            assert True
        else:
            assert False

    def test_rsub(self):
        """Test DualNum rsub"""

        # Subt w float
        n5 = 1.5 - n1
        assert n5.real == 105.5
        assert n5.dual == 1

        # Subt w int
        n6 = 10 - n1
        assert n6.real == 97
        assert n6.dual == 1

        # Reject unsupported types
        try:
            '1' - n6
        except TypeError:
            assert True
        else:
            assert False

    def test_rmul(self):
        """Test DualNum rmul"""

        # Mult w float
        n5 = 1.5 * n1
        assert n5.real == 160.5
        assert n5.dual == 1.5

        # Mult w int
        n6 = 2 * n1
        assert n6.real == 214
        assert n6.dual == 2

        # Reject unsupported types
        try:
            '1' * n6
        except TypeError:
            assert True
        else:
            assert False

    def test_rtruediv(self):
        """Test DualNum rtruediv"""

        # Div w float
        n5 = 1.5 / n207
        assert np.isclose(n5.real, 2 * np.e / 3)
        assert np.isclose(n5.dual, 4 / 3)

        # Div w int
        n6 = 2 / n1
        assert np.isclose(n6.real, 53.5)
        assert np.isclose(n6.dual, 0.5)

        # Reject unsupported types
        try:
            '1' / n6
        except TypeError:
            assert True
        else:
            assert False

        # Reject division by 0
        try:
            0 / n1
        except ValueError:
            assert True
        else:
            assert False

    def test_eq(self):
        """Test DualNum eq"""

        # Test two same dual nums
        assert n1 == n1

        # Test scalar with same value as real component of dual number
        assert n1 == 107

        # Reject unsupported types
        try:
            n1 == '1'
        except TypeError:
            assert True
        else:
            assert False

    def test_ne(self):
        """Test DualNum ne"""

        # Test two different dual nums
        assert n1 != n2

        # Test scalar with different value from real component of dual number
        assert n1 != 207

        # Reject unsupported types
        try:
            n1 != '1'
        except TypeError:
            assert True
        else:
            assert False
    
    def test_iadd(self):
        """Test DualNum iadd"""

        # Test two different dual nums
        n3 = DualNumber(107, 1)
        n3 += n2
        assert n3.real == 314
        assert n3.dual == 3

        # Test two of the same dual nums
        n4 = DualNumber(107, 1)
        n4 += n4
        assert n4.real == 214
        assert n4.dual == 2

        # Addn w float
        n5 = DualNumber(107, 1)
        n5 += 1.5
        assert n5.real == 108.5
        assert n5.dual == 1

        # Addn w int
        n6 = DualNumber(107, 1)
        n6 += 10
        assert n6.real == 117
        assert n6.dual == 1

        # Addn of non-DN
        n7 = 107
        n7 += 10
        assert np.isclose(n7, 117)

        # Reject unsupported types
        try:
            n6 += '1'
        except TypeError:
            assert True
        else:
            assert False

    def test_isub(self):
        """Test DualNum isub"""

        # Test two different dual nums
        n3 = DualNumber(107, 1)
        n3 -= n2
        assert n3.real == -100
        assert n3.dual == -1

        # Test two of the same dual nums
        n4 = DualNumber(107, 1)
        n4 -= n4
        assert n4.real == 0
        assert n4.dual == 0

        # Subt w float
        n5 = DualNumber(107, 1)
        n5 -= 1.5
        assert n5.real == 105.5
        assert n5.dual == 1

        # Subt w int
        n6 = DualNumber(107, 1)
        n6 -= 10
        assert n6.real == 97
        assert n6.dual == 1

        # Subt of non-DN
        n7 = 107
        n7 -= 10
        assert n7 == 97

        # Reject unsupported types
        try:
            n6 -= '1'
        except TypeError:
            assert True
        else:
            assert False

    def test_imul(self):
        """Test DualNum imul"""

        # Two of the same dual nums
        n3 = DualNumber(107, 1)
        n3 *= n3
        assert n3.real == 11449
        assert n3.dual == 214

        # Two different complex nums
        n4 = DualNumber(107, 1)
        n4 *= n2
        assert n4.real == 22149
        assert n4.dual == 421

        # Mult w float
        n5 = DualNumber(107, 1)
        n5 *= 1.5
        assert n5.real == 160.5
        assert n5.dual == 1.5

        # Mult w int
        n6 = DualNumber(107, 1)
        n6 *= 2
        assert n6.real == 214
        assert n6.dual == 2

        # Mult of non-DN
        n7 = 107
        n7 *= 2
        assert n7 == 214

        # Reject unsupported types
        try:
            n6 *= '1'
        except TypeError:
            assert True
        else:
            assert False

    def test_itruediv(self):
        """Test DualNum itruediv"""

        # Two of the same dual nums
        n3 = DualNumber(107, 1)
        n3 /= n3
        assert np.isclose(n3.real, 1)
        assert np.isclose(n3.dual, 0)

        # Two different complex nums
        n4 = DualNumber(np.e, 2)
        n4 /= n107
        assert np.isclose(n4.real, 1)
        assert np.isclose(n4.dual, 1 / np.e)

        # Div w float
        n5 = DualNumber(np.e, 2)
        n5 /= 1.5
        assert np.isclose(n5.real, 2 * np.e / 3)
        assert np.isclose(n5.dual, 4 / 3)

        # Div w int
        n6 = DualNumber(107, 1)
        n6 /= 2
        assert np.isclose(n6.real, 53.5)
        assert np.isclose(n6.dual, 0.5)

        # Div of non-DN
        n7 = 107
        n7 /= 2
        assert n7 == 53.5

        # Reject unsupported types
        try:
            n6 /= '1'
        except TypeError:
            assert True
        else:
            assert False

        # Reject division by 0
        try:
            n5 /= 0 
        except ValueError:
            assert True
        else:
            assert False
        
        # Reject division by 0 real component of dual number
        try:
            n6 /= n0
        except ValueError:
            assert True
        else:
            assert False

    def test_pow(self):
        """Test DualNum pow"""

        # Two of the same dual nums
        n3 = n207 ** n207
        assert n3.real == np.e ** np.e
        assert n3.dual == 4 * np.e ** np.e

        # Two different complex nums
        n4 =  n107 ** n207
        assert n4.real == np.e ** np.e
        assert n4.dual == 3 * np.e ** np.e

        # Pow w float
        n5 = n1 ** 2.0
        assert n5.real == 11449
        assert n5.dual == 214

        # Pow w int
        n6 = n1 ** 2
        assert n6.real == 11449
        assert n6.dual == 214

        # Pow of non-DN
        n7 = 107 ** 2
        assert n7 == 11449

        # Reject unsupported types
        try:
            n6 ** '1'
        except TypeError:
            assert True
        else:
            assert False
    
    def test_rpow(self):
        """Test DualNum rpow"""

        # Pow is float
        n5 = 2.0 ** n1
        assert n5.real == 11449
        assert n5.dual == 214

        # Pow is int
        n6 = 2 ** n1
        assert n6.real == 11449
        assert n6.dual == 214

        # Reject unsupported types
        try:
            '1' ** n6
        except TypeError:
            assert True
        else:
            assert False

    def test_neg(self):
        """Test DualNum neg"""

        # Test negation of dual num
        n5 = -n1
        assert n5.real == -107
        assert n5.dual == -1

        # Test negation of non-DN
        n6 = -107
        assert n6 == -107

        # Reject unsupported types
        try:
            -'1'
        except TypeError:
            assert True
        else:
            assert False

    def test_str(self):
        """Test DualNum str"""

        # String representation of DualNumber
        assert(str(n1) == "107 + 1*epsilon")

    def test_repr(self):
        """Test DualNum repr"""

        # Printable representation of DualNumber
        assert(repr(n1) == "DualNumber(107, 1)")
    
    def test_sin(self):
        """Test DualNum sin"""

        # Sin of dual num
        t9 = fad.sin(t1)
        assert np.isclose(t9.real, 0)
        assert np.isclose(t9.dual, -np.pi / 2)

        # Sin of non-DN
        t10 = fad.sin(0)
        assert np.isclose(t10, 0)

        # Reject unsupported types
        try:
            fad.sin('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_cos(self):
        """Test DualNum cos"""

        # Cos of dual num
        t9 = fad.cos(t1)
        assert np.isclose(t9.real, -1)
        assert np.isclose(t9.dual, 0)

        # Cos of non-DN
        t10 = fad.cos(0)
        assert np.isclose(t10, 1)
        
        # Reject unsupported types
        try:
            fad.cos('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_tan(self):
        """Test DualNum tan"""

        # Tan of dual num
        t9 = fad.tan(t1)
        assert np.isclose(t9.real, 0)
        assert np.isclose(t9.dual, np.pi / 2)

        # Tan of non-DN
        t10 = fad.tan(0)
        assert np.isclose(t10, 0)

        # Reject unsupported types
        try:
            fad.tan('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_sec(self):
        """Test DualNum sec"""

        # Sec of dual num
        t9 = fad.sec(t1)
        assert np.isclose(t9.real, -1)
        assert np.isclose(t9.dual, 0)

        # Sec of non-DN
        t10 = fad.sec(0)
        assert np.isclose(t10, 1)

        # Reject unsupported types
        try:
            fad.sec('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_csc(self):
        """Test DualNum csc"""

        # Csc of dual num
        t9 = fad.csc(t3)
        assert np.isclose(t9.real, 1)
        assert np.isclose(t9.dual, 0)

        # Csc of non-DN
        t10 = fad.csc(-np.pi / 2)
        assert np.isclose(t10, -1)

        # Reject unsupported types
        try:
            fad.csc('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_cot(self):
        """Test DualNum cot"""

        # Cot of dual num
        t9 = fad.cot(t3)
        assert np.isclose(t9.real, 0)
        assert np.isclose(t9.dual, 0)

        # Cot of non-DN
        t10 = fad.cot(-np.pi / 2)
        assert np.isclose(t10, 0)

        # Reject unsupported types
        try:
            fad.cot('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_arcsin(self):
        """Test DualNum arcsin"""

        # Arcsin of dual num
        t9 = fad.arcsin(t2)
        assert np.isclose(t9.real, 0)
        assert np.isclose(t9.dual, np.pi)

        # Arcsin of non-DN
        t10 = fad.arcsin(0.5)
        assert np.isclose(t10, np.pi / 6)

        # Reject unsupported types
        try:
            fad.arcsin('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_arccos(self):
        """Test DualNum arccos"""

        # Arccos of dual num
        t9 = fad.arccos(t2)
        assert np.isclose(t9.real, np.pi / 2)
        assert np.isclose(t9.dual, -np.pi)

        # Arccos of non-DN
        t10 = fad.arccos(0.5)
        assert np.isclose(t10, np.pi / 3)

        # Reject unsupported types
        try:
            fad.arccos('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_arctan(self):
        """Test DualNum arctan"""

        # Arctan of dual num
        t9 = fad.arctan(t2)
        assert np.isclose(t9.real, 0)
        assert np.isclose(t9.dual, np.pi)

        # Arctan of non-DN
        t10 = fad.arctan(1)
        assert np.isclose(t10, np.pi / 4)

        # Reject unsupported types
        try:
            fad.arctan('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_sinh(self):
        """Test DualNum sinh"""

        # Sinh of dual num
        n5 = fad.sinh(n107)
        assert np.isclose(n5.real, ee / 2 - ene / 2)
        assert np.isclose(n5.dual, ene / 2 + ee / 2)

        # Sinh of non-DN
        n6 = fad.sinh(np.e)
        assert np.isclose(n6, ee / 2 - ene / 2)

        # Reject unsupported types
        try:
            fad.sinh('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_cosh(self):
        """Test DualNum cosh"""

        # Cosh of dual num
        n5 = fad.cosh(n107)
        assert np.isclose(n5.real, ene / 2 + ee / 2)
        assert np.isclose(n5.dual, ee / 2 - ene / 2)

        # Cosh of non-DN
        n6 = fad.cosh(np.e)
        assert np.isclose(n6, ene / 2 + ee / 2)

        # Reject unsupported types
        try:
            fad.cosh('1')
        except TypeError:
            assert True
        else:
            assert False


    def test_tanh(self):
        """Test DualNum tanh"""

        # Tanh of dual num
        n5 = fad.tanh(n107)
        assert np.isclose(n5.real, ee / (ene + ee) - ene / (ene + ee))
        assert np.isclose(n5.dual, 1 / (ene / 2 + ee / 2) ** 2)

        # Tanh of non-DN
        n6 = fad.tanh(np.e)
        assert np.isclose(n6, ee / (ene + ee) - ene / (ene + ee))

        # Reject unsupported types
        try:
            fad.tanh('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_csch(self):
        """Test DualNum csch"""

        # Csch of dual num
        n5 = fad.csch(n107)
        assert np.isclose(n5.real, 2 / (ee - ene))
        assert np.isclose(n5.dual, -1 / (ee / 2 - ene / 2) / (ene / 2 + ee / 2))

        # Csch of non-DN
        n6 = fad.csch(np.e)
        assert np.isclose(n6, 2 / (ee - ene))

        # Reject unsupported types
        try:
            fad.csch('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_sech(self):
        """Test DualNum sech"""

        # Sech of dual num
        n5 = fad.sech(n107)
        assert np.isclose(n5.real, 2 / (ene + ee))
        assert np.isclose(n5.dual, -1 / (ee / 2 - ene / 2) / (ene / 2 + ee / 2))

        # Sech of non-DN
        n6 = fad.sech(np.e)
        assert np.isclose(n6, 2 / (ene + ee))

        # Reject unsupported types
        try:
            fad.sech('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_coth(self):
        """Test DualNum coth"""

        # Coth of dual num
        n5 = fad.coth(n107)
        assert np.isclose(n5.real, (ene + ee) / (ee - ene))
        assert np.isclose(n5.dual, -1 / (ene / 2 + ee / 2) ** 2)

        # Coth of non-DN
        n6 = fad.coth(np.e)
        assert np.isclose(n6, (ene + ee) / (ee - ene))

        # Reject unsupported types
        try:
            fad.coth('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_exp(self):
        """Test DualNum exp"""

        # Exp of dual num
        n5 = fad.exp(n107)
        assert np.isclose(n5.real, np.e ** np.e)
        assert np.isclose(n5.dual, np.e ** np.e)

        n6 = fad.exp(n9, 10)
        assert np.isclose(n6.real, 10 ** 10)
        assert np.isclose(n6.dual, 10 ** 10 * np.log(10))

        # Exp of non-DN
        n7 = fad.exp(np.e)
        assert np.isclose(n7, np.e ** np.e)

        n8 = fad.exp(10, 10)
        assert np.isclose(n8, 10 ** 10)

        # Reject unsupported types
        try:
            fad.exp('1') and fad.log('1', 10)
        except TypeError:
            assert True
        else:
            assert False

    def test_sqrt(self):
        """Test DualNum sqrt"""

        # Sqrt of dual num
        n5 = fad.sqrt(n107)
        assert np.isclose(n5.real, np.sqrt(np.e))
        assert np.isclose(n5.dual, 1 / (2 * np.sqrt(np.e)))

        # Sqrt of non-DN
        n6 = fad.sqrt(np.e)
        assert np.isclose(n6, np.sqrt(np.e))

        # Ensure user cannot take sqrt of negative numbers
        try:
            fad.sqrt(t7)
        except ValueError:
            assert True
        else:
            assert False

        # Reject unsupported types
        try:
            fad.sqrt('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_log(self):
        """Test DualNum log"""

        # Log of dual num
        n5 = fad.log(n107)
        assert np.isclose(n5.real, 1)
        assert np.isclose(n5.dual, 1 / np.e)

        n6 = fad.log(n9, 10)
        assert np.isclose(n6.real, 1)
        assert np.isclose(n6.dual, 1 / (10 * np.log(10)))

        # Log of non-DN
        n7 = fad.log(np.e)
        assert np.isclose(n7, 1)

        n8 = fad.log(10, 10)
        assert np.isclose(n8, 1)

        # Reject unsupported types
        try:
            fad.log('1') and fad.log('1', 10)
        except TypeError:
            assert True
        else:
            assert False

        # Reject log of negative numbers
        try:
            fad.log(t4) and fad.log(t4, 10)
        except ValueError:
            assert True
        else:
            assert False

    def test_logistic(self):
        """Test DualNum logistic"""

        # Logistic of dual num
        n5 = fad.logistic(n107)
        assert np.isclose(n5.real, 1 / (1 + ene))
        assert np.isclose(n5.dual, ene / (1 + ene) ** 2)

        # Logistic of non-DN
        n6 = fad.logistic(np.e)
        assert np.isclose(n6, 1 / (1 + ene))

        # Reject unsupported types
        try:
            fad.logistic('1')
        except TypeError:
            assert True
        else:
            assert False

    def test_directional_derivative(self):
        """Test DualNum directional_derivative"""

        def fn1(x):
            return fad.sin(x)

        def fn2(x):
            return x ** 2

        def fn3(x, y):
            return x + y

        # directional_derivative of dual num
        dual1 = fad.directional_derivative(fn1, np.pi, 1)
        assert np.isclose(dual1, -1)

        dual2 = fad.directional_derivative(fn2, [1], [-1])
        assert np.isclose(dual2, -2)

        dual2 = fad.directional_derivative(fn3, [1, 1], [-1, -1])
        assert np.isclose(dual2, -np.sqrt(2))

        # Reject unsupported types for fn input
        try:
            fad.directional_derivative('1', np.e, 1)
        except TypeError:
            assert True
        else:
            assert False
    
        # Reject unsupported types for point input
        try:
            fad.directional_derivative(fad.logistic, '1', 1)
        except TypeError:
            assert True
        else:
            assert False

        # Reject unsupported types for dir input
        try:
            fad.directional_derivative(fad.logistic, np.e, '1')
        except TypeError:
            assert True
        else:
            assert False

        # Reject point and dir of different lengths
        try:
            fad.directional_derivative(fad.logistic, np.e, [1,2])
        except ValueError:
            assert True
        else:
            assert False

        # Reject point and dir of different lengths
        try:
            fad.directional_derivative(fad.logistic, [np.e, np.e], [1,2])
        except ValueError:
            assert True
        else:
            assert False
