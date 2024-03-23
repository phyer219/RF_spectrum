'''
存放通用的物理函数, 不仅仅限于此项目
'''
import numpy as np
from scipy import integrate
from scipy.special import roots_legendre as leg


def gauquad(func, a: float, b: float, n: int = 50) -> tuple:
    """ use Gaussian quadrature to integrate :math:`\\int^b_a
    f(x)\\mathrm{d}x`.

    函数 f 的积分区间为 [a,b]
    取 n 个 Legendre 的根
    def Gaussian quadrature integration
    integrate function f from a to b
    take n Legendre roots
    相比于 scipy.integrate.fixed_quad, 此函数不要求输入数组, 可以做多维积分

    Parameters
    ----------
    func : function
           A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    res : float
          The integral of func from `a` to `b`.
    err : float
          占位
    """

    def ft(t):
        return func((b-a)*t/2 + (a+b)/2) * (b-a)/2
    x, w = leg(n)
    res = 0
    for i in range(n):
        res = res + w[i]*ft(x[i])
    err = 0
    return res, err


def ts(func, a: float, b: float, n: int = 51) -> tuple:
    """Tanh-sinh quadrature 方法. 适用于端点发散的情况.

    ref: https://zqw.ink/2019/10/07/NumCal/

    Parameters
    ----------
    func : function
           A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    res : float
          The integral of func from `a` to `b`.
    err : float
          占位
    """
    up = 4
    h = 2*up / (n-1)
    t = np.linspace(-up, up, n, endpoint=True)
    x = np.tanh(1/2*np.pi*np.sinh(t))
    w = 1/2*h*np.pi*np.cosh(t)
    w = w/(np.cosh(1/2*np.pi*np.sinh(t))**2)
    gc = 0
    for i in range(n):
        p = (x[i]*(b-a) + a + b)/2
        gc = gc + func(p)*w[i]
    err = 0
    gc = gc * (b-a)/2
    return gc, err


def bose(beta, energy):
    """ Bose 分布函数
    有些计算中, energy 也可能是负的.
    """
    x = -beta * energy
    if energy > 0:
        return np.exp(x) / (1 - np.exp(x))
    else:
        return 1 / (np.exp(-x) - 1)


def fermi(beta, energy):
    '''
    Fermi distribution function
    '''
    x = -beta * energy
    return np.exp(x) / (1 + np.exp(x))


class PrincipalValueInt():
    """Calculate a 2nd order Cauchy integral.

    :math:`\\int \\mathrm{d}x \\frac{f(x)}{a x^2 + b x + \\mathrm{i}0^+}`

    Note
    ----
    Do not include the `self` parameter in the ``Parameters`` section.

    Parameters
    ----------
    numerator : {function, float}
                numerator :math:`f(x)`
    coeff : list
            [a, b, c] is the coefficients in the dominatior.

    Attributes
    ----------
    get_image : float
                Calculate the imaginary part of the integral.
    get_real : float
        Calculate the real part of the integral.
    """

    def __init__(self, numerator, coeff, down_bound, up_bound, debug=False):
        """初始化, numerator 都是函数. 分母为 a*x**2 + b*x + c"""
        self.debug = debug
        self.numerator = numerator
        if isinstance(numerator, (int, float)):
            self.numerator = lambda _: numerator
        else:
            self.numerator = numerator
        self.down_bound = down_bound
        self.up_bound = up_bound
        self.a = coeff[0]
        self.b = coeff[1]
        self.c = coeff[2]
        self.delta = self.b**2 - 4*self.a*self.c
        self.root_exist = (self.delta > 0) and (self.a != 0)  # a = 0 回到简单的
        # 情况

        if self.root_exist:
            # 如果根存在, 计算两根
            if self.a > 0:
                self.root1 = (-self.b - np.sqrt(self.delta)) / (2*self.a)
                self.root2 = (-self.b + np.sqrt(self.delta)) / (2*self.a)
            else:
                self.root1 = (-self.b + np.sqrt(self.delta)) / (2*self.a)
                self.root2 = (-self.b - np.sqrt(self.delta)) / (2*self.a)
            # 判断两根是否位于积分区间内
            self.root1_in = down_bound < self.root1 and self.root1 < up_bound
            self.root2_in = down_bound < self.root2 and self.root2 < up_bound

    def get_imag(self):
        """计算积分的虚部."""
        if self.root_exist:
            # 计算积分结果
            imag = (self.root1_in) * self.numerator(self.root1)
            imag += (self.root2_in) * self.numerator(self.root2)
            imag *= -np.pi / np.abs(self.root2 - self.root1)
        else:
            # 根不存在, 虚部为 0
            imag = 0
        imag *= 1/self.a  # bug No.2 分子要除以 a 才行.
        return imag

    def get_real(self):
        """
        计算实部的主值积分的方法是用 weight='cauchy'
        """
        # up_bound_optimized = self.up_bound
        # if self.up_bound > 1e4:
        #     # 积分上限很大时, 默认远处 decay 到 0. 要不然算法找不到 peak 的地方,
        #     # 会得出积分结果是 0.
        #     # Mathematica 有类似做法:
        #     # NIntegrate[1/(x-10^8)^2, {x, 1, 10^8+1}] is wrong
        #     # TODO: 不知道有没有更好的处理方法.
        #     # 处理有些粗暴, 更好的做法是将区间细分. 有空再写吧!
        #     up_bound_optimized = np.infty

        if self.root_exist:
            # 计算积分结果
            if (not self.root1_in) and (not self.root2_in):
                if self.debug:
                    print('No root in!!!!!!!!!')
                res = integrate.quad((lambda x: self.numerator(x)
                                      / (self.a*x**2 + self.b*x + self.c)),
                                     self.down_bound, self.up_bound)[0]
                inte_metheod = 'no_root_in'
            elif self.root1_in and self.root2_in:
                if self.debug:
                    print('All root in~~~~~~~~~')
                mid = (self.root2 + self.root1)/2

                real1 = integrate.quad((lambda x: self.numerator(x)
                                        / (self.a * (x - self.root2))),
                                       self.down_bound, mid,
                                       weight='cauchy', wvar=self.root1)[0]

                right_range = 2*self.root2 - self.root1
                if self.up_bound > right_range:
                    # 如果积分上限特别大, 就分段积, 要不然算法找不到 pole 的贡献.
                    if self.debug:
                        print('Big upbound, range has been split!')
                    real2 = integrate.quad((lambda x: self.numerator(x)
                                            / (self.a * (x - self.root1))),
                                           mid, right_range,
                                           weight='cauchy', wvar=self.root2)[0]
                    real2 += integrate.quad((lambda x: self.numerator(x)
                                             / (self.a*x**2 + self.b*x +
                                                self.c)),
                                            right_range, self.up_bound)[0]

                else:
                    real2 = integrate.quad((lambda x: self.numerator(x)
                                            / (self.a * (x - self.root1))),
                                           mid, self.up_bound,
                                           weight='cauchy', wvar=self.root2)[0]
                res = real1 + real2
                inte_metheod = 'all_root_in'
            elif self.root1_in:
                if self.debug:
                    print('root1 in 111111111111111111111')
                res = integrate.quad((lambda x: self.numerator(x)
                                      / (self.a * (x - self.root2))),
                                     self.down_bound, self.up_bound,
                                     weight='cauchy', wvar=self.root1)[0]
                inte_metheod = 'root1_in'
            else:
                if self.debug:
                    print('root2 in 2222222222222222')

                right_range = 2*self.root2 - self.root1
                if self.up_bound > right_range:
                    # 如果积分上限特别大, 就分段积, 要不然算法找不到 pole 的贡献.
                    if self.debug:
                        print('Big upbound, range has been split!')
                    res = integrate.quad((lambda x: self.numerator(x)
                                          / (self.a * (x - self.root1))),
                                         self.down_bound, right_range,
                                         weight='cauchy', wvar=self.root2)[0]
                    res += integrate.quad((lambda x: self.numerator(x)
                                           / (self.a*x**2 + self.b*x +
                                              self.c)),
                                          right_range, self.up_bound)[0]
                else:
                    res = integrate.quad((lambda x: self.numerator(x)
                                          / (self.a * (x - self.root1))),
                                         self.down_bound, self.up_bound,
                                         weight='cauchy', wvar=self.root2)[0]
                inte_metheod = 'root2_in'
        else:
            if self.a == 0:
                if self.debug:
                    print('a = 0 ! 000000000000')
                res = integrate.quad(lambda x: self.numerator(x)/self.b,
                                     self.down_bound, self.up_bound,
                                     weight='cauchy', wvar=-self.c/self.b)[0]
                inte_metheod = 'trivial'
            else:
                if self.debug:
                    print('Root Not Exist!...................')
                res = integrate.quad((lambda x: self.numerator(x)
                                      / (self.a*x**2 + self.b*x + self.c)),
                                     self.down_bound, self.up_bound)[0]
                inte_metheod = 'root_not_exist'
        return res, inte_metheod
