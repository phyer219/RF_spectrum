# from ctypes import pointer
import json
import numpy as np
# import matplotlib.pyplot as plt
from scipy.special import sph_harm, eval_legendre
from scipy.integrate import quad
from scipy import optimize
from scipy.misc import derivative
# from scipy.special import eval_legendre
# from . import codefunc as cf
# from . import phyfunc as pf
# from . import rfio

from . import phyfunc as pf
from .util import counter, timer


def cos_theta_kq(theta_k, phi_k, theta_q, phi_q):
    """
    k, q 夹角的余弦值
    See John David Jackson, Classical Electrodynamics 3rd page 110.
    """
    x = (np.sin(theta_k)*np.sin(theta_q) * np.cos(phi_k - phi_q)
         + np.cos(theta_k)*np.cos(theta_q))
    return x


def theta_q_pole(tk, b, fq) -> list:
    """
    theta_q(phi_q) which satisfy
    cos_theta_kq = b
    b: constant
    tk: theta_k
    fq: phi

    """
    a = - b**2 + np.cos(tk)**2 + np.cos(fq)**2*np.sin(tk)**2
    if a < 0:
        return []
    a = np.sqrt(np.cos(tk)**2 * a)

    def x(a):
        x = b - a * np.cos(fq) / np.cos(tk) * np.tan(tk)
        x /= np.cos(tk)
        x /= 1 + np.cos(fq)**2 * np.tan(tk)**2
        return x

    def y(a):
        y = a + b*np.cos(fq)*np.sin(tk)
        y /= np.cos(tk)**2 + np.cos(fq)**2*np.sin(tk)**2
        return y
    return [np.arctan2(y(a), x(a)), np.arctan2(y(-a), x(-a))]


# def fq_points(tk, c) -> list:
#     a = c**2 - np.cos(tk)**2
#     # print('aaaaaaaaaaaaaaa', a)
#     if a < 0:
#         return []
#     if np.sqrt(a) / np.sin(tk) > 1 or np.sqrt(a) / np.sin(tk) < -1:
#         return []
#     fq1 = np.arccos(np.sqrt(a) / np.sin(tk))
#     fq2 = np.arccos(- np.sqrt(a) / np.sin(tk))
#     # print(fq1, fq2)
#     return [np.abs(fq1),         np.abs(fq2),
#             -np.abs(fq1)+2*np.pi, -np.abs(fq2)+2*np.pi
#             ]


def phi_q_pole(tk, B):
    fqp = np.arccos(np.sqrt(1 + (B**2-1)/np.sin(tk)**2))
    return [fqp, 2*np.pi-fqp]


class NSR_BEC:
    def __init__(self, partial_wave: int, lam, Eb, mu):
        """
        use temperature as unit.

        lam: the character length. for p-wave, lam = R
                                   for d-wave, lam = v
             the magnitude order should satisfy R ~ v^(1/3).
        """
        self.lpd = int(partial_wave)
        self.beta = 1           # inverse temperature
        self.Eb = Eb
        self.mu = mu
        self.lam = lam

        self.phi_k = 0

        if self.Eb > 0 and self.mu >= self.Eb:
            raise ValueError('in BCS side, mu must <= 0')
        elif self.Eb < 0 and self.mu >= self.Eb/2:
            raise ValueError('in BEC side, mu must <= Eb/2')

    def numerator(self, q, k, theta_k, theta_q, phi_q):
        """
        numerator of the self energy
        """
        x = cos_theta_kq(theta_k=theta_k, phi_k=self.phi_k, theta_q=theta_q,
                         phi_q=phi_q)

        const = 2/np.pi * self.lam       # 2????????????4??????????????
        vol_ele = q**2 * np.sin(theta_q)
        kp2 = k**2 + q**2/4 - k*q*x

        theta_k_p = 1/np.sqrt(q**2/4 + k**2 - q*k*x)
        theta_k_p *= (q/2*np.cos(theta_q)
                      - k*np.cos(theta_k))
        theta_k_p = np.arccos(theta_k_p)
        ylm = sph_harm(0, self.lpd, self.phi_k, theta_k_p).real
        # scipy 对于 theta 和 phi 的习惯不一样...

        omega_atom = (k**2 + q**2 - 2*k*q*x)/2 - self.mu
        omega_dimmer = q**2/4 - 2*self.mu + self.Eb
        n_atom = pf.bose(beta=1, energy=omega_atom)
        n_dimmer = pf.bose(beta=1, energy=omega_dimmer)
        n_diff = n_atom - n_dimmer

        numerator = const * vol_ele * kp2**self.lpd * ylm**2 * n_diff
        return numerator

    def sigma_thetaq_phiq(self, omega, k, theta_k, theta_q, phi_q):
        """
        self energy function.
        """
        phi_k = 0
        x = cos_theta_kq(theta_k=theta_k, phi_k=phi_k, theta_q=theta_q,
                         phi_q=phi_q)
        coeff = [1/4, -k*x, omega + k**2/2 + self.mu - self.Eb]

        imag = pf.PrincipalValueInt(lambda q: self.numerator(q=q, k=k,
                                                             theta_k=theta_k,
                                                             theta_q=theta_q,
                                                             phi_q=phi_q),
                                    coeff=coeff,
                                    down_bound=0, up_bound=np.inf).get_imag()
        sigma = imag
        return sigma

    # @timer
    def sigma_phiq(self, omega, k, theta_k, phi_q, limit=50):
        """ 积掉 theta_q, 还剩 phi_q
        """
        def foo(theta_q):
            foo = self.sigma_thetaq_phiq(omega=omega, k=k, theta_k=theta_k,
                                         theta_q=theta_q, phi_q=phi_q)
            return foo

        c = omega + k**2/2 + self.mu - self.Eb
        inte_points = []
        # print(c, 'ccccccccccc')
        if c >= 0:
            p_0 = theta_q_pole(tk=theta_k, b=0, fq=phi_q)
            if p_0:
                inte_points += p_0
            p_c = theta_q_pole(tk=theta_k, b=np.sqrt(c)/k, fq=phi_q)
            if p_c:
                inte_points += p_c
        # print('phiq.............', phi_q)
        # print(inte_points)
        sigma_phiq = quad(foo, 1e-10, np.pi, points=inte_points, limit=limit)[0]
        return sigma_phiq

    def sigma(self, omega, k, theta_k):
        """积掉 phi_q. only imag part"""
        def foo(phi_q):
            return self.sigma_phiq(omega=omega, k=k,
                                   theta_k=theta_k, phi_q=phi_q)
        c = omega + k**2/2 + self.mu - self.Eb
        inte_points = []
        if c >= 0:
            B = np.sqrt(c)/k
            if B**2 > np.cos(theta_k)**2:
                fqp = phi_q_pole(tk=theta_k, B=B)
                inte_points += fqp
        sigma = quad(foo, 1e-10, 2*np.pi, points=inte_points)[0]
        return sigma

    def get_density(self):
        """
        As a function of: mu, Eb.
        for two component p-wave,
        use the single effective componet fermi energy.

        Because the dimmer term, we must have mu <= Eb/2.
        if mu >= Eb/2, there will be no dimmer?

        In BCS side , Eb > 0 (nonphysical?).
        In BEC side, Eb < 0. The dimmer critial temperature is determined by
        mu = Eb/2
        """
        if not hasattr(self, 'density'):
            def atom(q):
                at = q**2 * pf.bose(beta=self.beta, energy=q**2/2-self.mu)
                at /= self.lpd*np.pi**2
                return at

            def fluc(q):
                fl = q**2 * pf.bose(beta=self.beta,
                                    energy=q**2/4-2*self.mu+self.Eb)
                fl /= np.pi**2
                return fl
            n_atom = quad(atom, 0, np.infty)[0]
            # n_atom += quad(atom, 1, np.infty)[0]
            n_fluc = quad(fluc, 0, np.infty)[0]
            n_total = n_fluc + n_atom
            self.n_fluc = n_fluc
            self.density = n_total
        return self.density

    def get_fermi_momentum(self):
        """
        """
        if not hasattr(self, 'fermi_momentum'):
            self.fermi_momentum = (3 * self.lpd *
                                   np.pi**2 * self.get_density())**(1/3)
        return self.fermi_momentum

    def get_fermi_energy(self):
        """
        unit of figure, fermi energy E_F/T.

        A QUESTION: use the total number or atom number?
        maybe the total number. treat the dimmer as two atom, not a moleculer.
        """
        if not hasattr(self, 'fermi_energy'):
            self.fermi_energy = self.get_fermi_momentum()**2 / 2
        return self.fermi_energy

    def contact(self):
        def foo(q):
            return q**2 * pf.bose(beta=1, energy=q**2/4-2*self.mu+self.Eb)
        return quad(foo, 0, 10)[0]*self.lam / (2*np.pi**2)

    def spectralfunc(self, k, omega, theta_k, real=0):
        """
        only the pair contribution, not contain the free atom delta part.
        Returns:
        A_k_o: spectrum function
        imag: imag
        """
        if omega < k**2/2 - self.mu + self.Eb:
            xi_k = k**2/2 - self.mu
            imag = self.sigma(omega=omega, k=k, theta_k=theta_k)
            denominator = (omega - xi_k - real)**2 + imag**2
            if imag == 0:
                # check 一下解析分析, 确认此区间内虚部不为零
                print('WARNING: Analyse wrong!!!??? suspect delta peak')
                print('if k is large, then wen can neglect this peak')
                print('k is:', k)
                print('theta_k is:', theta_k)
                A_k_o = 0
            elif denominator == 0:
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                print('denominator of A(k, Omega) is zero. see if really zero')
                print('imag, omega, xi_k is:', imag, omega, xi_k)
                print(denominator)
                A_k_o = 0
            else:
                A_k_o = imag / ((omega - xi_k - real)**2 + imag**2)
                A_k_o *= -2
        else:
            A_k_o = 0
            imag = 0
        return A_k_o, imag

    def n_k_ometa_theta_k(self, k, omega, theta_k,
                          node_container=[], value_container=[]):
        """
        n(k, Omega)
        in order to check if the integral cut off is good, we recorded the
        calculated points in lists node_container and value_container.
        """
        # imag = self.sigma(omega=omega, k=k, theta_k=theta_k)
        A_k_o, imag = self.spectralfunc(k=k, omega=omega, theta_k=theta_k)
        if imag == 0:
            # check 一下解析分析, 确认此区间内虚部不为零
            print('WARNING: Analyse wrong!!!')
        n_k_o = 1/(2*np.pi) * pf.bose(beta=self.beta, energy=omega) * A_k_o
        node_container.append(omega)
        value_container.append(n_k_o)
        return n_k_o

    @counter
    @timer
    def n_k(self, k, theta_k, real=0, limit=10):
        """
        Returns:
        n: particle number
        nodes and values recorded the integrate points, in order to check if
        the cut off is good.
        """
        # @counter
        # @timer
        def n_unint(omega, node_container=[], value_container=[]):
            """
            in order to check if the integral cut off is good, we recorded the
            calculated points in lists node_container and value_container.
            """
            # imag = self.sigma(omega=omega, k=k, theta_k=theta_k)
            A_k_o, imag = self.spectralfunc(k=k, omega=omega,
                                            theta_k=theta_k, real=real)
            if imag == 0:
                # check 一下解析分析, 确认此区间内虚部不为零
                print('WARNING: Analyse wrong!!!')
            n_k_o = 1/(2*np.pi) * pf.bose(beta=self.beta, energy=omega) * A_k_o
            node_container.append(omega)
            value_container.append(n_k_o)
            return n_k_o

        omega_boundary = k**2/2-self.mu+self.Eb
        nodes = []
        values = []

        if self.Eb < 0:
            # BEC side, always a delta peak
            n_delta_part = pf.bose(beta=1, energy=k**2/2-self.mu)

            epsabs = 0.001*n_delta_part
            if epsabs < 1.49e-08:
                epsabs = 1.49e-08
                print('==============n_delta_part is so small!!!!!!!')
            if k < 2.5:
                n = quad(lambda omega: n_unint(omega, node_container=nodes,
                                               value_container=values),
                         -25, omega_boundary,
                         limit=limit, epsabs=epsabs)[0]
            else:
                n = quad(lambda omega: n_unint(omega,
                                               node_container=nodes,
                                               value_container=values),
                         -10*k, omega_boundary,
                         limit=limit, epsabs=epsabs)[0]
            # print('=======================integral part is', n)
            n += n_delta_part
            # print('=======================integral part + delta part is', n)

        else:
            # BCS side, delta peak is merged into the continumu
            # if self.lpd == 2:
            #     # for d-wave BCS side, we must choose such a high precision,
            #     # because the tail is too small. The differene between the n_k
            #     # and n for free atom, only can bee see smoothly with such a
            #     # high precision.
            #     epsabs = 1e-11
            #     limit = 60
            # else:
            #     epsabs = 1e-8
            epsabs = 1e-8
            limit = 50
            omega_delta = k**2/2-self.mu
            imag = self.sigma(omega=omega_delta, k=k, theta_k=theta_k)
            if imag == np.nan:
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                imag = .1
                print('omega=', omega_delta, 'k=', k, 'theta_k=', theta_k)
            print('?????????????????????????', imag,
                  omega_delta, k, theta_k)
            inte_points = []
            inte_points.append(omega_delta - 100*np.abs(imag))
            inte_points.append(omega_delta - 10*np.abs(imag))
            # I have used [omega_delta +- np.abs(imag)] but it wiil produce
            # round off error at some points, which will result in a
            # discontinuty in the n(k).
            if omega_delta + 10*np.abs(imag) < omega_boundary:
                inte_points.append(omega_delta + 10*np.abs(imag))
            if omega_delta + 100*np.abs(imag) < omega_boundary:
                inte_points.append(omega_delta + 100*np.abs(imag))
            # print("imag of sigma is:", imag)
            # print("omega_boundary is:", omega_boundary)
            # print("omega_delta is:", omega_delta)
            # print("integral points is:", inte_points)
            print('################# k, theta_k: ', k, theta_k)
            n_res = quad(lambda omega: n_unint(omega,
                                               node_container=nodes,
                                               value_container=values),
                         omega_delta - 1000*np.abs(imag),
                         k**2/2-self.mu+self.Eb, limit=limit,
                         points=inte_points, epsabs=epsabs)
            n = n_res[0]
            print('integral res and error is:', n_res)
            print('imag is )))))))))))', imag)
            print('k is:::::::', k)
            n += quad(lambda omega: n_unint(omega,
                                            node_container=nodes,
                                            value_container=values),
                      -np.inf, omega_delta - 1000*np.abs(imag),
                      limit=limit, epsabs=epsabs)[0]
        idx = np.argsort(np.array(nodes))
        nodes = np.array(nodes)[idx].tolist()
        values = np.array(values)[idx].tolist()
        return n, nodes, values

    def rf_spec_k_theta_k(self, nu, k, theta_k):
        xi_k = k**2/2 - self.mu
        omega = xi_k - nu
        # imag = self.sigma(omega=omega, k=k, theta_k=theta_k)
        ako = self.spectralfunc(k=k, omega=omega, theta_k=theta_k)[0]
        rf_k = 1/2 * ako * (pf.bose(beta=1, energy=omega))
        rf_k *= k**2 * np.sin(theta_k)
        rf_k /= 4 * np.pi**2
        return rf_k

    def rf_spec_theta_k(self, nu, theta_k, limit=3):

        # @cf.counter
        def foo(k):
            return self.rf_spec_k_theta_k(nu=nu, k=k, theta_k=theta_k)

        # if nu <= 50:
        #     bound_a = 0
        #     bound_b = 10
        # else:
        #     print('Omega not support!!!!!!!!!!!!!!!!!')
        #     bound_a = 0
        #     bound_b = 10
        cut = 10
        # rf1 = quad(foo, bound_a, bound_b, limit=limit)[0]
        rf = quad(foo, 0, cut, limit=limit)[0]
        rf += quad(foo, cut, np.inf, limit=limit)[0]
        return rf

    @timer
    def rf_spec(self, nu, limit=3):
        """nu elements (-Eb, inf]"""

        @timer
        @counter
        def fin_int(theta_k):
            return self.rf_spec_theta_k(nu=nu, theta_k=theta_k)

        rf = quad(fin_int, 1e-8, np.pi, limit=limit)[0]
        return rf

    @timer
    def n_k_l(self, k, legendre_l, limit=10):
        nkl = quad(lambda tk:
                   (self.n_k(k=k, theta_k=tk, limit=limit)[0]
                    * np.sin(tk) * eval_legendre(legendre_l, np.cos(tk))),
                   0, np.pi, limit=3)
        return nkl

    def n_k_l_tail(self, k, legendre_l):
        """
        int_lgd = integrate(sin(theta_k) * P_legendre_l(cos(theta_k))
               * Y_lpd, m(cos(theta_k)))
        """
        if legendre_l == 0:
            int_lgd = 1 / (2*np.pi)
        elif legendre_l == 2:
            if self.lpd == 1:
                int_lgd = 1 / (5*np.pi)
            elif self.lpd == 2:
                int_lgd = 1 / (7*np.pi)
            else:
                raise Exception('Not proper legendre or lpd.'
                                + f'lpd is {self.lpd}')
        elif legendre_l == 4:
            if self.lpd == 1:
                int_lgd = 0
            elif self.lpd == 2:
                int_lgd = 1 / (7*np.pi)
            else:
                raise Exception('Not proper legendre or lpd 2')
        else:
            raise Exception('Not proper legendre or lpd 3')
        nkl = 16*np.pi**2 * k**(2*self.lpd-4) * self.contact() * int_lgd
        return nkl
# ===============================Virial Expansion =============================

    def dphase(self, k, Eb):
        dp = -k**2 * (k**2 - 3*Eb) * self.lam
        dp /= Eb**2 - 2*Eb*k**2 + k**4 + k**6*self.lam**2
        return dp
    def dphase_d(self, k, Eb):
        dp = -5*Eb*k**4*self.lam + 3*k**6*self.lam
        dp /= k**10*self.lam**2 + (Eb - k**2)**2
        dp *= -1
        return dp
    def virialp(self, Eb):
        """ Delta b_2"""
        dbt1 = np.exp(-Eb)
        dbt2 = 1/np.pi  # 由于只考虑了 m = 0 的分量, 所以这里没有系数 2l + 1
        if self.lpd==1:
            dbt2 *= quad(lambda k: self.dphase(k, Eb)*np.exp(-k**2), 0,
                         np.inf)[0]
        elif self.lpd==2:
            dbt2 *= quad(lambda k: self.dphase_d(k, Eb)*np.exp(-k**2), 0,
                         np.inf)[0]
        dbt = dbt2
        if self.Eb < 0:
            dbt += dbt1
        dbt *= np.sqrt(2)
        return dbt

    def virial_contact(self):
        if self.lpd == 2:
            def foo(Eb):
                return self.virialp(Eb=Eb)
            ddbt = derivative(foo, self.Eb, dx=1e-6)*(-self.lam)
            # d/dD-1 = dEb/dD-1 * d/dEb
        elif self.lpd == 1:
            def foo(x):
                vm = 1/x
                Eb = -self.lam/vm
                return self.virialp(Eb=Eb)

            vm = - self.lam/self.Eb
            ddbt = derivative(foo, 1/vm, dx=1e-6)
        Q = 2/(2*np.pi)**(3/2)
        vc = Q * np.exp(2*self.mu) * ddbt
        # vana = 2*np.sqrt(2)*lam/(2*np.pi)**(3/2) * np.exp(-Eb) * np.exp(2*mu)
        return vc
# ===============================Virial Expansion End==========================


def find_mu(Eb_T, Eb_Ef, lpd, lam_kf, save_parameters=True):
    """find mu/T when Eb/T fixed
    for p-wave, lam = Rm
    for d-wave, lam = vm
    lam is a free parameter. We can fix kf*Rm for p-wave or kf*3*lam for d
    wave, then output the lam in unit of T, in order to use in numerical
    calculation.
    """
    lam = lam_kf
    mu = optimize.root(lambda mu:
                       Eb_T/NSR_BEC(partial_wave=lpd, lam=lam, Eb=Eb_T,
                                    mu=mu).get_fermi_energy() - Eb_Ef,
                       Eb_T/2-10).x[0]
    nb = NSR_BEC(partial_wave=lpd, lam=lam, Eb=Eb_T, mu=mu)
    ef = nb.get_fermi_energy()
    # kf = nb.get_fermi_momentum()
    # muef = mu/ef
    T_Ef = 1/ef
    kf = nb.get_fermi_momentum()
    if lpd == 1:
        lam = lam / kf
    elif lpd == 2:
        lam = lam / kf**3
    else:
        raise Exception('lpd not right!!!!!!!!!')
    # lam = 1 / (lam*kf)
    parameters = {'lam': lam,
                  'Eb': Eb_T,
                  'mu': mu,
                  'lpd': lpd,
                  'T_Ef': T_Ef,
                  'E_b_Ef': Eb_T/ef,
                  }
    if save_parameters:
        print(parameters)
        with open('parameters.json', 'w') as f:
            json.dump(parameters, f)
    return {'partial_wave': lpd,
            'lam': lam,
            'Eb': Eb_T,
            'mu': mu
            }
