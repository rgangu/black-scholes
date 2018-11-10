# Rohit Gangupantulu
# Derivatives sensitivity measurement

import numpy as np
import pandas as pd
import datetime as dt
from calendar_kb import Calendar_kb
from black_scholes_ver03 import Black_Scholes_kb
import scipy as sc
from scipy import stats
import matplotlib.pyplot as plt


class GreeksParameters_kb(Black_Scholes_kb):
    def __init__(self, type_option, S0, K, r, t, volatility, dr, fr, divid, today, maturity_date, day_convention):
        Black_Scholes_kb.__init__(self, type_option, S0, K, r, t, volatility, dr, fr, divid, today, maturity_date,
                                  day_convention)

    def delta_fun(self, scope):
        vdelta = []
        for self.S0 in scope:
            d1 = self.d1_fun()
            if (self.type_option == self.type_option):
                delta = stats.norm.cdf(d1, 0, 1)
            else:
                delta = -stats.norm.cdf(-d1, 0, 1)
            vdelta.append(delta)
        return vdelta

    def gamma_fun(self, scope):
        vgamma = []
        for self.S0 in scope:
            d1 = self.d1_fun()
            gamma = stats.norm.pdf(d1, 0, 1) / (self.S0 * self.volatility * np.sqrt(self.year_fraction_two_dates_fun()))
            vgamma.append((gamma))
        return vgamma

    def vega_fun(self, scope):
        vvega = []
        for self.volatility in scope:
            d1 = (np.log(self.S0 / self.K) + (
                        self.r - self.divid + 0.5 * self.volatility ** 2) * self.year_fraction_two_dates_fun()) / (
                         np.sqrt(self.year_fraction_two_dates_fun()) * self.volatility)
            vega = self.S0 * np.sqrt(self.year_fraction_two_dates_fun()) * stats.norm.pdf(d1, 0, 1)
            vvega.append(vega)
        return vvega


if __name__ == '__main__':
    greeks_6m = GreeksParameters_kb(type_option='call',
                                    S0=41,
                                    K=40,
                                    r=0.08,
                                    t=0,
                                    volatility=0.25,
                                    today=dt.date(2018, 6, 8),
                                    maturity_date=dt.date(2018, 12, 8),
                                    day_convention='Actual/365',
                                    dr=0.02,  # domestic rate
                                    fr=0.018,
                                    divid=0)  # foreing rate

    greeks_3m = GreeksParameters_kb(type_option='call',
                                 S0=41,
                                 K=40,
                                 r=0.08,
                                 t=0,
                                 volatility=0.25,
                                 today=dt.date(2018, 9, 8),
                                 maturity_date=dt.date(2018, 12, 8),
                                 day_convention='Actual/365',
                                 dr=0.02,  # domestic rate
                                 fr=0.018,
                                 divid=0)  # foreing rate

    greeks_7d = GreeksParameters_kb(type_option='call',
                                    S0=41,
                                    K=40,
                                    r=0.08,
                                    t=0,
                                    volatility=0.25,
                                    today=dt.date(2018, 12, 1),
                                    maturity_date=dt.date(2018, 12, 8),
                                    day_convention='Actual/365',
                                    dr=0.02,  # domestic rate
                                    fr=0.018,
                                    divid=0)  # fore

    delta_v_6m = greeks_6m.delta_fun(scope=np.arange(30, 56))
    delta_v_3m=greeks_3m.delta_fun(scope=np.arange(30, 56))
    delta_v_7d = greeks_7d.delta_fun(scope=np.arange(30, 56))
    plt.plot(np.arange(30, 56), delta_v_6m,label="6 months")
    plt.plot(np.arange(30, 56), delta_v_3m,label="3 months")
    plt.plot(np.arange(30, 56), delta_v_7d,label="7 days")
    plt.xlabel("Spot Price")
    plt.ylabel("Delta")
    plt.title("Delta of Call Options")
    plt.legend()
    plt.show()

    delta_v_6m = greeks_6m.gamma_fun(scope=np.arange(30, 56))
    delta_v_3m = greeks_3m.gamma_fun(scope=np.arange(30, 56))
    delta_v_7d = greeks_7d.gamma_fun(scope=np.arange(30, 56))
    plt.plot(np.arange(30, 56), delta_v_6m, label="6 months")
    plt.plot(np.arange(30, 56), delta_v_3m, label="3 months")
    plt.plot(np.arange(30, 56), delta_v_7d, label="7 days")
    plt.xlabel("Spot Price")
    plt.ylabel("Delta")
    plt.title("Delta of Call Options")
    plt.legend()
    plt.show()