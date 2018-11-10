# Rohit Gangupantulu
# Black-Scholes implementation attempt with basic options

import numpy as np
import pandas as pd
import datetime as dt
from calendar_kb import Calendar_kb
import scipy as sc
from scipy import stats
import matplotlib.pyplot as plt



class Black_Scholes_kb(Calendar_kb):
        def __init__(self, type_option, S0, K, r, t, volatility, dr, fr, divid, today, maturity_date,
                     day_convention):
            Calendar_kb.__init__(self, today, maturity_date, day_convention)
            self.type_option = type_option
            self.S0 = S0
            self.K = K
            self.r = r
            self.t = t # time to maturity
            self.volatility = volatility
            self.divid = divid
            self.dr = dr  # domestic free rate
            self.fr = fr


        def d1_fun(self):
            d1 = (np.log(self.S0 / self.K) + (
                    self.r - self.divid + 0.5 * self.volatility ** 2) * self.year_fraction_two_dates_fun()) / (
                         np.sqrt(self.year_fraction_two_dates_fun()) * self.volatility)
            return d1

        def d2_fun(self):
            d2 = (np.log(self.S0 / self.K) + (
                    self.r - self.divid - 0.5 * self.volatility ** 2) * self.year_fraction_two_dates_fun()) / (
                         np.sqrt(self.year_fraction_two_dates_fun()) * self.volatility)
            return d2

        def days_to_maturity(self):
            return self.maturity_date.toordinal()-self.today.toordinal()


        def black_scholes_price_fun(self):
            d1 = self.d1_fun()
            d2 = self.d2_fun()
            if (self.type_option == self.type_option):
                price = self.S0 * np.exp(-self.year_fraction_two_dates_fun() * self.divid) * sc.stats.norm.cdf(d1, 0,
                                                                                                                      1) - self.K * np.exp(
                    -self.year_fraction_two_dates_fun() * self.r) * stats.norm.cdf(d2, 0, 1)
            else:
                price = self.K * np.exp(-self.year_fraction_two_dates_fun() * self.r) * stats.norm.cdf(-d2, 0,
                                                                                                              1) - self.S0 * np.exp(
                    -self.year_fraction_two_dates_fun() * self.divid) * stats.norm.cdf(-d1, 0, 1)
            return price

        def simulationST(self,n):
            ST=np.zeros(n)
            for i in range(len(ST)):
                z = np.random.standard_normal()
                ST[i]=self.S0*np.exp(self.r-0.5*self.volatility**2*self.year_fraction_two_dates_fun()+self.volatility*np.sqrt(self.year_fraction_two_dates_fun())*z)
            return ST


        def payoff_simulation(self):#delivery_data random or given
            ST=self.simulationST(n=30)

            vpayoff=np.zeros(len(ST))
            for i in range(len(ST)):
                if (self.type_option=='call'):
                    vpayoff[i]=max(ST[i]-self.K ,0)
                else:
                    vpayoff[i]=max(self.K-ST[i],0)
            zipped=list((zip(ST,vpayoff)))
            sorted_zip=sorted(zipped, key=lambda x: x[0]) #without sorting there is a problem with ploting
            return sorted_zip




        def plot_payoff(self):
            under_and_payoff=self.payoff_simulation()
            x_val = [x[0] for x in under_and_payoff]
            y_val = [x[1] for x in under_and_payoff]
            plt.plot(x_val,y_val)
            plt.show()


        def modify_parameter(self,scope):
            return [[self.black_scholes_price_fun() for self.S0 in scope],scope] #to zmieniamy parametr ktory chcey modyfikowac


if __name__ == '__main__':
    o_black_scholes_6m = Black_Scholes_kb(type_option='call',
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

    o_black_scholes_3m=Black_Scholes_kb(type_option='call',
                                                   S0=41,
                                                   K=40,
                                                   r=0.08,
                                                   t=0,
                                                   volatility=0.25,
                                                    today=dt.date(2018,9,8),
                                                    maturity_date=dt.date(2018,12,8),
                                                    day_convention='Actual/365',
                                                   dr=0.02,  # domestic rate
                                                   fr=0.018,
                                                   divid=0)  # foreing rate

    o_black_scholes_7d=Black_Scholes_kb(type_option='call',
                                                   S0=41,
                                                   K=40,
                                                   r=0.08,
                                                   t=0,
                                                   volatility=0.25,
                                                    today=dt.date(2018,12,1),
                                                    maturity_date=dt.date(2018,12,8),
                                                    day_convention='Actual/365',
                                                   dr=0.02,  # domestic rate
                                                   fr=0.018,
                                                   divid=0)  #



    # shows how the option price is related to increase of underlying and decrease maturity
    increase_price6m=o_black_scholes_6m.modify_parameter(scope=np.arange(30,50))[0]
    increase_price3m=o_black_scholes_3m.modify_parameter(scope=np.arange(30,50))[0]
    increase_price7d=o_black_scholes_7d.modify_parameter(scope=np.arange(30,50))[0]

    plt.plot(np.arange(30,50),increase_price6m, label="6m to maturity")
    plt.plot(np.arange(30,50),increase_price3m,label="3m to maturity")
    plt.plot(np.arange(30,50),increase_price7d,'--',label="7d to maturity")
    plt.xlabel("Spot Price")
    plt.ylabel("Option Price")
    plt.title("Relation price option to maturity")
    plt.legend()
    plt.show()


    # shows how option price is related to increase of underlying and decrease maturity

    print("The End")
