import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import *
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


def generate_forward_rate():
    # Step 1: Read data
    capdata = pd.read_excel('Data/capdata.xlsx')
    capdata['Expiry'] = pd.to_datetime(capdata['Expiry'], format='%m/%d/%Y')
    today = datetime(2019, 7, 16)
    # Step 2: Compute year fraction and sort
    capdata['T'] = (capdata['Expiry'] - today).dt.days / 365
    capdata = capdata[capdata['T'] > 0].sort_values('T')
    # Step 3: Take T and Discount
    capdata['tau_i']= capdata['tau_i']
    T_market = capdata['T'].values
    Tau_market = capdata['tau_i'].values
    P_market = capdata['Discount'].values
    # Step 4: Cubic spline interpolation on ln(P)
    log_P = np.log(P_market)
    cs = CubicSpline(T_market, log_P, bc_type='natural')
    # Step 5: Evaluate forward rates
    T_eval = np.linspace(T_market.min(), T_market.max(), 500)
    fM = -cs.derivative()(T_eval)
    # Step 6: Finite difference approximation
    df_raw = -np.gradient(log_P, 0.25)
    # df_raw = -np.gradient(log_P, 0.25)
    # Step 7: Short rate at first maturity point
    r0 = -np.log(P_market[0]) / T_market[0]
    spline_df = pd.DataFrame({
        'T_eval': T_eval,
        'ForwardRate_spline': fM
    })
    spline_df.to_csv('forward_rate_spline.csv', index=False)

    finite_diff_df = pd.DataFrame({
        'T_market': T_market,
        'Discount': P_market,
        'ForwardRate_fd': df_raw
    })
    finite_diff_df.to_csv('forward_rate_finite_diff.csv', index=False)
    # Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(T_eval, fM, label='Cubic Spline Forward Rates')
    # plt.scatter(T_market, df_raw, color='red', label='Raw Finite Difference')
    # plt.axhline(r0, color='green', linestyle='--', label=f'Short Rate r(0) = {r0:.2%}')
    # plt.xlabel('Maturity (Years)')
    # plt.ylabel('Forward Rate')
    # plt.legend()
    # plt.title('Instantaneous Forward Rate Curve (From Excel Market Data)')
    # plt.grid(True)
    # plt.show()

generate_forward_rate()
# Load data
capdata = pd.read_excel('Data/capdata.xlsx')
capdata['Expiry'] = pd.to_datetime(capdata['Expiry'], format='%m/%d/%Y')
# Preprocess capdata
market_curve = pd.read_csv('zero_curve.xlsx')
capdata['forward_rate'] = 1 / capdata['tau_i'] * (capdata['Discount'].shift(1) / capdata['Discount'] - 1)

capdata['forward_rate'].iloc[0] = 1 / market_curve['Year'].iloc[0] * (
            market_curve['Discount'].iloc[0] / capdata['Discount'].iloc[0] - 1)

# Extract market parameters
caplet_price = capdata['PV'].values
tau_list = capdata['tau_i'].values
discount_list = capdata['Discount'].values
reseting_date_list = capdata['T_iM1'].values
N_list = np.ones(len(caplet_price)) * 10000000
X_list = capdata['CapStrike'].values / 100
discount_list = np.concatenate(([market_curve['Discount'].iloc[0]], discount_list))
marketo = -np.log(market_curve['Discount'].iloc[0]) / 0.25



def generate_short_rate():
    capdata = pd.read_excel('Data/capdata.xlsx')
    capdata['Expiry'] = pd.to_datetime(capdata['Expiry'], format='%m/%d/%Y')
    today = datetime(2019, 7, 16)
    capdata['year_frac'] = (capdata['Expiry'] - today).dt.days / 365
    capdata['spot_rate'] = -np.log(capdata['Discount']) / capdata['year_frac']
    capdata['spot_rate'].to_csv('Data/spot_rates.csv', index=False)


# Hull-White model functions
def Caplets(X, N, tau_i, TM1_i, P_T, P_S, a, sigma):
    X_new = 1 / (1 + X * tau_i)
    N_new = N * (1 + X * tau_i)
    B = 1 / a * (1 - np.exp(-a * (tau_i)))
    sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * TM1_i)) / (2 * a)) * B
    h = 1 / sigma_p * np.log((P_S / P_T) / X_new) + sigma_p / 2
    price = N_new * (X_new * P_T * stats.norm.cdf(-h + sigma_p) - P_S * stats.norm.cdf(-h))
    return price


def HM_ZB_CF(t, T, rt, FMT, PMT, PMt, a, sigma):
    B = 1 / a * (1 - np.exp(-a * (T - t)))
    A = PMT / PMt * np.exp(B * FMT - sigma ** 2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B ** 2)
    Price = A * np.exp(-B * rt)
    return Price


def Caplet_list(ro, a, sigma, tau_list, reseting_date_list, discount_list, X_list, M_list, marketo):
    bond_price_list = np.zeros(len(reseting_date_list) + 1)
    Caplet_price_list = np.zeros(len(reseting_date_list))
    FMT = marketo

    for i in range(len(reseting_date_list) + 1):
        if i == 0:
            T = reseting_date_list[i]
        else:
            T = reseting_date_list[i - 1] + tau_list[i - 1]
        PMT = discount_list[i]
        bond_price_list[i] = HM_ZB_CF(0, T, ro, ro, PMT, 1, a, sigma)

    for j in range(len(reseting_date_list)):
        X = X_list[j]
        M = M_list[j]
        tau_j = tau_list[j]
        TM_Lj = reseting_date_list[j]
        P_T = bond_price_list[j]
        P_S = bond_price_list[j + 1]
        Caplet_price_list[j] = Caplets(X, M, tau_j, TM_Lj, P_T, P_S, a, sigma)

    return Caplet_price_list


def Target_Error(x, caplet_price, tau_list, reseting_date_list, discount_list, X_list, M_list, marketo,
                 calibration_objective='caplet'):
    price_list = Caplet_list(x[0], x[1], x[2], tau_list, reseting_date_list, discount_list, X_list, M_list, marketo)
    if calibration_objective == 'cap':
        price_list = price_list.cumsum()
        caplet_price = caplet_price.cumsum()
    mse = ((price_list - caplet_price) ** 2).sum()
    return mse


# # Model calibration
# fun = Target_Error
# calibration_objective = 'caplet'
# x0 = np.array([marketo, 0.2, 0.05])
# args = (caplet_price, tau_list, reseting_date_list, discount_list, X_list, N_list, marketo, calibration_objective)
# res = minimize(fun, x0, method='trust-constr', args=args, options={'disp': True})
#
# print('Optimized solution (a, sigma):')
# print('Squared Error:', res.fun)
# print('Optimized Solution:', res.x)
# print('Success:', res.success)
# print('Message:', res.message)
#
# # Plot results
# caplet_NM = Caplet_list(res.x[0], res.x[1], res.x[2], tau_list, reseting_date_list, discount_list, X_list, N_list,
#                         marketo)
#
# plt.figure(figsize=(10, 6))
# plt.plot(reseting_date_list, caplet_NM, label='Hull-White model')
# plt.plot(reseting_date_list, caplet_price, label='Market data')
# plt.xlabel('Reset Date')
# plt.ylabel('Caplet Price')
# plt.legend()
# plt.show()


# Volatility conversion functions
def Black_Cap_Pricing(tau_list, TWA_list, discount_list, forward_list, N, K, sigma):
    price_list = []
    for i in range(len(tau_list)):
        F = forward_list[i]
        P_t = discount_list[i]
        tau_i = tau_list[i]
        v_i = sigma * np.sqrt(TWA_list[i])
        d1 = (np.log(F / K) + v_i ** 2 / 2) / v_i
        d2 = (np.log(F / K) - v_i ** 2 / 2) / v_i
        price = N * P_t * tau_i * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        price_list.append(price)
    return np.sum(price_list), price_list


def Price_to_Vol(Optimal_price, tau_list, TWL_list, discount_list, forward_list, N, K, tolerance=1e-8):
    lower, upper = 1e-6, 1.0
    flower = Black_Cap_Pricing(tau_list, TWL_list, discount_list, forward_list, N, K, lower)[0] - Optimal_price
    while Black_Cap_Pricing(tau_list, TWL_list, discount_list, forward_list, N, K, upper)[0] < Optimal_price:
        upper *= 2
    guess = (lower + upper) / 2
    for _ in range(100):
        fguess = Black_Cap_Pricing(tau_list, TWL_list, discount_list, forward_list, N, K, guess)[0] - Optimal_price
        if abs(fguess) < tolerance:
            break
        if fguess * flower < 0:
            upper = guess
        else:
            lower = guess
            flower = fguess
        guess = (lower + upper) / 2
    return guess


# Calculate implied volatilities

# forward_list = capdata['forward_rate'].values
# model_vols = np.zeros_like(tau_list)
# market_vols = np.zeros_like(tau_list)
#
# for i in range(len(tau_list)):
#     model_vols[i] = Price_to_Vol(caplet_NM[i], tau_list[:i + 1], reseting_date_list[:i + 1],
#                                  discount_list[:i + 1], forward_list[:i + 1], N_list[i], X_list[i])
#     market_vols[i] = Price_to_Vol(caplet_price[i], tau_list[:i + 1], reseting_date_list[:i + 1],
#                                   discount_list[:i + 1], forward_list[:i + 1], N_list[i], X_list[i])
#
# plt.figure(figsize=(10, 6))
# plt.plot(reseting_date_list, model_vols, label='Model Volatility')
# plt.plot(reseting_date_list, market_vols, label='Market Volatility')
# plt.xlabel('Reset Date')
# plt.ylabel('Volatility')
# plt.legend()
# plt.show()

# generate_short_rate()
