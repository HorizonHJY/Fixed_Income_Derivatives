import math
import numpy as np
import pandas as pd
from typing import Union, Callable
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.optimize import brentq

# Black formula for caplets/floorlets
def black_formula(K, F, v, is_call=True):
    d1 = (np.log(F / K) + 0.5 * v**2) / v
    d2 = d1 - v
    if is_call:
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)

# Black Cap Pricing
def Black_Cap_Pricing(file_path):
    cap_data = pd.read_excel(file_path)
    price = 0.0
    for idx, row in cap_data.iterrows():
        strike = row['CapStrike']/100
        P = row['Discount']           # P(0,Ti)
        F = row['ResetRate'] / 100    # Forward rate F(0,Ti-1,Ti)
        tau_i = row['tau_i']          # Day count fraction
        T = row['T_iM1']              # Time to maturity (T_{i-1})
        sigma = row['CapVols'] / 100  # Caplet volatility
        v = sigma * np.sqrt(T)        # v_i = σ_i√T_{i-1}
        bl = black_formula(strike, F, v)
        price += row['Notional'] * P * tau_i * bl
    return price

# Hull-White zero-coupon bond put
def hw_zbp(t, T, S, notional, strike_X, a, sigma, P_T, P_S):
    B = (1 - np.exp(-a * (S - T))) / a
    volatility = (sigma / a) * np.sqrt((1 - np.exp(-2 * a * (T - t))) / (2 * a)) * B
    d1 = (np.log(P_S / (strike_X * P_T)) + 0.5 * volatility**2) / volatility
    d2 = d1 - volatility
    return notional * (strike_X * P_T * norm.cdf(-d2) - P_S * norm.cdf(-d1))

# HW Caplet Pricing using ZBP
def HW_Caplets(file_path, a, sigma):
    """
    Price caplets using Hull-White model via Zero Bond Put (ZBP) method
    Args:
        file_path: Path to Excel file with market data
        a: Mean reversion rate for HW model
        sigma: Volatility parameter for HW model
    Returns:
        List of caplet prices
    """
    # Load and sort data by maturity
    cap_data = pd.read_excel(file_path).sort_values('T_i')
    prices = []

    for idx, row in cap_data.iterrows():
        # 1. Parse and validate parameters
        X = row['CapStrike'] / 100  # Convert strike from % to decimal
        tau_i = row['tau_i']  # Day count fraction
        N = row['Notional']  # Nominal amount
        ti_1 = row['T_iM1']  # Option expiry (T_{i-1})
        ti = row['T_i']  # Rate period end (T_i)
        P_T = row['Discount']  # Discount factor to T_{i-1}
        # 2. Get P_S = P(0,T_i) by matching T_i in data
        # Note: Assumes data is sorted and contains exact T_i matches
        P_S = cap_data[cap_data['T_i'] == ti]['Discount'].values[0]

        # 3. Calculate adjusted strike and notional
        X_prime_i = 1 / (1 + X * tau_i)  # Strike adjustment
        N_prime_i = N * (1 + X * tau_i)  # Notional adjustment

        # Debug print (can be commented out in production)
        # print(f"Row {idx}: X={X:.6f}, tau_i={tau_i:.6f}, X_prime_i={X_prime_i:.6f}, "
        #       f"P_T={P_T:.6f}, P_S={P_S:.6f}, ratio={P_S / (X_prime_i * P_T):.6f}")

        # 4. Calculate ZBP price (caplet price)
        zb_price = hw_zbp(
            t=0, T=ti_1, S=ti,
            notional=N_prime_i,
            strike_X=X_prime_i,
            a=a, sigma=sigma,
            P_T=P_T, P_S=P_S
        )
        prices.append(zb_price)

    return prices


def solve_volatility(objective, a=0.0001, max_b=100.0, max_iter=5):
    b = 1.0  # Initial upper bound (100%)
    for _ in range(max_iter):
        try:
            vol = brentq(objective, a=a, b=b, xtol=1e-6)
            return vol
        except ValueError:
            b *= 2  # Expand search interval
            if b > max_b:
                return np.nan
    return np.nan


def Price_to_Vol(hw_prices, file_path):
    cap_data = pd.read_excel(file_path)
    vols = []

    for idx, row in cap_data.iterrows():
        # 1. Parse market data
        F = row['ResetRate'] / 100  # Forward rate (decimal)
        K = row['CapStrike'] / 100  # Strike (decimal)
        T = max(row['T_iM1'], 1e-6)  # Time to expiry (avoid 0)
        P = row['Discount']  # Discount factor
        tau = row['tau_i']  # Day count fraction
        N = row['Notional']  # Nominal amount
        target_price = hw_prices[idx]  # HW model price

        # 2. Calculate theoretical price bounds
        min_price = N * P * tau * max(F - K, 0)  # Minimum (intrinsic value)
        max_price = N * P * tau * F  # Maximum (σ→∞ limit)

        # 3. Validate target price against bounds
        if target_price < min_price - 1e-6:  # Account for floating point errors
            vols.append(0.0)  # Volatility floored at 0%
            continue
        elif target_price > max_price + 1e-6:
            vols.append(np.inf)  # Volatility capped at infinity
            continue

        # 4. Define objective function for root finding
        def objective(sigma):
            """Calculate difference between Black price and target price"""
            v = sigma * np.sqrt(T)  # Total volatility
            bl_price = N * P * tau * black_formula(K, F, v)
            return bl_price - target_price

        # 5. Solve for implied volatility
        vol = solve_volatility(objective)
        # Convert to percentage and handle failures
        vols.append(vol * 100 if not np.isnan(vol) else np.nan)

    # sigmas = np.linspace(0, 2, 100)
    # objectives = [objective(s) for s in sigmas]
    # plt.plot(sigmas, objectives)
    # plt.axhline(0, color='r')
    # plt.xlabel('Volatility')
    # plt.ylabel('Objective')
    # plt.show()
    #
    # plt.plot(cap_data['T_i'], cap_data['Discount'], 'o-')
    # plt.xlabel('T_i')
    # plt.ylabel('Discount')
    # plt.show()

    return vols

def hw_cap(data_path, mr, vol):

    caplet_prices = HW_Caplets(data_path, mr, vol)
    cap_price = sum(caplet_prices)

    return cap_price


class HullWhiteAnalytical:
    """
    Hull-White model analytical solution for zero-coupon bond pricing.
    Supports both flat and market-based yield curves.
    """

    @staticmethod
    def B(t: float, T: float, a: float) -> float:
        """Compute B(t,T) term."""
        return (1 - np.exp(-a * (T - t))) / a
        # captures how sensitive the bond price is to the short rate
        # if a close to zero, then B(t,T) become T - t
        # if a is large, then rates mean revert quickly, and future rates are less impactful

    @staticmethod
    def A(t: float, T: float, a: float, sigma: float,
          P_M_0_T: float, P_M_0_t: float, f_M_0_t: float) -> float:
        """Compute A(t,T) adjustment factor."""
        B_t_T = HullWhiteAnalytical.B(t, T, a)
        exponent = B_t_T * f_M_0_t - (sigma ** 2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B_t_T ** 2
        return (P_M_0_T / P_M_0_t) * np.exp(exponent)
        # discounting adjustment to fit the current market price
        # P_M_0_T, P_M_0_t: Market zero-coupon bond prices at 0 for times T and t
        # f_M_0_t: Market instantaneous forward rate at time 0 for maturity t
        # exp(exponent): Adjustment for stochastic volatility of the short rate

    @staticmethod
    def bond_price(t: float, T: float, r_t: float, a: float, sigma: float,
                   P_M_0_T: Union[float, Callable],
                   P_M_0_t: Union[float, None] = None,
                   f_M_0_t: Union[float, None] = None) -> float:
        # t: Current time
        # T: Maturity time
        # r_t: Short rate at time t
        # a: Mean reversion speed
        # sigma: Volatility
        # P_M_0_T: Market bond price P(0,T), or function P(0,T)
        # P_M_0_t: Market bond price P(0,t) (if t > 0)
        # f_M_0_t: Forward rate f(0,t) (optional)
        # Returns: P(t,T): Zero-coupon bond price at time t

        if callable(P_M_0_T):
            P_M_0_T_val = P_M_0_T(T)
            P_M_0_t_val = P_M_0_T(t) if t > 0 else 1.0
            # Defaults P_M_0_t = 1.0 if t=0
            if f_M_0_t is None and t > 0:
                delta = min(1e-4, T - t)
                f_M_0_t_val = -(np.log(P_M_0_T(t + delta)) - np.log(P_M_0_T(t))) / delta
            else:
                f_M_0_t_val = f_M_0_t if f_M_0_t is not None else 0.03
        else:
            P_M_0_T_val = P_M_0_T
            P_M_0_t_val = 1.0 if t == 0 else P_M_0_t
            # Set default to 1.0 when t=0
            if f_M_0_t is None and t > 0 and P_M_0_t is not None:
                f_M_0_t_val = -(np.log(P_M_0_T_val) - np.log(P_M_0_t_val)) / (T - t)
            else:
                f_M_0_t_val = f_M_0_t if f_M_0_t is not None else 0.03
            # Estimates forward rates from two price inputs if not given

        A_t_T = HullWhiteAnalytical.A(t, T, a, sigma, P_M_0_T_val, P_M_0_t_val, f_M_0_t_val)
        B_t_T = HullWhiteAnalytical.B(t, T, a)
        return A_t_T * np.exp(-B_t_T * r_t)


def simulate_hull_white_paths(r0: float, a: float, sigma: float, T: float,
                              theta_func: callable, dt: float = 0.01,
                              n_paths: int = 100000) -> np.ndarray:
    # r0: Initial short rate
    # a: Mean reversion speed - how fast rates revert to equilibrium
    # sigma: Volatility of short rate
    # T: Maturity time
    # theta_func: Function theta(t) that returns the time-varying drift term
    # dt: Time step for simulation
    # n_paths: Number of paths (must be even for antithetics)
    # Returns: Array of short rate paths (n_paths x n_steps)

    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)  # Timeline from 0 to T
    r = np.zeros((n_paths, n_steps + 1))  # A matrix of shape to store the value of r(t) for each path
    r[:, 0] = r0

    # Antithetic variates
    Z = np.random.normal(0, 1, (n_paths // 2, n_steps))  # Simulate half the paths as normal random walks
    Z = np.vstack([Z, -Z])  # Mirror the shocks for the other half

    for i in range(1, n_steps + 1):
        dt_step = t[i] - t[i - 1]
        theta_t = theta_func(t[i - 1])  # Dynamic theta calculation

        # Exact solution for r(t)
        r[:, i] = (
                r[:, i - 1] * np.exp(-a * dt_step) +  # Mean reversion effect
                theta_t * (1 - np.exp(-a * dt_step)) +  # Long-term drift
                sigma * np.sqrt((1 - np.exp(-2 * a * dt_step)) / (2 * a)) * Z[:, i - 1]
        # Volatility term scaled by time and random noise
        )
    return r


def mc_bond_price(r0: float, a: float, sigma: float, T: float,
                  theta_func: callable = None, dt: float = 0.01,
                  n_paths: int = 100000) -> float:
    if theta_func is None:
        # Default theta for flat forward curve f(0,t) = r0
        theta_func = lambda t: r0 * a + (sigma ** 2 / (2 * a)) * (1 - np.exp(-2 * a * t))

    r_paths = simulate_hull_white_paths(r0, a, sigma, T, theta_func, dt, n_paths)
    integral_r = np.sum(r_paths[:, :-1] * dt, axis=1)
    return np.mean(np.exp(-integral_r))


# Example 1: Flat forward curve
def flat_curve_example():
    r0, a, sigma, T = 0.03, 0.1, 0.01, 5.0
    price = mc_bond_price(r0, a, sigma, T)
    print(f"Flat curve P(0,{T}): {price:.6f}")


# Example 2: Upward-sloping forward curve
def nonflat_curve_example():
    r0, a, sigma, T = 0.03, 0.1, 0.01, 5.0

    # Define arbitrary forward curve f(0,t) = 0.02 + 0.005*t
    times = np.linspace(0, T, 100)
    f_market = 0.02 + 0.005 * times
    f_func = interp1d(times, f_market, kind='linear', fill_value="extrapolate")

    # Calculate theta(t) for this curve
    df_dt = 0.005  # Derivative of our forward curve
    theta_func = lambda t: (df_dt + a * f_func(t) +
                            (sigma ** 2 / (2 * a)) * (1 - np.exp(-2 * a * t)))

    price = mc_bond_price(r0, a, sigma, T, theta_func)
    print(f"Non-flat curve P(0,{T}): {price:.6f}")

#2.4
def bond_put_hull_white(
    t, T, S, X, a, sigma, P_tS, P_tT
):
    """
    Hull-White (extended Vasicek) European put on a zero-coupon bond.

    Parameters
    ----------
    t : float
        'Current' time (often 0 in examples).
    T : float
        Option expiry time (T).
    S : float
        Underlying bond maturity (S > T).
    X : float
        Strike price (in terms of bond price).
    a : float
        Mean-reversion speed.
    sigma : float
        Volatility parameter (Hull-White).
    P_tS : float
        Price at time t of ZCB paying 1 at time S.
    P_tT : float
        Price at time t of ZCB paying 1 at time T.

    Returns
    -------
    put_price : float
        The Hull-White model put price on the bond.
    """

    # 1. Compute the HW "B(t,S)" factor
    #    B(t,S) = (1 - e^{-a(S - t)}) / a
    B_tS = (1.0 - math.exp(-a*(S - t))) / a

    # 2. The forward-bond "normal" stdev sigma_p
    #    sigma_p = sigma * sqrt( (1 - e^{-2a(T - t)}) / (2a) ) * B(t,S)
    var_factor = (1.0 - math.exp(-2.0*a*(T - t))) / (2.0*a)
    sigma_p = sigma * math.sqrt(var_factor) * B_tS

    # 3. The "h" value
    #    h = (1/sigma_p)*ln( P(t,S)/(X P(t,T)) ) + (sigma_p/2)
    if sigma_p < 1e-14:
        # Avoid numerical blow-up if sigma_p ~ 0
        # Then the put has zero volatility
        # Price is simply max(X * P_tT - P_tS, 0)
        return max(X*P_tT - P_tS, 0.0)

    h = (1.0 / sigma_p)*math.log( (P_tS)/(X*P_tT) ) + 0.5*sigma_p

    # 4. Apply the closed-form put formula
    #    ZBP = X * P(t,T)* Phi(-h + sigma_p) - P(t,S)* Phi(-h)
    put_price = X*P_tT*norm.cdf(-h + sigma_p) - P_tS*norm.cdf(-h)
    return put_price

#2.5
def HW_ZBPut_SM_Q(r0, a, sigma, T, S, X, dt=0.01, n_paths=100000):
    N = int(T / dt) + 1
    times = np.linspace(0, T, N)
    r = np.zeros((n_paths, N))
    r[:, 0] = r0
    theta = lambda t: r0 * a + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))

    for i in range(1, N):
        t = times[i - 1]
        theta_t = theta(t)
        Z = np.random.normal(0, 1, size=n_paths)
        r[:, i] = (
            r[:, i - 1] * np.exp(-a * dt)
            + theta_t * (1 - np.exp(-a * dt))
            + sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a)) * Z
        )
    rsum = np.sum(r[:, :-1] * dt, axis=1)
    r_T = r[:, -1]

    def B(t, T): return (1 - np.exp(-a * (T - t))) / a
    def A(t, T):
        tau = T - t
        return np.exp((sigma ** 2 / (2 * a ** 2)) * (tau - (1 - np.exp(-a * tau)) / a - 0.5 * (1 - np.exp(-a * tau)) ** 2 / a))

    PTS = A(T, S) * np.exp(-B(T, S) * r_T)
    payoff = np.maximum(X - PTS, 0)
    d = np.exp(-rsum) * payoff
    print(f"Mean of P(T,S): {np.mean(PTS)}")
    print(f"Min of P(T,S): {np.min(PTS)}")
    print(f"payoff: {payoff}")
    print(f"Strike: {X}")

    return np.mean(d)

#2.6
def HW_ZBPut_SM_T(r0, a, sigma, T, S, X, dt=0.01, n_paths=100000):
    N = int(T / dt) + 1
    times = np.linspace(0, T, N)
    r = np.zeros((n_paths, N))
    r[:, 0] = r0

    def B(t, T): return (1 - np.exp(-a * (T - t))) / a
    def A(t, T):
        tau = T - t
        return np.exp((sigma ** 2 / (2 * a ** 2)) * (tau - (1 - np.exp(-a * tau)) / a - 0.5 * (1 - np.exp(-a * tau)) ** 2 / a))
    for i in range(1, N):
        t = times[i - 1]
        theta_T = r0 * a + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t)) + sigma * B(t, T)
        Z = np.random.normal(0, 1, size=n_paths)
        r[:, i] = (
            r[:, i - 1] * np.exp(-a * dt)
            + theta_T * (1 - np.exp(-a * dt))
            + sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a)) * Z
        )

    rT = r[:, -1]
    PTS = A(T, S) * np.exp(-B(T, S) * rT)
    payoff = np.maximum(X - PTS, 0)
    P0T = A(0, T) * np.exp(-B(0, T) * r0)
    print(f"Mean of P(T,S): {np.mean(PTS)}")
    print(f"Min of P(T,S): {np.min(PTS)}")
    print(f"payoff: {payoff}")
    print(f"Strike: {X}")
    return P0T * np.mean(payoff)


# === Run the pricing ===
if __name__ == "__main__":

    #define variables and import data source
    file_path = 'Data/capdata.xlsx'
    a = 0.5      # mean reversion
    sigma = 0.01  # volatility
    # file_path = 'capdata.xlsx'

    # 1.1: Black model cap price
    black_price = Black_Cap_Pricing(file_path)
    print(f"Function 1.1: Black model total cap price: {black_price:.4f}")

    # 1.2: HW caplets

    hw_prices = HW_Caplets(file_path, a, sigma)
    print(f'Function 1.2:Caplets prices: {hw_prices}')

    # 1.3: Implied vol from HW prices
    implied_vols = Price_to_Vol(hw_prices, file_path)
    print("Function 1.3: Implied Black volatilities:", implied_vols)

    # 2.1: HW total cap price
    hw_cap_price = hw_cap(file_path, a, sigma)
    print(f"Function 2.1: Hull-White model total cap price: {hw_cap_price:.4f}")

    #2.2
    # Market inputs
    P1 = np.exp(-0.02 * 1.0)  # P(0,1)
    P2 = np.exp(-0.025 * 2.0)  # P(0,2)
    # Consistent forward rate between 1 and 2
    f = -(np.log(P2) - np.log(P1)) / (2.0 - 1.0)

    print("Consistent Non-flat Curve Example:")
    price = HullWhiteAnalytical.bond_price(
        t=1.0, T=2.0, r_t=0.03, a=a, sigma=sigma,
        P_M_0_T=P2, P_M_0_t=P1
    )
    print(f"P(1,2) = {price:.6f}")

    print("\nExplicit Forward Rate Example:")
    price_explicit = HullWhiteAnalytical.bond_price(
        t=1.0, T=2.0, r_t=0.03, a=0.1, sigma=0.01,
        P_M_0_T=P2, P_M_0_t=P1, f_M_0_t=f
    )
    print(f"P(1,2) = {price_explicit:.6f}")

    #2.3

    print("Flat Forward Curve")
    flat_curve_example()
    print()

    print("Upward-Sloping Forward Curve")
    nonflat_curve_example()

    #2.4
    t    = 0.0     # now
    T    = 1.0     # option expiry in 1 year
    S    = 2.0     # bond matures in 2 years
    X    = 0.95    # strike on the bond
    a    = 0.1     # mean-reversion
    sigma= 0.01    # volatility
    P_tS = 0.90    # ZCB price for maturity S=2
    P_tT = 0.97    # ZCB price for maturity T=1

    put_hw = bond_put_hull_white(t, T, S, X, a, sigma, P_tS, P_tT)
    print("Function 2.4: HW ZCB Put Price (Analytical sol) =", put_hw)

#2.5 2.6
    r0 = 0.03
    T = 1.0
    S = 2.0
    X = HullWhiteAnalytical.bond_price(t=0, T=2.0, r_t=0.03, a=a, sigma=0.03,P_M_0_T=P2, P_M_0_t=P1)

    zbp_price_q = HW_ZBPut_SM_Q(r0=r0, a=a, sigma=0.03, T=T, S=S, X=X)
    print(f"ZB Put Option Price under Q-measure: {zbp_price_q:.4f}")

    zbp_price_t = HW_ZBPut_SM_T(r0=r0, a=a, sigma=0.03, T=T, S=S, X=X)
    print(f"ZB Put Option Price under T-measure: {zbp_price_t:.4f}")




