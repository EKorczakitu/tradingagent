import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Advanced Discrete Trading Environment
    Features:
    - Dynamic Slippage (Volatilitets-afhængig)
    - Differential Sharpe Ratio (DSR) Reward (Moody's online risk-adjustment)
    """
    def __init__(self, df_features, df_raw, spread=0.001, eta=0.01):
        super(TradingEnv, self).__init__()
        
        self.features_data = df_features.values.astype(np.float32)
        self.close_prices = df_raw['Close'].values.astype(np.float32)
        self.open_prices = df_raw['Open'].values.astype(np.float32)

        self.timestamps = pd.to_datetime(df_raw.index)
        
        # Calculate realistically executable returns: Enter at Open[t+1], Exit at Close[t+1]
        self.market_log_returns = np.zeros(len(self.close_prices), dtype=np.float32)
        self.market_log_returns[:-1] = np.log(self.close_prices[1:] / (self.open_prices[1:] + 1e-9))
        self.prices_data = self.close_prices
        
        # Pre-calculate Volatility for Dynamic Slippage (Undgår Data Leak)
        raw_ret = pd.Series(self.prices_data).pct_change().fillna(0).values
        self.market_vol = pd.Series(raw_ret).rolling(20).std().fillna(0.0001).values

        self.max_steps = len(self.prices_data) - 1
        self.base_spread = spread
        
        # --- DSR PARAMETRE ---
        self.eta = eta  # EMA decay rate for DSR
        self.A = 0.0    # EMA of returns
        self.B = 0.0    # EMA of squared returns
        
        # Actions: 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.features_data.shape[1],), 
            dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        
        # Nulstil DSR variabler ved hver ny episode
        self.A = 0.0
        self.B = 0.0 
        
        return self.features_data[self.current_step], {}

    def step(self, action):
        market_log_ret = self.market_log_returns[self.current_step]
        current_vol = self.market_vol[self.current_step]
        prev_position = self.position
        
        # Decode Action
        target_position = 0
        if action == 1: target_position = 1
        elif action == 2: target_position = -1
        
        current_hour = self.timestamps[self.current_step].hour
        
        # Dynamic Slippage Cost
        exec_cost = self.base_spread + (current_vol * 0.5)
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * exec_cost
        
        # Gross Return for current step
        gross_return = target_position * market_log_ret
        
        # Intraday Exit Logik (Gå flad over natten, betal spread)
        if current_hour >= 16:
            forced_exit_turnover = abs(0 - target_position)
            forced_exit_cost = forced_exit_turnover * exec_cost
            trade_cost += forced_exit_cost
            self.position = 0
        else:
            self.position = target_position

        # Nettoafkast
        net_return = gross_return - trade_cost
        
        # Update Balance
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # --- DIFFERENTIAL SHARPE RATIO (DSR) REWARD ---
        # 1. Udregn ændringen (Delta) baseret på det nye afkast
        delta_A = self.eta * (net_return - self.A)
        delta_B = self.eta * (net_return**2 - self.B)
        
        # 2. Beregn varians (Nævnerens indre del)
        variance = self.B - self.A**2
        
        # 3. Beregn selve DSR (Undgå division med nul eller negative rødder)
        if variance > 1e-8:
            # Moody's formel med 0.5 faktoren for Delta B
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / (variance**(1.5))
        else:
            # Fallback hvis modellen lige er startet eller variansen er 0
            dsr = 0.0
            
        # 4. Opdatér A og B til NÆSTE step
        self.A += delta_A
        self.B += delta_B
            
        # 5. Skaler og clip reward for at holde Neurale Netværk stabile
        # Vi ganger med et lille tal hvis DSR-værdierne bliver for voldsomme for PPO,
        # men DSR er normalt relativt velopdragen.
        reward = np.clip(dsr, -10.0, 10.0)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'net_return': net_return,
            'balance': new_balance,
            'position': self.position,
            'price': self.prices_data[self.current_step],
            'dsr': dsr # Praktisk til at debugge reward-signalet i callbacks
        }
        
        return self.features_data[self.current_step], float(reward), terminated, False, info