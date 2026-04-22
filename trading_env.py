import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Advanced Continuous Trading Environment
    Features:
    - Continuous Position Sizing (Box: -1.0 til 1.0)
    - Dynamic Slippage (Volatilitets-afhængig)
    - Differential Sharpe Ratio (DSR) Reward
    """
    def __init__(self, df_features, df_raw, spread=0.001, eta=0.01):
        super(TradingEnv, self).__init__()
        
        self.features_data = df_features.values.astype(np.float32)
        self.close_prices = df_raw['Close'].values.astype(np.float32)
        self.open_prices = df_raw['Open'].values.astype(np.float32)
        self.timestamps = pd.to_datetime(df_raw.index)
        
        self.market_log_returns = np.zeros(len(self.close_prices), dtype=np.float32)
        self.market_log_returns[:-1] = np.log(self.close_prices[1:] / (self.open_prices[1:] + 1e-9))
        self.prices_data = self.close_prices
        
        raw_ret = pd.Series(self.prices_data).pct_change().fillna(0).values
        self.market_vol = pd.Series(raw_ret).rolling(20).std().fillna(0.0001).values

        self.max_steps = len(self.prices_data) - 1
        self.base_spread = spread
        
        self.eta = eta  
        self.A = 0.0    
        self.B = 0.0    
        
        # SOTA: Kontinuerlig Action Space for Volatility Scaling
        # Action -1.0 = Max Short, 1.0 = Max Long, 0.0 = Cash
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.features_data.shape[1],), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.position = 0.0
        self.balance_history = [10000.0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.balance_history = [10000.0]
        self.A = 0.0
        self.B = 0.0 
        return self.features_data[self.current_step], {}

    def step(self, action):
        market_log_ret = self.market_log_returns[self.current_step]
        current_vol = self.market_vol[self.current_step]
        prev_position = self.position
        
        # SOTA: Float Target Position baseret på agentens output
        target_position = float(np.clip(action[0], -1.0, 1.0))
        current_hour = self.timestamps[self.current_step].hour
        
        exec_cost = self.base_spread + (current_vol * 0.5)
        
        if current_hour >= 16:
            target_position = 0.0 # Force flat
            
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * exec_cost
        
        # Position-scaled return
        gross_return = target_position * market_log_ret
        net_return = gross_return - trade_cost
        
        self.position = target_position
        
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # Differential Sharpe Ratio Update
        delta_A = self.eta * (net_return - self.A)
        delta_B = self.eta * (net_return**2 - self.B)
        variance = self.B - self.A**2
        
        if variance > 1e-8:
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / (variance**(1.5))
        else:
            dsr = 0.0
            
        self.A += delta_A
        self.B += delta_B
            
        # Mildere klipning for at bevare asymmetrien i store moves
        reward = np.clip(dsr, -15.0, 15.0) 
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'net_return': net_return,
            'balance': new_balance,
            'position': self.position,
            'price': self.prices_data[self.current_step],
            'dsr': dsr
        }
        
        return self.features_data[self.current_step], float(reward), terminated, False, info