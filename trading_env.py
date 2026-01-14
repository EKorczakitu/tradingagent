import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Optimized High-Performance Trading Environment.
    
    Opdatering:
    - Nu 'Backtest-ready': Returnerer balance, price og position i info-dict.
    - Bevarer Numpy-hastigheden (ingen pandas .iloc i loopet).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df_features, df_raw, spread=0.0002, downside_penalty=3.0, reward_scale=100.0):
        super(TradingEnv, self).__init__()
        
        # 1. Convert to Numpy Arrays for speed
        self.features_data = df_features.values.astype(np.float32)
        self.prices_data = df_raw['Close'].values.astype(np.float32)
        
        # 2. Pre-calculate Market Returns
        self.market_log_returns = np.zeros(len(self.prices_data), dtype=np.float32)
        self.market_log_returns[:-1] = np.log(self.prices_data[1:] / self.prices_data[:-1])
        
        self.spread = spread
        self.downside_penalty = downside_penalty
        self.reward_scale = reward_scale
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(self.features_data.shape[1],), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(self.features_data) - 2
        self.position = 0
        self.balance_history = [10000.0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        return self.features_data[self.current_step], {}

    def step(self, action):
        # 1. Get Market Data (Instant numpy lookup)
        market_log_ret = self.market_log_returns[self.current_step]
        
        # 2. Position Logic
        prev_position = self.position
        target_position = prev_position
        
        if action == 1: target_position = 1    # Long
        elif action == 2: target_position = -1 # Short
        elif action == 0: target_position = prev_position # Hold
        
        # 3. Calculate Return (Based on PREVIOUS position - Execution Timing Fix)
        gross_return = prev_position * market_log_ret
        
        # 4. Calculate Cost
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * self.spread
        
        # 5. Net Result
        net_return = gross_return - trade_cost
        
        # 6. Update Balance (Genindført for Backtest)
        # Vi bruger simpel rentesrente formel: Balance * exp(log_return)
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)
        
        # 7. Reward Engineering
        reward = net_return
        if net_return < 0:
            reward = net_return - (self.downside_penalty * (net_return ** 2))
        
        reward *= self.reward_scale
        
        # 8. Update State
        self.position = target_position
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # 9. Info (Genindført 'price', 'balance', 'position' for backtest.py)
        # current_step er nu inkrementeret, så vi kigger på "næste" pris, hvilket matcher logikken
        current_price = self.prices_data[self.current_step]
        
        info = {
            'net_return': net_return,
            'balance': new_balance,
            'position': self.position,
            'price': current_price
        }
        
        return self.features_data[self.current_step], reward, terminated, False, info