import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Optimized High-Performance Trading Environment.
    Changes:
    1. Converts Pandas -> Numpy in __init__ (100x faster indexing).
    2. Pre-calculates market returns (removes math from step loop).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df_features, df_raw, spread=0.0002, downside_penalty=3.0, reward_scale=100.0):
        super(TradingEnv, self).__init__()
        
        # 1. Convert to Numpy Arrays for speed (CRITICAL)
        # We drop the time index, we don't need it for the Agent, only for the human.
        self.features_data = df_features.values.astype(np.float32)
        self.prices_data = df_raw['Close'].values.astype(np.float32)
        
        # 2. Pre-calculate Market Returns (Vectorization)
        # We calculate log returns for the entire dataset at once
        # shift(-1) logic handled by numpy indexing
        self.market_log_returns = np.zeros(len(self.prices_data), dtype=np.float32)
        # log(P_t+1 / P_t)
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
        # --- OPTIMIZED STEP FUNCTION ---
        
        # 1. Get Pre-calculated Data (Instant access)
        market_log_ret = self.market_log_returns[self.current_step]
        
        # 2. Position Logic
        prev_position = self.position
        new_position = prev_position
        
        if action == 1: new_position = 1
        elif action == 2: new_position = -1
        elif action == 0: new_position = prev_position
        
        # 3. Cost Calculation
        turnover = abs(new_position - prev_position)
        trade_cost = turnover * self.spread
        
        self.position = new_position
        
        # 4. Return & Reward
        net_return = (self.position * market_log_ret) - trade_cost
        
        reward = net_return
        if net_return < 0:
            reward = net_return - (self.downside_penalty * (net_return ** 2))
        
        reward *= self.reward_scale

        # 5. Update
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Optional: Track balance (only math inside step)
        # new_balance = self.balance_history[-1] * np.exp(net_return) 
        # self.balance_history.append(new_balance)
        
        info = {'net_return': net_return} # Keep info minimal for speed
        
        return self.features_data[self.current_step], reward, terminated, False, info