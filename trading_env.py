import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df_features, df_raw, spread=0.0002, downside_penalty=3.0, reward_scale=100.0):
        super(TradingEnv, self).__init__()
        
        # ... (Existing numpy conversion code is fine) ...
        self.features_data = df_features.values.astype(np.float32)
        self.prices_data = df_raw['Close'].values.astype(np.float32)
        
        self.market_log_returns = np.zeros(len(self.prices_data), dtype=np.float32)
        self.market_log_returns[:-1] = np.log(self.prices_data[1:] / self.prices_data[:-1])
        
        self.spread = spread
        self.reward_scale = reward_scale
        
        # --- NEW: Memory for Rolling Sharpe Calculation ---
        self.returns_memory = [] 
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(self.features_data.shape[1],), 
            dtype=np.float32
        )
        # ... (Rest of init) ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        self.returns_memory = [] # Clear memory on reset
        return self.features_data[self.current_step], {}

    def step(self, action):
        # ... (Steps 1-6: Market Data, Position Logic, Costs remain the same) ...
        
        # 1. Get Market Data
        market_log_ret = self.market_log_returns[self.current_step]
        
        # 2. Position Logic
        prev_position = self.position
        target_position = prev_position
        if action == 1: target_position = 1
        elif action == 2: target_position = -1
        elif action == 0: target_position = prev_position
        
        # 3-6. Calculations
        gross_return = prev_position * market_log_ret
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * self.spread
        net_return = gross_return - trade_cost
        
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # --- 7. NEW: Rolling Sortino/Sharpe Reward ---
        # We store the return to calculate volatility over time
        self.returns_memory.append(net_return)
        
        # Keep only last 50 steps (approx 2 days of trading hours)
        if len(self.returns_memory) > 50:
            self.returns_memory.pop(0)
            
        recent_returns = np.array(self.returns_memory)
        
        # Calculate Risk (Downside Deviation is better for trading)
        # If returns are negative, that's "bad volatility". Positive spikes are "good".
        downside_returns = recent_returns[recent_returns < 0]
        
        if len(downside_returns) > 0:
            risk = np.std(downside_returns)
        else:
            risk = 0.01 # Default baseline risk if no losses yet

        # Prevent division by zero
        risk = max(risk, 0.0001)
        
        # Reward is Return scaled by Risk (Sharpe-like)
        # We add a small constant return to reward simply "surviving" without blowing up
        reward = (net_return / risk) * self.reward_scale
        
        # Prevents exploding gradients if risk is near-zero
        reward = np.clip(reward, -10, 10)
        # ... (Steps 8-9: Update State and Info remain the same) ...
        
        self.position = target_position
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        current_price = self.prices_data[self.current_step]
        
        info = {
            'net_return': net_return,
            'balance': new_balance,
            'position': self.position,
            'price': current_price
        }
        
        return self.features_data[self.current_step], reward, terminated, False, info