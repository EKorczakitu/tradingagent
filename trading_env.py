import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Institutional Grade Trading Environment
    - Fee Simulation
    - Sortino/Sharpe Reward
    - Position Management
    """
    def __init__(self, df_features, df_raw, spread=0.0002, downside_penalty=3.0, reward_scale=100.0):
        super(TradingEnv, self).__init__()
        
        # Data Setup
        self.features_data = df_features.values.astype(np.float32)
        self.prices_data = df_raw['Close'].values.astype(np.float32)
        
        # Pre-calculate market returns for speed
        self.market_log_returns = np.zeros(len(self.prices_data), dtype=np.float32)
        # Avoid division by zero with small epsilon, though Close should never be 0
        self.market_log_returns[:-1] = np.log(self.prices_data[1:] / (self.prices_data[:-1] + 1e-9))
        
        # --- FIX: Define max_steps ---
        self.max_steps = len(self.prices_data) - 1
        
        self.spread = spread
        self.reward_scale = reward_scale
        
        # Memory for Rolling Sharpe
        self.returns_memory = [] 
        
        # Action Space: 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: N features
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(self.features_data.shape[1],), 
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        self.returns_memory = [] 
        return self.features_data[self.current_step], {}

    def step(self, action):
        # 1. Get Market Data for NEXT step
        market_log_ret = self.market_log_returns[self.current_step]
        
        # 2. Position Logic
        prev_position = self.position
        target_position = prev_position
        
        if action == 1: target_position = 1   # Long
        elif action == 2: target_position = -1 # Short
        elif action == 0: target_position = prev_position # Hold
        
        # 3. Cost Calculation
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * self.spread
        
        # 4. Return Calculation
        gross_return = prev_position * market_log_ret
        net_return = gross_return - trade_cost
        
        # 5. Balance Update
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # 6. Reward Engineering (Sortino/Sharpe)
        self.returns_memory.append(net_return)
        if len(self.returns_memory) > 50: self.returns_memory.pop(0)
            
        recent_returns = np.array(self.returns_memory)
        downside_returns = recent_returns[recent_returns < 0]
        
        # Calculate risk (Downside Deviation)
        if len(downside_returns) > 0:
            risk = np.std(downside_returns)
        else:
            risk = 0.01 # Default baseline
            
        risk = max(risk, 0.0001) # Avoid zero division
        
        # Reward = Return / Risk
        reward = (net_return / risk) * self.reward_scale
        
        # Clip Reward for Stability
        reward = np.clip(reward, -10, 10)
        
        # 7. Next Step
        self.position = target_position
        self.current_step += 1
        
        # 8. Termination
        terminated = self.current_step >= self.max_steps
        current_price = self.prices_data[self.current_step]
        
        info = {
            'net_return': net_return,
            'balance': new_balance,
            'position': self.position,
            'price': current_price
        }
        
        return self.features_data[self.current_step], reward, terminated, False, info