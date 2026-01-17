import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Back to Basics: Discrete Trading Environment (The Golden Logic)
    """
    def __init__(self, df_features, df_raw, spread=0.0002):
        super(TradingEnv, self).__init__()
        
        self.features_data = df_features.values.astype(np.float32)
        self.prices_data = df_raw['Close'].values.astype(np.float32)
        
        self.market_log_returns = np.zeros(len(self.prices_data), dtype=np.float32)
        self.market_log_returns[:-1] = np.log(self.prices_data[1:] / (self.prices_data[:-1] + 1e-9))
        
        self.max_steps = len(self.prices_data) - 1
        self.spread = spread
        
        # --- TILBAGE TIL DISCRETE (Det der virkede!) ---
        # 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(self.features_data.shape[1],), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        # Hukommelse til din gamle Sortino-reward (den virkede jo!)
        self.returns_memory = [] 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        self.returns_memory = [] 
        return self.features_data[self.current_step], {}

    def step(self, action):
        market_log_ret = self.market_log_returns[self.current_step]
        prev_position = self.position
        
        # Mapping: 0->0, 1->1, 2->-1
        target_position = 0
        if action == 1: target_position = 1
        elif action == 2: target_position = -1
        
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * self.spread
        
        gross_return = target_position * market_log_ret
        net_return = gross_return - trade_cost
        
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # --- Den gamle Reward Function (Simpel men effektiv) ---
        self.returns_memory.append(net_return)
        if len(self.returns_memory) > 50: self.returns_memory.pop(0)
            
        recent_returns = np.array(self.returns_memory)
        risk = np.std(recent_returns) if len(recent_returns) > 1 else 0.01
        risk = max(risk, 0.0001)
        
        reward = (net_return / risk) * 100.0 # Reward Scaling
        reward = np.clip(reward, -10, 10)
        
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