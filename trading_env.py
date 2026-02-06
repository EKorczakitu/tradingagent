import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Advanced Discrete Trading Environment
    Features:
    - Dynamic Slippage (Volatilitets-afhængig)
    - Sortino-baseret Reward (Downside risk focus)
    """
    def __init__(self, df_features, df_raw, spread=0.0002):
        super(TradingEnv, self).__init__()
        
        self.features_data = df_features.values.astype(np.float32)
        self.prices_data = df_raw['Close'].values.astype(np.float32)
        
        # Pre-calculate Log Returns
        self.market_log_returns = np.zeros(len(self.prices_data), dtype=np.float32)
        self.market_log_returns[:-1] = np.log(self.prices_data[1:] / (self.prices_data[:-1] + 1e-9))
        
        # Pre-calculate Volatility for Dynamic Slippage
        # (Vi bruger rullende std på raw returns som proxy for markedsuro)
        raw_ret = np.diff(self.prices_data) / (self.prices_data[:-1] + 1e-9)
        self.market_vol = np.zeros_like(self.prices_data)
        self.market_vol[:-1] = pd.Series(raw_ret).rolling(20).std().fillna(0.0001).values

        self.max_steps = len(self.prices_data) - 1
        self.base_spread = spread
        
        # Actions: 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(self.features_data.shape[1],), 
            dtype=np.float32
        )
        
        # State
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        self.returns_memory = [] # Til Sortino
        self.memory_len = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        self.returns_memory = [] 
        return self.features_data[self.current_step], {}

    def step(self, action):
        market_log_ret = self.market_log_returns[self.current_step]
        current_vol = self.market_vol[self.current_step]
        prev_position = self.position
        
        # Decode Action
        target_position = 0
        if action == 1: target_position = 1
        elif action == 2: target_position = -1
        
        # --- DYNAMIC SLIPPAGE ---
        # Hvis volatiliteten er høj, stiger spreadet (simulerer dårlig execution)
        # Formel: Base Spread + (Vol * 0.5)
        exec_cost = self.base_spread + (current_vol * 0.5)
        
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * exec_cost
        
        gross_return = target_position * market_log_ret
        net_return = gross_return - trade_cost
        
        # Update Balance
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # --- SORTINO REWARD ---
        self.returns_memory.append(net_return)
        if len(self.returns_memory) > self.memory_len: 
            self.returns_memory.pop(0)
            
        recent = np.array(self.returns_memory)
        
        # Find Downside Deviation (Negative afkast)
        neg_ret = recent[recent < 0]
        if len(neg_ret) > 1:
            downside_risk = np.std(neg_ret)
        else:
            downside_risk = 0.01 # Fallback
            
        risk = max(downside_risk, 0.0001)
        
        # Reward: Afkast divideret med downside risiko (Sortino-ish)
        # Multiplicer med 100 for at skalere til Neural Network venlige tal
        reward = (net_return / risk) * 100.0
        reward = np.clip(reward, -10, 10)
        
        self.position = target_position
        self.current_step += 1
        
        terminated = self.current_step >= self.max_steps
        
        info = {
            'net_return': net_return,
            'balance': new_balance,
            'position': self.position,
            'price': self.prices_data[self.current_step]
        }
        
        return self.features_data[self.current_step], reward, terminated, False, info