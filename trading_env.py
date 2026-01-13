import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Optimized Trading Environment.
    Fixes:
    1. Correct Transaction Costs (Turnover based).
    2. Reward Scaling (For better Gradient Descent).
    3. Robust Indexing.
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, df_features, df_raw, spread=0.0002, downside_penalty=3.0, reward_scale=100.0):
        super(TradingEnv, self).__init__()
        
        assert len(df_features) == len(df_raw), "Data lengths must match"
        
        self.df_features = df_features
        self.df_raw = df_raw
        self.spread = spread
        self.downside_penalty = downside_penalty
        self.reward_scale = reward_scale # New Parameter
        
        # 0: Hold, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space
        self.observation_space = spaces.Box(
            low=-10.0, # Clip inputs at 10 std devs to prevent instability
            high=10.0, 
            shape=(df_features.shape[1],), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(df_features) - 2 # Safety buffer
        self.position = 0 
        self.balance_history = [10000.0]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance_history = [10000.0]
        return self._get_observation(), {}

    def step(self, action):
        truncated = False
        terminated = False
        
        # 1. Get Price Data
        current_price = self.df_raw.iloc[self.current_step]['Close']
        next_price = self.df_raw.iloc[self.current_step + 1]['Close']
        
        market_log_ret = np.log(next_price / current_price)
        
        # 2. Determine Position Change
        prev_position = self.position
        new_position = prev_position # Default
        
        if action == 1: new_position = 1    # Long
        elif action == 2: new_position = -1 # Short
        elif action == 0: new_position = prev_position # Hold
        
        # 3. Calculate Cost (Turnover Logic)
        # 0 -> 1 (Buy 1 unit) = Cost * 1
        # -1 -> 1 (Buy 2 units) = Cost * 2
        turnover = abs(new_position - prev_position)
        trade_cost = turnover * self.spread
        
        self.position = new_position
        
        # 4. Calculate Return
        # We assume immediate execution at current step, so we get the return of the NEW position
        gross_return = self.position * market_log_ret
        net_return = gross_return - trade_cost
        
        # 5. Reward Engineering (Sortino + Scaling)
        reward = 0
        if net_return < 0:
            penalty = self.downside_penalty * (net_return ** 2)
            reward = (net_return - penalty)
        else:
            reward = net_return
            
        # SCALE REWARD for PPO stability
        reward *= self.reward_scale

        # 6. Step Update
        self.current_step += 1
        
        # Balance Tracking (For Backtest/Human, not for Agent)
        last_balance = self.balance_history[-1]
        new_balance = last_balance * np.exp(net_return)
        self.balance_history.append(new_balance)
        
        # Termination
        if self.current_step >= self.max_steps:
            terminated = True
            
        info = {
            'net_return': net_return,
            'position': self.position,
            'price': next_price,
            'balance': new_balance
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        # Clip observation to prevent outliers exploding the Neural Net
        obs = self.df_features.iloc[self.current_step].values
        return np.clip(obs, -10, 10)