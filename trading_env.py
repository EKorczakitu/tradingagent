import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Institutional Grade Trading Environment V2
    - Continuous Action Space (-1.0 to 1.0)
    - Risk-Adjusted Reward (No Rolling Sharpe instability)
    """
    def __init__(self, df_features, df_raw, spread=0.0002):
        super(TradingEnv, self).__init__()
        
        # Data Setup
        self.features_data = df_features.values.astype(np.float32)
        self.prices_data = df_raw['Close'].values.astype(np.float32)
        
        # Log returns for speed
        self.market_log_returns = np.zeros(len(self.prices_data), dtype=np.float32)
        self.market_log_returns[:-1] = np.log(self.prices_data[1:] / (self.prices_data[:-1] + 1e-9))
        
        self.max_steps = len(self.prices_data) - 1
        self.spread = spread
        
        # --- ÆNDRING 1: Continuous Action Space ---
        # Agenten outputter ét tal mellem -1 (Fuld Short) og 1 (Fuld Long)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space (Uændret)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(self.features_data.shape[1],), 
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.position = 0.0 # Nu en float!
        self.balance_history = [10000.0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.balance_history = [10000.0]
        return self.features_data[self.current_step], {}

    def step(self, action):
        # 1. Get Market Data
        market_log_ret = self.market_log_returns[self.current_step]
        
        # 2. Position Logic (Continuous)
        prev_position = self.position
        
        # Action kommer som et array fra modellen, f.eks. [0.75]
        # Vi clipper for en sikkerheds skyld
        target_position = float(np.clip(action[0], -1, 1))
        
        # 3. Cost Calculation
        # Vi betaler kun spread af ÆNDRINGEN i positionen
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * self.spread
        
        # 4. Return Calculation
        # Simulerer afkast baseret på positionen (prev_position holder vi gennem perioden)
        # Note: Dette er en simplificering. I live trading vil man gradvist skifte.
        gross_return = prev_position * market_log_ret
        net_return = gross_return - trade_cost
        
        # 5. Balance Update
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # --- ÆNDRING 2: Robust Reward Function ---
        # I stedet for Sharpe, bruger vi direkte risiko-straf.
        
        # Straf for volatilitet (Agenten lærer at undgå store sving)
        risk_penalty = 0.05 * (abs(net_return) ** 2) * 100 
        
        # Straf for at handle for meget (Transaction Cost awareness)
        cost_penalty = trade_cost * 10
        
        # Reward er afkast minus smerte
        # Vi ganger med 100 for at få tallet op i et område, det neurale netværk kan lide (f.eks. -1 til 1)
        reward = (net_return * 100) - risk_penalty - cost_penalty

        # Clip reward for at undgå eksplosioner (f.eks. ved vilde markedshop)
        reward = np.clip(reward, -5, 5)
        
        # 6. Next Step
        self.position = target_position
        self.current_step += 1
        
        # 7. Termination
        terminated = self.current_step >= self.max_steps
        current_price = self.prices_data[self.current_step]
        
        info = {
            'net_return': net_return,
            'balance': new_balance,
            'position': self.position,
            'price': current_price
        }
        
        return self.features_data[self.current_step], reward, terminated, False, info