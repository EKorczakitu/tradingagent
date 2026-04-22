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
    def __init__(self, df_features, df_raw, spread=0.001):
        super(TradingEnv, self).__init__()
        
        self.features_data = df_features.values.astype(np.float32)
        self.close_prices = df_raw['Close'].values.astype(np.float32)
        self.open_prices = df_raw['Open'].values.astype(np.float32)

        self.timestamps = pd.to_datetime(df_raw.index)
        
        # --- LØSNING 2: Execution Mismatch ---
        # Calculate realistically executable returns: Enter at Open[t+1], Exit at Close[t+1]
        self.market_log_returns = np.zeros(len(self.close_prices), dtype=np.float32)
        # Ændret nævner fra close_prices[:-1] til open_prices[1:]
        self.market_log_returns[:-1] = np.log(self.close_prices[1:] / (self.open_prices[1:] + 1e-9))
        self.prices_data = self.close_prices
        
        # --- LØSNING 1: Future Data Leak i Volatilitet ---
        # Pre-calculate Volatility for Dynamic Slippage
        # Brug pct_change() i stedet for np.diff() for at sikre, at vi kun ser bagud i tid.
        raw_ret = pd.Series(self.prices_data).pct_change().fillna(0).values
        self.market_vol = pd.Series(raw_ret).rolling(20).std().fillna(0.0001).values

        self.max_steps = len(self.prices_data) - 1
        self.base_spread = spread
        
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
        
        # Aflæs klokkeslættet for det nuværende step
        current_hour = self.timestamps[self.current_step].hour
        
        # Formel for handelsomkostninger
        exec_cost = self.base_spread + (current_vol * 0.5)
        
        # Omkostninger for at indtage/justere positionen for denne time
        turnover = abs(target_position - prev_position)
        trade_cost = turnover * exec_cost
        
        # Bruttoafkast for den aktuelle time
        gross_return = target_position * market_log_ret
        
        # --- LØSNING 3: Intraday Exit Logik ---
        # Den danske børs lukker kl. 17.00. Vi tillader agenten at tjene penge på kl. 16-17 candle'en,
        # men i slutningen af timen tvinger vi den til at lukke positionen (og betale spread),
        # så den ikke holder positionen over natten.
        if current_hour >= 16:
            # Beregn omkostningen for at tvinge positionen tilbage til 0 ved lukketid
            forced_exit_turnover = abs(0 - target_position)
            forced_exit_cost = forced_exit_turnover * exec_cost
            trade_cost += forced_exit_cost
            
            # Næste start-position (næste dags morgen) bliver 0
            self.position = 0
        else:
            # Gemmer den nuværende position til næste time
            self.position = target_position

        # Nettoafkast
        net_return = gross_return - trade_cost
        
        # Update Balance
        current_balance = self.balance_history[-1]
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # --- ASYMMETRIC STEP REWARD ---
        if net_return < 0:
            reward = net_return * 1.1 * 100.0
        else:
            reward = net_return * 100.0
            
        reward = np.clip(reward, -10.0, 10.0)
        
        self.current_step += 1
        
        terminated = self.current_step >= self.max_steps
        
        info = {
            'net_return': net_return,
            'balance': new_balance,
            'position': self.position,
            'price': self.prices_data[self.current_step]
        }
        
        return self.features_data[self.current_step], reward, terminated, False, info