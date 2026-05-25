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
        self.volume = np.maximum(df_raw['Volume'].values.astype(np.float32), 1e-5) # Undgår division med nul
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
        
        # Risk Management Trackers
        self.ath_balance = 10000.0
        self.current_day = None
        self.daily_start_balance = 10000.0
        self.intraday_stop_loss = -0.03 # 3% intra-day stop
        self.max_drawdown_limit = -0.15 # 15% absolut max drawdown

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.balance_history = [10000.0]
        self.ath_balance = 10000.0
        self.current_day = None
        self.daily_start_balance = 10000.0
        self.A = 0.0
        self.B = 0.0 
        return self.features_data[self.current_step], {}

    def step(self, action):
        market_log_ret = self.market_log_returns[self.current_step]
        current_vol = self.market_vol[self.current_step]
        prev_position = self.position
        
        # SOTA: Float Target Position baseret på agentens output
        # Robust håndtering af action (afhængig af om den kommer fra VecEnv eller ej)
        if isinstance(action, (np.ndarray, list)) and len(np.shape(action)) > 0:
            target_position = float(np.clip(action[0], -1.0, 1.0))
        else:
            target_position = float(np.clip(action, -1.0, 1.0))
        current_hour = self.timestamps[self.current_step].hour
        current_date = self.timestamps[self.current_step].date()
        current_balance = self.balance_history[-1]
        
        # Daglig nulstilling af tracker
        if self.current_day != current_date:
            self.current_day = current_date
            self.daily_start_balance = current_balance
            
        # Tjek Intra-day Stop Loss (før vi accepterer agentens target_position)
        daily_return = (current_balance / self.daily_start_balance) - 1.0
        if daily_return < self.intraday_stop_loss:
            target_position = 0.0 # Force flat for resten af dagen
        
        if current_hour >= 16:
            target_position = 0.0 # Force flat
            
        turnover = abs(target_position - prev_position)
        
        # SOTA Market Impact (Square-Root Law for Slippage)
        c_impact = 0.1
        market_impact = c_impact * current_vol * np.sqrt(turnover / self.volume[self.current_step])
        exec_cost = self.base_spread + market_impact
        trade_cost = turnover * exec_cost
        
        # Position-scaled return
        gross_return = target_position * market_log_ret
        net_return = gross_return - trade_cost
        
        # Thermodynamisk asymmetrisk return
        if net_return < 0:
            asym_return = net_return * 2.0
        else:
            asym_return = net_return
        
        self.position = target_position
        
        new_balance = current_balance * np.exp(net_return)
        self.balance_history.append(new_balance)
        self.ath_balance = max(self.ath_balance, new_balance)
        
        # Tjek Max Drawdown for at afbryde episoden
        drawdown = (new_balance / self.ath_balance) - 1.0
        truncated = bool(drawdown < self.max_drawdown_limit)

        # Differential Sharpe Ratio Update baseret på asymmetrisk return
        delta_A = self.eta * (asym_return - self.A)
        delta_B = self.eta * (asym_return**2 - self.B)
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
            'dsr': dsr,
            'drawdown': drawdown
        }
        
        return self.features_data[self.current_step], float(reward), terminated, truncated, info