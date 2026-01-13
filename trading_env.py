import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Et handelsmiljø til Reinforcement Learning, der følger Gymnasium-standarden.
    Dette miljø håndterer 'Dual-Data' input:
    1. df_features: Hvad agenten ser (Z-score skaleret).
    2. df_raw: Hvad miljøet bruger til at beregne PnL (Priser).
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, df_features, df_raw, spread=0.0002, downside_penalty=3.0):
        super(TradingEnv, self).__init__()
        
        # Data validering
        assert len(df_features) == len(df_raw), "Dataframes skal have samme længde!"
        # Vi sikrer os, at vi ikke har NaN værdier
        assert not df_features.isnull().values.any(), "Features må ikke indeholde NaN"
        
        self.df_features = df_features
        self.df_raw = df_raw
        self.spread = spread
        self.downside_penalty = downside_penalty # Lambda i formlen
        
        # Definition af Action Space (0: Hold, 1: Long, 2: Short)
        self.action_space = spaces.Discrete(3)
        
        # Definition af Observation Space (Antal features)
        # Vi bruger -inf til inf, da Z-scores teoretisk er ubegrænsede
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(df_features.shape[1],), 
            dtype=np.float32
        )
        
        # Tilstands-variabler
        self.current_step = 0
        self.max_steps = len(df_features) - 1
        self.position = 0 # 0: Neutral, 1: Long, -1: Short
        self.entry_price = 0.0
        
        # Tracking til statistik
        self.total_reward = 0.0
        self.trades = 0
        self.balance_history = [] # Simuleret balance for visualisering
        
    def reset(self, seed=None, options=None):
        """
        Nulstiller miljøet til start-tilstanden.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trades = 0
        self.balance_history = [10000.0] # Start med 10.000 (fiktiv valuta)
        
        # Returner første observation
        return self._get_observation(), {}

    def step(self, action):
        """
        Udfører én handling i miljøet.
        Action 0: Hold (Behold nuværende position: -1, 0, eller 1)
        Action 1: Gå Long (Hvis short -> luk og køb. Hvis neutral -> køb. Hvis long -> hold)
        Action 2: Gå Short (Hvis long -> luk og sælg. Hvis neutral -> sælg. Hvis short -> hold)
        """
        
        # 1. Hent nuværende markedsdata
        current_price = self.df_raw.iloc[self.current_step]['Close']
        # Vi kigger fremad for at beregne afkastet til NÆSTE step (Next Bar Return)
        # Det er dette afkast, agenten får for sin handling i dag.
        
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps - 1:
            terminated = True
            return self._get_observation(), 0, terminated, truncated, {}
            
        next_price = self.df_raw.iloc[self.current_step + 1]['Close']
        
        # Beregn Log Return af selve markedet (ln(P_t+1 / P_t))
        market_log_ret = np.log(next_price / current_price)
        
        # 2. Udfør handling og beregn omkostninger
        trade_cost = 0.0
        previous_position = self.position
        
        # Logikken for positions-skift
        if action == 1: # Ønske: Vær Long
            if self.position != 1:
                self.position = 1
                trade_cost = self.spread # Vi betaler spread for at gå ind
                self.trades += 1
                
        elif action == 2: # Ønske: Vær Short
            if self.position != -1:
                self.position = -1
                trade_cost = self.spread
                self.trades += 1
                
        elif action == 0: # Ønske: Hold nuværende
            pass # Ingen ændring, ingen omkostning
        
        # 3. Beregn Agentens Afkast (Strategy Return)
        # Hvis vi er Long (+1): Får vi market_log_ret
        # Hvis vi er Short (-1): Får vi -market_log_ret
        # Hvis vi er Neutral (0): Får vi 0
        
        # Vi bruger positionen fra STARTEN af steppet til at beregne afkastet, 
        # men hvis vi lige har handlet, har vi betalt omkostningen.
        # Bemærk: I RL simplificerer vi ofte ved at sige, at vi får afkastet af den NYE position.
        strategy_log_ret = self.position * market_log_ret
        
        # Træk omkostninger fra
        net_return = strategy_log_ret - trade_cost
        
        # 4. Sortino Reward Logic (Differential Downside Penalty)
        # R_t = LogReturn - (lambda * Downside^2)
        
        reward = 0
        if net_return < 0:
            # Hvis vi taber penge, straffes vi ekstra hårdt (kvadratisk)
            penalty = self.downside_penalty * (net_return ** 2)
            reward = net_return - penalty
        else:
            # Hvis vi tjener penge, får vi bare det rene afkast (ingen straf for upside volatility)
            reward = net_return

        # Opdater tracking
        self.total_reward += reward
        self.current_step += 1
        
        # Simuleret 'Equity Curve' (for sjov/grafik, bruges ikke til træning)
        last_balance = self.balance_history[-1]
        # Simpel tilnærmelse: Balance * e^(log_return)
        new_balance = last_balance * np.exp(net_return)
        self.balance_history.append(new_balance)

        # Check for slut
        if self.current_step >= self.max_steps - 1:
            terminated = True
            
        # Get info
        info = {
            'net_return': net_return,
            'position': self.position,
            'price': next_price,
            'balance': new_balance
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """Hjælpefunktion til at hente features for nuværende step."""
        return self.df_features.iloc[self.current_step].values

    def render(self, mode='human'):
        """Simpel print-funktion til debugging."""
        profit_pct = ((self.balance_history[-1] - 10000) / 10000) * 100
        print(f"Step: {self.current_step}, Price: {self.df_raw.iloc[self.current_step]['Close']:.2f}, "
              f"Pos: {self.position}, Balance: {self.balance_history[-1]:.2f} ({profit_pct:.2f}%)")