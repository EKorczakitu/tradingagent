import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_metrics(returns_series, risk_free_rate=0.0):
    """
    Beregner en komplet suite af trading metrics.
    """
    # Sikr at vi arbejder med et array/series
    r = returns_series.dropna()
    if len(r) < 2:
        return {}

    # 1. Returns Metrics
    total_return = (1 + r).prod() - 1
    # Antag 1-times data: 252 handelsdage * 8 timer (eller 24 crypto)
    # Juster 'periods_per_year' afhængigt af din data (Crypto=365*24=8760, Aktier=252*7=1764)
    periods_per_year = 252 * 8  # Estimat for time-data på aktier
    
    avg_return = r.mean()
    cagr = (1 + total_return) ** (periods_per_year / len(r)) - 1 if len(r) > 0 else 0
    
    # 2. Volatility Metrics
    volatility = r.std() * np.sqrt(periods_per_year)
    
    # 3. Risk-Adjusted Return
    # Sharpe
    sharpe = (avg_return * periods_per_year - risk_free_rate) / (volatility + 1e-9)
    
    # Sortino (Kun downside volatility)
    negative_returns = r[r < 0]
    downside_std = negative_returns.std() * np.sqrt(periods_per_year)
    sortino = (avg_return * periods_per_year - risk_free_rate) / (downside_std + 1e-9)
    
    # 4. Drawdown
    cum_returns = (1 + r).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "Total Return": total_return,
        "CAGR (Annualized)": cagr,
        "Volatility (Ann.)": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar
    }

def print_report(metrics, title="Backtest Resultater"):
    print(f"\n{'='*10} {title} {'='*10}")
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 40)
    for k, v in metrics.items():
        if "Ratio" in k:
            fmt = "{:.2f}"
        else:
            fmt = "{:.2%}"
        print(f"{k:<25} | {fmt.format(v)}")
    print("-" * 40 + "\n")

def run_backtest_engine(env, model, title="Test Set"):
    """
    Kører agenten igennem miljøet og samler statistik.
    """
    obs, _ = env.reset()
    done = False
    
    returns_log = []
    actions_log = []
    portfolio_values = [env.balance_history[0]]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Gem data
        returns_log.append(info.get('net_return', 0))
        actions_log.append(info.get('position', 0))
        portfolio_values.append(info.get('balance', 10000))
        
        if done or truncated:
            break
            
    # Konverter til Series for nem beregning
    returns_series = pd.Series(returns_log)
    
    # Beregn Metrics
    metrics = calculate_metrics(returns_series)
    print_report(metrics, title=title)
    
    # Plot Cumulative Returns
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title(f'Equity Curve - {title}')
    plt.xlabel('Steps')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Gem plottet i stedet for at vise det (bedre til HPC)
    plt.savefig(f"backtest_{title.lower().replace(' ', '_')}.png")
    print(f"Plot gemt som backtest_{title.lower().replace(' ', '_')}.png")
    
    return metrics