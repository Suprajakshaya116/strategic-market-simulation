import numpy as np
import pandas as pd
from typing import Dict, Any, Union


def calculate_realized_volatility(prices: np.ndarray, window: int = 10) -> float:
    """Calculate annualized or step-based realized volatility from price series."""
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < 2:
        return 0.0
    returns = np.diff(np.log(np.maximum(prices, 1e-8)))
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns) * np.sqrt(252))


def calculate_effective_spread(execution_prices: np.ndarray, midpoints: np.ndarray) -> np.ndarray:
    """Calculate effective spread: 2 * |execution_price - midpoint| / midpoint."""
    exec_p = np.asarray(execution_prices, dtype=np.float64)
    mids = np.asarray(midpoints, dtype=np.float64)
    eps = 1e-8
    return 2.0 * np.abs(exec_p - mids) / (mids + eps)


def calculate_kyles_lambda(price_changes: np.ndarray, order_imbalances: np.ndarray) -> float:
    """Estimate Kyle's Lambda price impact coefficient: Delta P_t = lambda * OI_t + epsilon.

    A higher lambda indicates lower market depth / higher adverse selection.
    """
    dp = np.asarray(price_changes, dtype=np.float64)
    oi = np.asarray(order_imbalances, dtype=np.float64)
    if len(dp) < 3 or np.all(oi == oi[0]):
        return 0.0
    # OLS slope: Cov(dp, oi) / Var(oi)
    cov = np.cov(dp, oi)[0, 1]
    var_oi = np.var(oi)
    if var_oi < 1e-12:
        return 0.0
    return float(cov / var_oi)


def calculate_amihud_illiquidity(returns: np.ndarray, volumes: np.ndarray) -> float:
    """Calculate Amihud Illiquidity ratio: (1 / T) * sum( |R_t| / Volume_t ).

    Measures price response per unit of trading volume.
    """
    rets = np.asarray(returns, dtype=np.float64)
    vols = np.asarray(volumes, dtype=np.float64)
    valid_mask = vols > 0
    if not np.any(valid_mask):
        return 0.0
    ratio = np.abs(rets[valid_mask]) / vols[valid_mask]
    return float(np.mean(ratio))


def calculate_variance_ratio(prices: np.ndarray, lag: int = 2) -> float:
    """Calculate Lo-MacKinlay Variance Ratio: Var(r_k) / (k * Var(r_1)).

    Measures market informational efficiency (VR = 1.0 represents a random walk).
    """
    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) <= lag + 2:
        return 1.0
    log_p = np.log(np.maximum(prices, 1e-8))
    r1 = np.diff(log_p)
    var1 = np.var(r1)
    if var1 < 1e-12:
        return 1.0
    rk = log_p[lag:] - log_p[:-lag]
    vark = np.var(rk)
    return float(vark / (lag * var1))


def summarize_market_microstructure(market_data: Union[pd.DataFrame, Dict[str, list]]) -> Dict[str, float]:
    """Compute complete statistical microstructure summary for a market trajectory."""
    if isinstance(market_data, dict):
        df = pd.DataFrame(market_data)
    else:
        df = market_data.copy()

    prices = df["price"].values if "price" in df else np.array([])
    fundamentals = df["fundamental_value"].values if "fundamental_value" in df else prices
    spreads = df["spread"].values if "spread" in df else np.array([])
    liquidity = df["liquidity"].values if "liquidity" in df else np.array([])
    volumes = df["volume"].values if "volume" in df else np.array([])
    imbalances = df["order_imbalance"].values if "order_imbalance" in df else np.array([])

    dp = np.diff(prices) if len(prices) > 1 else np.array([0.0])
    rets = df["return"].values if "return" in df else (dp / np.maximum(prices[:-1], 1e-8) if len(prices) > 1 else np.array([0.0]))

    summary = {
        "mean_price": float(np.mean(prices)) if len(prices) > 0 else 0.0,
        "price_volatility": float(np.std(prices)) if len(prices) > 0 else 0.0,
        "realized_volatility": calculate_realized_volatility(prices),
        "mean_spread": float(np.mean(spreads)) if len(spreads) > 0 else 0.0,
        "max_spread": float(np.max(spreads)) if len(spreads) > 0 else 0.0,
        "mean_liquidity": float(np.mean(liquidity)) if len(liquidity) > 0 else 0.0,
        "min_liquidity": float(np.min(liquidity)) if len(liquidity) > 0 else 0.0,
        "mean_volume": float(np.mean(volumes)) if len(volumes) > 0 else 0.0,
        "mean_order_imbalance": float(np.mean(imbalances)) if len(imbalances) > 0 else 0.0,
        "kyles_lambda": calculate_kyles_lambda(dp, imbalances[:len(dp)]) if len(dp) > 1 else 0.0,
        "amihud_illiquidity": calculate_amihud_illiquidity(rets, volumes[:len(rets)]),
        "variance_ratio": calculate_variance_ratio(prices, lag=2),
    }

    if len(prices) > 0 and len(fundamentals) > 0:
        summary["fundamental_tracking_rmse"] = float(np.sqrt(np.mean((prices - fundamentals) ** 2)))

    return summary
