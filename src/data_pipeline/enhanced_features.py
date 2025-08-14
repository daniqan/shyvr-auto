"""
Enhanced technical indicator calculations for comprehensive feature extraction
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class EnhancedFeatureCalculator:
    """Calculate comprehensive technical indicators and market features"""
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return 0.0
        ema = prices.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return 0.0
        sma = prices.rolling(window=period).mean()
        return float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_adx(ohlcv_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        if len(ohlcv_data) < period + 1:
            return 0.0
        
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        close = ohlcv_data['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff() * -1
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # When both are positive, keep the larger one
        mask = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[mask & (plus_dm < minus_dm)] = 0
        minus_dm[mask & (minus_dm < plus_dm)] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_cci(ohlcv_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate Commodity Channel Index"""
        if len(ohlcv_data) < period:
            return 0.0
        
        typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma) / (0.015 * mad)
        
        return float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_stochastic(ohlcv_data: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, float]:
        """Calculate Stochastic Oscillator (K and D)"""
        if len(ohlcv_data) < period:
            return {'stoch_k': 0.0, 'stoch_d': 0.0}
        
        low_min = ohlcv_data['low'].rolling(window=period).min()
        high_max = ohlcv_data['high'].rolling(window=period).max()
        
        # Calculate %K
        k_percent = 100 * ((ohlcv_data['close'] - low_min) / (high_max - low_min))
        
        # Smooth %K to get %K
        stoch_k = k_percent.rolling(window=smooth_k).mean()
        
        # Calculate %D (SMA of %K)
        stoch_d = stoch_k.rolling(window=smooth_d).mean()
        
        return {
            'stoch_k': float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 0.0,
            'stoch_d': float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 0.0
        }
    
    @staticmethod
    def calculate_williams_r(ohlcv_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Williams %R"""
        if len(ohlcv_data) < period:
            return 0.0
        
        high_max = ohlcv_data['high'].rolling(window=period).max()
        low_min = ohlcv_data['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_max - ohlcv_data['close']) / (high_max - low_min))
        
        return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_obv(ohlcv_data: pd.DataFrame) -> float:
        """Calculate On-Balance Volume"""
        if len(ohlcv_data) < 2:
            return 0.0
        
        obv = [0]
        for i in range(1, len(ohlcv_data)):
            if ohlcv_data['close'].iloc[i] > ohlcv_data['close'].iloc[i-1]:
                obv.append(obv[-1] + ohlcv_data['volume'].iloc[i])
            elif ohlcv_data['close'].iloc[i] < ohlcv_data['close'].iloc[i-1]:
                obv.append(obv[-1] - ohlcv_data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return float(obv[-1])
    
    @staticmethod
    def calculate_cmf(ohlcv_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate Chaikin Money Flow"""
        if len(ohlcv_data) < period:
            return 0.0
        
        high = ohlcv_data['high']
        low = ohlcv_data['low']
        close = ohlcv_data['close']
        volume = ohlcv_data['volume']
        
        # Money Flow Multiplier
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # CMF
        cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_mfi(ohlcv_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index"""
        if len(ohlcv_data) < period + 1:
            return 0.0
        
        typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
        raw_money_flow = typical_price * ohlcv_data['volume']
        
        # Calculate positive and negative money flow
        money_flow_positive = pd.Series(0.0, index=ohlcv_data.index)
        money_flow_negative = pd.Series(0.0, index=ohlcv_data.index)
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                money_flow_positive.iloc[i] = raw_money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                money_flow_negative.iloc[i] = raw_money_flow.iloc[i]
        
        # Calculate money ratio
        positive_flow = money_flow_positive.rolling(window=period).sum()
        negative_flow = money_flow_negative.rolling(window=period).sum()
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        
        return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 0.0
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: Dict[str, int]) -> Dict[str, float]:
        """Calculate returns for different periods"""
        returns = {}
        
        for name, period in periods.items():
            if len(prices) > period:
                ret = (prices.iloc[-1] / prices.iloc[-period-1] - 1)
                returns[f'returns_{name}'] = float(ret) if not pd.isna(ret) else 0.0
            else:
                returns[f'returns_{name}'] = 0.0
        
        return returns
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, period: int = 24) -> float:
        """Calculate price volatility (standard deviation of returns)"""
        if len(prices) < period + 1:
            return 0.0
        
        returns = prices.pct_change().tail(period)
        volatility = returns.std()
        
        return float(volatility) if not pd.isna(volatility) else 0.0
    
    @staticmethod
    def calculate_price_changes(prices: pd.Series, periods: Dict[str, int]) -> Dict[str, float]:
        """Calculate price changes for different periods"""
        changes = {}
        
        for name, period in periods.items():
            if len(prices) > period:
                change = prices.iloc[-1] - prices.iloc[-period-1]
                changes[f'price_change_{name}'] = float(change) if not pd.isna(change) else 0.0
            else:
                changes[f'price_change_{name}'] = 0.0
        
        return changes
    
    def extract_all_features(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Extract all enhanced technical features"""
        features = {}
        
        if ohlcv_data.empty:
            return features
        
        close_prices = ohlcv_data['close']
        
        # Moving averages
        features['ema_12'] = self.calculate_ema(close_prices, 12)
        features['ema_26'] = self.calculate_ema(close_prices, 26)
        features['ema_50'] = self.calculate_ema(close_prices, 50)
        features['ema_200'] = self.calculate_ema(close_prices, 200)
        
        features['sma_20'] = self.calculate_sma(close_prices, 20)
        features['sma_50'] = self.calculate_sma(close_prices, 50)
        features['sma_200'] = self.calculate_sma(close_prices, 200)
        
        # Advanced indicators
        features['adx'] = self.calculate_adx(ohlcv_data)
        features['cci'] = self.calculate_cci(ohlcv_data)
        
        stoch = self.calculate_stochastic(ohlcv_data)
        features.update(stoch)
        
        features['williams_r'] = self.calculate_williams_r(ohlcv_data)
        
        # Volume indicators
        features['obv'] = self.calculate_obv(ohlcv_data)
        features['cmf'] = self.calculate_cmf(ohlcv_data)
        features['mfi'] = self.calculate_mfi(ohlcv_data)
        
        # Returns and volatility
        returns = self.calculate_returns(close_prices, {
            '1h': 1,
            '24h': 24,
            '7d': 168  # 7 * 24 hours
        })
        features.update(returns)
        
        features['volatility_24h'] = self.calculate_volatility(close_prices, 24)
        
        # Price changes
        changes = self.calculate_price_changes(close_prices, {
            '1h': 1,
            '24h': 24,
            '7d': 168
        })
        features.update(changes)
        
        return features