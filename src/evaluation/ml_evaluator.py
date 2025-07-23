"""
ML-Enhanced Token Evaluator
Combines machine learning predictions with fundamental analysis for comprehensive token evaluation
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import structlog

from src.discovery.base import DiscoveredToken
from src.evaluation.base import (
    TokenEvaluatorBase, EvaluationResult, EvaluationStatus, RiskLevel,
    SecurityFlags, FundamentalMetrics
)
from src.ml_analysis.base import PredictionResult, PredictionDirection
from src.ml_analysis.model_manager import ModelManager


logger = structlog.get_logger()


class MLEnhancedEvaluator(TokenEvaluatorBase):
    """ML-enhanced token evaluator that combines ML predictions with fundamental analysis"""
    
    def __init__(self, model_manager: ModelManager, config: Optional[Dict] = None):
        self.model_manager = model_manager
        self.config = config or {}
        self.logger = structlog.get_logger().bind(evaluator=self.__class__.__name__)
        
        # ML prediction weights in overall evaluation
        self.ml_weight = self.config.get('ml_weight', 0.4)  # 40% ML, 60% fundamental
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.risk_adjustment_factor = self.config.get('risk_adjustment_factor', 1.2)
    
    async def evaluate_token(self, token: DiscoveredToken, 
                           historical_data: Optional[Dict] = None) -> EvaluationResult:
        """
        Evaluate a token using ML predictions and fundamental analysis
        
        Args:
            token: Token to evaluate
            historical_data: Optional historical price/volume data
            
        Returns:
            Comprehensive evaluation result
        """
        start_time = datetime.now()
        
        try:
            # Initialize evaluation result
            result = EvaluationResult(
                token=token,
                evaluated_at=start_time,
                status=EvaluationStatus.IN_PROGRESS
            )
            
            # Run ML analysis and fundamental analysis in parallel
            ml_task = self._run_ml_analysis(token, historical_data)
            fundamental_task = self._run_fundamental_analysis(token)
            
            ml_prediction, fundamental_metrics = await asyncio.gather(
                ml_task, fundamental_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(ml_prediction, Exception):
                self.logger.warning("ML analysis failed", 
                                  token=token.address, 
                                  error=str(ml_prediction))
                ml_prediction = None
            
            if isinstance(fundamental_metrics, Exception):
                self.logger.warning("Fundamental analysis failed",
                                  token=token.address,
                                  error=str(fundamental_metrics))
                fundamental_metrics = None
            
            # Combine analyses
            result = await self._combine_analyses(result, ml_prediction, fundamental_metrics)
            
            # Calculate final scores and recommendations
            result = self._calculate_final_evaluation(result)
            
            # Set completion status
            result.status = EvaluationStatus.COMPLETED
            result.evaluation_duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.info("Token evaluation completed",
                           token=token.address,
                           score=result.overall_score,
                           risk=result.overall_risk.value,
                           approved=result.is_approved,
                           action=result.recommended_action)
            
            return result
            
        except Exception as e:
            self.logger.error("Token evaluation failed", 
                            token=token.address, 
                            error=str(e))
            
            return EvaluationResult(
                token=token,
                evaluated_at=start_time,
                status=EvaluationStatus.FAILED,
                overall_risk=RiskLevel.VERY_HIGH,
                warnings=[f"Evaluation failed: {str(e)}"],
                evaluation_duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def _run_ml_analysis(self, token: DiscoveredToken, 
                             historical_data: Optional[Dict]) -> Optional[PredictionResult]:
        """Run ML prediction analysis"""
        try:
            # Check if model manager is healthy
            health_status = await self.model_manager.health_check()
            if not health_status.get('overall_healthy', False):
                self.logger.warning("Model manager not healthy", token=token.address)
                return None
            
            # Extract token-specific historical data
            token_data = None
            if historical_data and token.address in historical_data:
                token_data = historical_data[token.address]
            
            # Get ML prediction (preferably ensemble)
            prediction = await self.model_manager.analyze_token(
                token, 
                token_data, 
                use_ensemble=True
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error("ML analysis failed", token=token.address, error=str(e))
            return None
    
    async def _run_fundamental_analysis(self, token: DiscoveredToken) -> FundamentalMetrics:
        """Run fundamental analysis on token"""
        # Create basic fundamental metrics from token data
        metrics = FundamentalMetrics(
            price_usd=token.price_usd,
            price_change_24h=token.price_change_24h,
            market_cap=token.market_cap,
            volume_24h=token.volume_24h,
        )
        
        # Calculate volume to liquidity ratio if possible
        if token.volume_24h and token.market_cap:
            # Use market cap as liquidity proxy for now
            metrics.volume_to_liquidity_ratio = token.volume_24h / token.market_cap
        
        # Estimate holder metrics based on available data
        if token.market_cap:
            # Rough estimation based on market cap
            if token.market_cap > 10000000:  # >$10M
                metrics.holder_count = 1000
            elif token.market_cap > 1000000:  # >$1M
                metrics.holder_count = 500
            elif token.market_cap > 100000:  # >$100K
                metrics.holder_count = 100
            else:
                metrics.holder_count = 50
        
        # Set liquidity estimate
        if token.market_cap:
            # Estimate 10% of market cap as liquidity
            metrics.liquidity_usd = token.market_cap * 0.1
        
        return metrics
    
    async def _combine_analyses(self, result: EvaluationResult, 
                              ml_prediction: Optional[PredictionResult],
                              fundamental_metrics: Optional[FundamentalMetrics]) -> EvaluationResult:
        """Combine ML predictions with fundamental analysis"""
        
        # Set fundamental metrics
        result.fundamental_metrics = fundamental_metrics
        
        # Create basic security flags
        result.security_flags = SecurityFlags(
            contract_verified=True,  # Assume verified for now
            security_score=60.0  # Default moderate security score
        )
        
        # Store ML prediction in metadata
        if ml_prediction:
            result.metadata['ml_prediction'] = ml_prediction.to_dict()
            result.metadata['ml_confidence'] = ml_prediction.confidence
            result.metadata['ml_direction'] = ml_prediction.direction.value
            
            # Add ML-based notes
            if ml_prediction.is_buy_signal(self.confidence_threshold):
                result.notes.append(f"ML model suggests BUY with {ml_prediction.confidence:.1%} confidence")
            elif ml_prediction.direction in [PredictionDirection.SELL, PredictionDirection.STRONG_SELL]:
                result.notes.append(f"ML model suggests SELL with {ml_prediction.confidence:.1%} confidence")
            else:
                result.notes.append(f"ML model suggests HOLD with {ml_prediction.confidence:.1%} confidence")
            
            # Add prediction timeframes
            if ml_prediction.price_prediction_24h:
                current_price = result.token.price_usd or 0
                if current_price > 0:
                    expected_return = (ml_prediction.price_prediction_24h - current_price) / current_price
                    result.notes.append(f"24h price prediction: ${ml_prediction.price_prediction_24h:.6f} ({expected_return:+.1%})")
        
        return result
    
    def _calculate_final_evaluation(self, result: EvaluationResult) -> EvaluationResult:
        """Calculate final scores and recommendations"""
        
        # Calculate fundamental score
        fundamental_score = 0.0
        if result.fundamental_metrics:
            fundamental_score = result.fundamental_metrics.calculate_fundamental_score() or 0.0
        
        # Get ML score
        ml_score = 0.0
        ml_confidence = 0.0
        ml_prediction = result.metadata.get('ml_prediction')
        
        if ml_prediction:
            ml_confidence = ml_prediction.get('confidence', 0.0) or 0.0
            ml_direction = ml_prediction.get('direction', 'hold')
            
            # Convert ML prediction to score (0-100)
            if ml_direction in ['strong_buy']:
                ml_score = 90.0 * ml_confidence
            elif ml_direction == 'buy':
                ml_score = 75.0 * ml_confidence
            elif ml_direction == 'hold':
                ml_score = 50.0
            elif ml_direction == 'sell':
                ml_score = 25.0 * ml_confidence
            elif ml_direction == 'strong_sell':
                ml_score = 10.0 * ml_confidence
        
        # Ensure scores are valid numbers
        fundamental_score = fundamental_score or 0.0
        ml_score = ml_score or 0.0
        
        # Combine scores
        if ml_score > 0 and fundamental_score > 0:
            result.overall_score = (
                fundamental_score * (1 - self.ml_weight) + 
                ml_score * self.ml_weight
            )
        elif fundamental_score > 0:
            result.overall_score = fundamental_score
        elif ml_score > 0:
            result.overall_score = ml_score
        else:
            result.overall_score = 30.0  # Default low score
        
        # Calculate risk levels (ensure all return valid floats)
        result.security_risk = self._calculate_security_risk(result) or 50.0
        result.liquidity_risk = self._calculate_liquidity_risk(result) or 50.0
        result.volatility_risk = self._calculate_volatility_risk(result) or 50.0
        result.social_risk = self._calculate_social_risk(result) or 50.0
        
        # Set overall risk
        result.overall_risk = result.calculate_overall_risk()
        
        # Determine recommendation
        result.recommended_action, result.confidence_level = self._determine_recommendation(result)
        
        # Set approval status
        result.is_approved = result.should_approve()
        
        return result
    
    def _calculate_security_risk(self, result: EvaluationResult) -> float:
        """Calculate security risk score (0-100, higher = more risky)"""
        if not result.security_flags:
            return 80.0  # High risk if no security data
        
        risk = 0.0
        
        # Base security factors
        if result.security_flags.is_honeypot:
            risk += 50.0
        if result.security_flags.is_rugpull_risk:
            risk += 40.0
        if result.security_flags.has_mint_function:
            risk += 20.0
        if result.security_flags.has_pause_function:
            risk += 15.0
        if result.security_flags.has_blacklist_function:
            risk += 15.0
        
        # Positive factors (reduce risk)
        if result.security_flags.ownership_renounced:
            risk -= 15.0
        if result.security_flags.liquidity_locked:
            risk -= 20.0
        if result.security_flags.contract_verified:
            risk -= 10.0
        
        # Apply security score
        security_score_risk = (100.0 - result.security_flags.security_score) * 0.3
        risk += security_score_risk
        
        return max(0.0, min(risk, 100.0))
    
    def _calculate_liquidity_risk(self, result: EvaluationResult) -> float:
        """Calculate liquidity risk score"""
        if not result.fundamental_metrics or not result.fundamental_metrics.liquidity_usd:
            return 70.0  # High risk if no liquidity data
        
        liquidity = result.fundamental_metrics.liquidity_usd
        
        # Risk decreases with higher liquidity
        if liquidity > 1000000:  # >$1M
            return 10.0
        elif liquidity > 100000:  # >$100K
            return 30.0
        elif liquidity > 10000:  # >$10K
            return 50.0
        else:
            return 80.0
    
    def _calculate_volatility_risk(self, result: EvaluationResult) -> float:
        """Calculate volatility risk from ML predictions"""
        ml_prediction = result.metadata.get('ml_prediction')
        if not ml_prediction:
            return 50.0  # Default moderate risk
        
        # Use prediction uncertainty as volatility proxy
        uncertainty = ml_prediction.get('prediction_uncertainty', 0.5)
        volatility_forecast = ml_prediction.get('volatility_forecast', 0.5)
        
        # Higher uncertainty/volatility = higher risk
        risk = (uncertainty + volatility_forecast) * 50.0
        
        return max(10.0, min(risk, 90.0))
    
    def _calculate_social_risk(self, result: EvaluationResult) -> float:
        """Calculate social sentiment risk"""
        if not result.fundamental_metrics:
            return 60.0  # Default moderate-high risk
        
        social_score = result.fundamental_metrics.social_score or 0.0
        
        # Lower social score = higher risk
        risk = (1.0 - social_score / 100.0) * 60.0
        
        return max(20.0, min(risk, 80.0))
    
    def _determine_recommendation(self, result: EvaluationResult) -> tuple[str, float]:
        """Determine final recommendation and confidence"""
        
        score = result.overall_score or 0.0
        risk = result.overall_risk
        ml_prediction = result.metadata.get('ml_prediction')
        
        # Base confidence on score and risk alignment
        confidence = score * 0.8  # Base confidence from score
        
        # Adjust based on ML prediction if available
        if ml_prediction:
            ml_confidence = ml_prediction.get('confidence', 0.0) or 0.0
            ml_direction = ml_prediction.get('direction', 'hold')
            
            # If ML and fundamental analysis align, boost confidence
            if ((score >= 70 and ml_direction in ['buy', 'strong_buy']) or
                (score <= 40 and ml_direction in ['sell', 'strong_sell']) or
                (40 < score < 70 and ml_direction == 'hold')):
                confidence += ml_confidence * 20.0  # Boost confidence
            else:
                confidence -= 10.0  # Reduce confidence for conflicting signals
        
        # Determine action based on score and risk
        if score >= 75 and risk in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
            action = "BUY"
        elif score >= 60 and risk in [RiskLevel.VERY_LOW, RiskLevel.LOW, RiskLevel.MEDIUM]:
            action = "HOLD"
        elif score <= 30 or risk in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            action = "AVOID"
        elif score <= 45:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Override with ML signal if very confident
        if ml_prediction and ml_prediction.get('confidence', 0.0) > 0.8:
            ml_direction = ml_prediction.get('direction', 'hold')
            if ml_direction == 'strong_buy' and risk != RiskLevel.VERY_HIGH:
                action = "BUY"
                confidence = max(confidence, 75.0)
            elif ml_direction == 'strong_sell':
                action = "SELL"
                confidence = max(confidence, 75.0)
        
        confidence = max(30.0, min(confidence, 95.0))  # Clamp confidence
        
        return action, confidence
    
    async def batch_evaluate(self, tokens: List[DiscoveredToken], 
                           historical_data: Optional[Dict[str, Any]] = None) -> List[EvaluationResult]:
        """Evaluate multiple tokens in batch"""
        results = []
        
        # Process tokens in batches to avoid overwhelming the ML models
        batch_size = self.config.get('batch_size', 10)
        
        for i in range(0, len(tokens), batch_size):
            batch = tokens[i:i + batch_size]
            
            # Create evaluation tasks
            tasks = [
                self.evaluate_token(token, historical_data)
                for token in batch
            ]
            
            # Run batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions and collect results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error("Batch evaluation failed",
                                    token=batch[j].address,
                                    error=str(result))
                    # Create failed result
                    failed_result = EvaluationResult(
                        token=batch[j],
                        evaluated_at=datetime.now(),
                        status=EvaluationStatus.FAILED,
                        overall_risk=RiskLevel.VERY_HIGH,
                        warnings=[f"Evaluation failed: {str(result)}"]
                    )
                    results.append(failed_result)
                else:
                    results.append(result)
        
        self.logger.info("Batch evaluation completed",
                        total=len(tokens),
                        successful=len([r for r in results if r.status == EvaluationStatus.COMPLETED]),
                        failed=len([r for r in results if r.status == EvaluationStatus.FAILED]))
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check evaluator health"""
        try:
            # Check model manager health
            ml_health = await self.model_manager.health_check()
            
            return {
                "evaluator_healthy": True,
                "ml_models_healthy": ml_health.get('overall_healthy', False),
                "ml_ensemble_available": ml_health.get('ensemble_available', False),
                "config": {
                    "ml_weight": self.ml_weight,
                    "confidence_threshold": self.confidence_threshold,
                    "risk_adjustment_factor": self.risk_adjustment_factor
                },
                "model_performance": self.model_manager.get_model_performance()
            }
            
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "evaluator_healthy": False,
                "error": str(e)
            }


class EvaluationError(Exception):
    """Base exception for evaluation errors"""
    pass


class MLEvaluationError(EvaluationError):
    """Raised when ML evaluation fails"""
    pass


class FundamentalEvaluationError(EvaluationError):
    """Raised when fundamental evaluation fails"""
    pass