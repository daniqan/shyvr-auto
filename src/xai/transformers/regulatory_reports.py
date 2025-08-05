"""
Regulatory Compliance Reporter for Transformer-based Trading Systems.

This module provides comprehensive regulatory compliance reporting for transformer-based
trading decisions, supporting multiple jurisdictions including MiFID II, SEC, and FCA requirements.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import hashlib
from pathlib import Path
import uuid

from ..trading_integration import TradingExplanation
from .interpretable_trading_signals import (
    TradingSignalExplanation,
    AttentionBasedFeatureImportance,
    TemporalContribution,
    MarketEventAttribution
)

logger = logging.getLogger(__name__)


@dataclass
class RegulatoryReport:
    """Data structure for regulatory compliance reports."""
    
    report_id: str
    jurisdiction: str
    framework: str
    report_type: str
    generated_timestamp: datetime
    covered_period_start: datetime
    covered_period_end: datetime
    trading_decisions: List[str]  # Decision IDs
    summary_statistics: Dict[str, Any]
    compliance_status: str
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientFacingExplanation:
    """Client-friendly explanation of trading decisions."""
    
    decision_summary: str
    rationale: str
    risk_factors: List[str]
    confidence_level: str
    model_explanation: str
    regulatory_notices: List[str]
    timestamp: datetime
    language: str = "en"


class RegulatoryComplianceReporter:
    """
    Comprehensive regulatory compliance reporter for transformer-based trading systems.
    
    Supports multiple regulatory frameworks and provides audit trails, client-facing
    explanations, and archival systems for regulatory compliance.
    """
    
    def __init__(
        self,
        supported_jurisdictions: Optional[List[str]] = None,
        archive_path: Optional[str] = None,
        retention_years: int = 7
    ):
        """
        Initialize regulatory compliance reporter.
        
        Args:
            supported_jurisdictions: List of supported jurisdictions
            archive_path: Path for archiving reports
            retention_years: Years to retain reports
        """
        self.supported_jurisdictions = supported_jurisdictions or [
            'EU', 'US', 'UK', 'ESMA', 'SEC', 'FCA', 'CFTC'
        ]
        
        self.archive_path = Path(archive_path) if archive_path else Path('./regulatory_archive')
        self.retention_years = retention_years
        
        # Regulatory framework configurations
        self.framework_configs = self._load_framework_configurations()
        
        # Compliance templates and requirements
        self.compliance_templates = self._load_compliance_templates()
        
        # Report storage
        self.generated_reports: Dict[str, RegulatoryReport] = {}
        self.archived_reports: Dict[str, str] = {}  # report_id -> archive_path
        
        # Audit trail
        self.audit_log: List[Dict[str, Any]] = []
        
        # Client explanation templates
        self.client_templates = self._load_client_templates()
        
        # Ensure archive directory exists
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized RegulatoryComplianceReporter for jurisdictions: {self.supported_jurisdictions}")
    
    def generate_mifid_ii_report(
        self,
        trading_explanations: List[TradingExplanation],
        period_start: datetime,
        period_end: datetime,
        client_id: Optional[str] = None
    ) -> RegulatoryReport:
        """
        Generate MiFID II compliance report.
        
        Args:
            trading_explanations: List of trading explanations
            period_start: Report period start
            period_end: Report period end
            client_id: Optional client identifier
            
        Returns:
            MiFID II compliance report
        """
        try:
            report_id = self._generate_report_id('MIFID_II', period_start, period_end)
            
            # Filter explanations for the period
            filtered_explanations = self._filter_explanations_by_period(
                trading_explanations, period_start, period_end
            )
            
            # Generate MiFID II specific analysis
            mifid_analysis = self._analyze_mifid_compliance(filtered_explanations)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(filtered_explanations)
            
            # Determine compliance status
            compliance_status = self._assess_mifid_compliance(mifid_analysis, summary_stats)
            
            # Generate recommendations
            recommendations = self._generate_mifid_recommendations(mifid_analysis, compliance_status)
            
            # Create report
            report = RegulatoryReport(
                report_id=report_id,
                jurisdiction='EU',
                framework='MiFID_II',
                report_type='ALGORITHMIC_TRADING_TRANSPARENCY',
                generated_timestamp=datetime.now(),
                covered_period_start=period_start,
                covered_period_end=period_end,
                trading_decisions=[exp.decision_id for exp in filtered_explanations],
                summary_statistics=summary_stats,
                compliance_status=compliance_status,
                recommendations=recommendations,
                metadata={
                    'client_id': client_id,
                    'total_decisions': len(filtered_explanations),
                    'transformer_decisions': len([e for e in filtered_explanations if 'transformer' in e.model_type.lower()]),
                    'avg_confidence': summary_stats.get('average_confidence', 0.0),
                    'mifid_analysis': mifid_analysis
                }
            )
            
            # Store and archive report
            self.generated_reports[report_id] = report
            self._archive_report(report)
            self._log_audit_event('REPORT_GENERATED', {'report_id': report_id, 'framework': 'MiFID_II'})
            
            logger.info(f"Generated MiFID II report {report_id} covering {len(filtered_explanations)} decisions")
            return report
            
        except Exception as e:
            logger.error(f"Error generating MiFID II report: {str(e)}")
            raise
    
    def generate_sec_report(
        self,
        trading_explanations: List[TradingExplanation],
        period_start: datetime,
        period_end: datetime,
        firm_id: Optional[str] = None
    ) -> RegulatoryReport:
        """
        Generate SEC compliance report.
        
        Args:
            trading_explanations: List of trading explanations  
            period_start: Report period start
            period_end: Report period end
            firm_id: Optional firm identifier
            
        Returns:
            SEC compliance report
        """
        try:
            report_id = self._generate_report_id('SEC', period_start, period_end)
            
            # Filter explanations for the period
            filtered_explanations = self._filter_explanations_by_period(
                trading_explanations, period_start, period_end
            )
            
            # Generate SEC specific analysis
            sec_analysis = self._analyze_sec_compliance(filtered_explanations)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(filtered_explanations)
            
            # Determine compliance status
            compliance_status = self._assess_sec_compliance(sec_analysis, summary_stats)
            
            # Generate recommendations
            recommendations = self._generate_sec_recommendations(sec_analysis, compliance_status)
            
            # Create report
            report = RegulatoryReport(
                report_id=report_id,
                jurisdiction='US',
                framework='SEC',
                report_type='ALGORITHMIC_TRADING_CONTROLS',
                generated_timestamp=datetime.now(),
                covered_period_start=period_start,
                covered_period_end=period_end,
                trading_decisions=[exp.decision_id for exp in filtered_explanations],
                summary_statistics=summary_stats,
                compliance_status=compliance_status,
                recommendations=recommendations,
                metadata={
                    'firm_id': firm_id,
                    'total_decisions': len(filtered_explanations),
                    'sec_analysis': sec_analysis,
                    'risk_controls_assessed': True
                }
            )
            
            # Store and archive report
            self.generated_reports[report_id] = report
            self._archive_report(report)
            self._log_audit_event('REPORT_GENERATED', {'report_id': report_id, 'framework': 'SEC'})
            
            logger.info(f"Generated SEC report {report_id} covering {len(filtered_explanations)} decisions")
            return report
            
        except Exception as e:
            logger.error(f"Error generating SEC report: {str(e)}")
            raise
    
    def generate_fca_report(
        self,
        trading_explanations: List[TradingExplanation],
        period_start: datetime,
        period_end: datetime,
        firm_reference: Optional[str] = None
    ) -> RegulatoryReport:
        """
        Generate FCA compliance report.
        
        Args:
            trading_explanations: List of trading explanations
            period_start: Report period start
            period_end: Report period end
            firm_reference: Optional firm reference
            
        Returns:
            FCA compliance report
        """
        try:
            report_id = self._generate_report_id('FCA', period_start, period_end)
            
            # Filter explanations for the period
            filtered_explanations = self._filter_explanations_by_period(
                trading_explanations, period_start, period_end
            )
            
            # Generate FCA specific analysis
            fca_analysis = self._analyze_fca_compliance(filtered_explanations)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(filtered_explanations)
            
            # Determine compliance status
            compliance_status = self._assess_fca_compliance(fca_analysis, summary_stats)
            
            # Generate recommendations
            recommendations = self._generate_fca_recommendations(fca_analysis, compliance_status)
            
            # Create report
            report = RegulatoryReport(
                report_id=report_id,
                jurisdiction='UK',
                framework='FCA',
                report_type='ALGORITHMIC_TRADING_GOVERNANCE',
                generated_timestamp=datetime.now(),
                covered_period_start=period_start,
                covered_period_end=period_end,
                trading_decisions=[exp.decision_id for exp in filtered_explanations],
                summary_statistics=summary_stats,
                compliance_status=compliance_status,
                recommendations=recommendations,
                metadata={
                    'firm_reference': firm_reference,
                    'total_decisions': len(filtered_explanations),
                    'fca_analysis': fca_analysis,
                    'governance_framework_assessed': True
                }
            )
            
            # Store and archive report
            self.generated_reports[report_id] = report
            self._archive_report(report)
            self._log_audit_event('REPORT_GENERATED', {'report_id': report_id, 'framework': 'FCA'})
            
            logger.info(f"Generated FCA report {report_id} covering {len(filtered_explanations)} decisions")
            return report
            
        except Exception as e:
            logger.error(f"Error generating FCA report: {str(e)}")
            raise
    
    def generate_multi_jurisdictional_report(
        self,
        trading_explanations: List[TradingExplanation],
        period_start: datetime,
        period_end: datetime,
        jurisdictions: List[str]
    ) -> Dict[str, RegulatoryReport]:
        """
        Generate reports for multiple jurisdictions.
        
        Args:
            trading_explanations: List of trading explanations
            period_start: Report period start
            period_end: Report period end
            jurisdictions: List of jurisdictions to report for
            
        Returns:
            Dictionary mapping jurisdictions to reports
        """
        reports = {}
        
        for jurisdiction in jurisdictions:
            try:
                if jurisdiction.upper() in ['EU', 'MIFID_II']:
                    report = self.generate_mifid_ii_report(
                        trading_explanations, period_start, period_end
                    )
                    reports['MiFID_II'] = report
                elif jurisdiction.upper() in ['US', 'SEC']:
                    report = self.generate_sec_report(
                        trading_explanations, period_start, period_end
                    )
                    reports['SEC'] = report
                elif jurisdiction.upper() in ['UK', 'FCA']:
                    report = self.generate_fca_report(
                        trading_explanations, period_start, period_end
                    )
                    reports['FCA'] = report
                else:
                    logger.warning(f"Unsupported jurisdiction: {jurisdiction}")
                    
            except Exception as e:
                logger.error(f"Error generating report for {jurisdiction}: {str(e)}")
                continue
        
        return reports
    
    def generate_client_facing_explanation(
        self,
        trading_explanation: TradingExplanation,
        client_profile: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> ClientFacingExplanation:
        """
        Generate client-friendly explanation of trading decision.
        
        Args:
            trading_explanation: Trading explanation
            client_profile: Optional client profile for customization
            language: Language for explanation
            
        Returns:
            Client-facing explanation
        """
        try:
            # Determine client sophistication level
            sophistication_level = self._determine_client_sophistication(client_profile)
            
            # Generate decision summary
            decision_summary = self._generate_client_decision_summary(
                trading_explanation, sophistication_level
            )
            
            # Generate rationale
            rationale = self._generate_client_rationale(
                trading_explanation, sophistication_level
            )
            
            # Extract risk factors
            risk_factors = self._extract_client_risk_factors(trading_explanation)
            
            # Generate confidence level description
            confidence_level = self._generate_confidence_description(
                trading_explanation.confidence
            )
            
            # Generate model explanation
            model_explanation = self._generate_client_model_explanation(
                trading_explanation, sophistication_level
            )
            
            # Generate regulatory notices
            regulatory_notices = self._generate_regulatory_notices(
                trading_explanation, client_profile
            )
            
            return ClientFacingExplanation(
                decision_summary=decision_summary,
                rationale=rationale,
                risk_factors=risk_factors,
                confidence_level=confidence_level,
                model_explanation=model_explanation,
                regulatory_notices=regulatory_notices,
                timestamp=datetime.now(),
                language=language
            )
            
        except Exception as e:
            logger.error(f"Error generating client-facing explanation: {str(e)}")
            # Return fallback explanation
            return ClientFacingExplanation(
                decision_summary=f"Trading decision: {trading_explanation.decision_type} for {trading_explanation.symbol}",
                rationale="Detailed analysis unavailable due to processing error.",
                risk_factors=["Analysis incomplete - please consult with advisor"],
                confidence_level="Unavailable",
                model_explanation="Model explanation unavailable.",
                regulatory_notices=["This decision was generated by an automated system."],
                timestamp=datetime.now(),
                language=language
            )
    
    def create_audit_trail(
        self,
        decision_id: str,
        trading_explanation: TradingExplanation,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive audit trail for trading decision.
        
        Args:
            decision_id: Decision identifier
            trading_explanation: Trading explanation
            additional_metadata: Additional metadata
            
        Returns:
            Audit trail entry
        """
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'decision_id': decision_id,
            'timestamp': datetime.now().isoformat(),
            'model_type': trading_explanation.model_type,
            'decision_type': trading_explanation.decision_type,
            'symbol': trading_explanation.symbol,
            'confidence': trading_explanation.confidence,
            'explanation_type': trading_explanation.explanation_data.explanation_type,
            'feature_count': len(trading_explanation.explanation_data.feature_importance),
            'has_attention_data': trading_explanation.attention_data is not None,
            'metadata_keys': list(trading_explanation.metadata.keys()),
            'data_hash': self._calculate_explanation_hash(trading_explanation)
        }
        
        if additional_metadata:
            audit_entry.update(additional_metadata)
        
        # Store in audit log
        self.audit_log.append(audit_entry)
        
        # Archive if needed
        if len(self.audit_log) > 10000:  # Archive when log gets large
            self._archive_audit_log()
        
        return audit_entry
    
    def retrieve_report(self, report_id: str) -> Optional[RegulatoryReport]:
        """Retrieve regulatory report by ID."""
        if report_id in self.generated_reports:
            return self.generated_reports[report_id]
        
        # Try to load from archive
        if report_id in self.archived_reports:
            return self._load_archived_report(report_id)
        
        return None
    
    def search_reports(
        self,
        jurisdiction: Optional[str] = None,
        framework: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> List[RegulatoryReport]:
        """
        Search reports by criteria.
        
        Args:
            jurisdiction: Filter by jurisdiction
            framework: Filter by regulatory framework
            period_start: Filter by period start
            period_end: Filter by period end
            
        Returns:
            List of matching reports
        """
        matching_reports = []
        
        for report in self.generated_reports.values():
            if jurisdiction and report.jurisdiction != jurisdiction:
                continue
            if framework and report.framework != framework:
                continue
            if period_start and report.covered_period_start < period_start:
                continue
            if period_end and report.covered_period_end > period_end:
                continue
            
            matching_reports.append(report)
        
        return matching_reports
    
    def get_compliance_summary(
        self,
        jurisdiction: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get compliance summary for recent period.
        
        Args:
            jurisdiction: Filter by jurisdiction
            days_back: Number of days to look back
            
        Returns:
            Compliance summary
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        relevant_reports = [
            report for report in self.generated_reports.values()
            if (not jurisdiction or report.jurisdiction == jurisdiction)
            and report.generated_timestamp >= cutoff_date
        ]
        
        if not relevant_reports:
            return {
                'total_reports': 0,
                'compliance_rate': 0.0,
                'issues_identified': 0,
                'recommendations_pending': 0
            }
        
        compliant_reports = len([r for r in relevant_reports if r.compliance_status == 'COMPLIANT'])
        total_recommendations = sum(len(r.recommendations) for r in relevant_reports)
        
        return {
            'total_reports': len(relevant_reports),
            'compliance_rate': compliant_reports / len(relevant_reports),
            'issues_identified': len(relevant_reports) - compliant_reports,
            'recommendations_pending': total_recommendations,
            'jurisdictions_covered': list(set(r.jurisdiction for r in relevant_reports)),
            'latest_report_date': max(r.generated_timestamp for r in relevant_reports).isoformat()
        }
    
    # Helper methods
    
    def _load_framework_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Load regulatory framework configurations."""
        return {
            'MiFID_II': {
                'requires_explanation': True,
                'requires_risk_disclosure': True,
                'requires_model_transparency': True,
                'client_communication_required': True,
                'retention_years': 7,
                'reporting_frequency': 'quarterly'
            },
            'SEC': {
                'requires_explanation': True,
                'requires_risk_disclosure': True,
                'requires_model_transparency': False,
                'audit_trail_required': True,
                'retention_years': 5,
                'reporting_frequency': 'annual'
            },
            'FCA': {
                'requires_explanation': True,
                'requires_risk_disclosure': True,
                'requires_model_transparency': True,
                'governance_required': True,
                'retention_years': 6,
                'reporting_frequency': 'semi_annual'
            }
        }
    
    def _load_compliance_templates(self) -> Dict[str, Dict[str, str]]:
        """Load compliance report templates."""
        return {
            'MiFID_II': {
                'executive_summary': "This report provides MiFID II algorithmic trading transparency analysis for the covered period.",
                'methodology': "Analysis based on transformer attention patterns and feature importance calculations.",
                'conclusion': "The algorithmic trading system demonstrates compliance with MiFID II transparency requirements."
            },
            'SEC': {
                'executive_summary': "This report assesses SEC algorithmic trading controls and risk management.",
                'methodology': "Evaluation of trading decision explanations and risk control mechanisms.",
                'conclusion': "Trading controls are functioning within regulatory parameters."
            },
            'FCA': {
                'executive_summary': "This report evaluates FCA algorithmic trading governance and oversight.",
                'methodology': "Assessment of decision-making transparency and governance controls.",
                'conclusion': "Governance framework meets FCA algorithmic trading requirements."
            }
        }
    
    def _load_client_templates(self) -> Dict[str, Dict[str, str]]:
        """Load client explanation templates."""
        return {
            'retail': {
                'decision_format': "We {decision_type} {amount} of {symbol} based on our analysis.",
                'rationale_format': "Our automated system identified favorable market conditions.",
                'risk_notice': "All investments carry risk. Past performance does not guarantee future results."
            },
            'professional': {
                'decision_format': "Algorithm generated {decision_type} signal for {symbol} with {confidence} confidence.",
                'rationale_format': "Analysis based on {model_type} with attention-weighted feature importance.",
                'risk_notice': "Model-based decisions carry inherent uncertainty. Consider market conditions."
            },
            'institutional': {
                'decision_format': "Transformer model recommends {decision_type} position in {symbol}.",
                'rationale_format': "Decision based on attention analysis of {feature_count} market features.",
                'risk_notice': "Systematic risk management protocols apply. Review risk parameters."
            }
        }
    
    def _generate_report_id(self, framework: str, start_date: datetime, end_date: datetime) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        period_hash = hashlib.md5(f"{start_date.isoformat()}_{end_date.isoformat()}".encode()).hexdigest()[:8]
        return f"{framework}_{timestamp}_{period_hash}"
    
    def _filter_explanations_by_period(
        self,
        explanations: List[TradingExplanation],
        start_date: datetime,
        end_date: datetime
    ) -> List[TradingExplanation]:
        """Filter explanations by time period."""
        filtered = []
        for exp in explanations:
            try:
                exp_time = datetime.fromisoformat(exp.timestamp.replace('Z', '+00:00'))
                if start_date <= exp_time <= end_date:
                    filtered.append(exp)
            except Exception as e:
                logger.warning(f"Error parsing timestamp {exp.timestamp}: {str(e)}")
                continue
        
        return filtered
    
    def _calculate_summary_statistics(self, explanations: List[TradingExplanation]) -> Dict[str, Any]:
        """Calculate summary statistics for explanations."""
        if not explanations:
            return {}
        
        # Basic statistics
        total_decisions = len(explanations)
        buy_decisions = len([e for e in explanations if e.decision_type.upper() == 'BUY'])
        sell_decisions = len([e for e in explanations if e.decision_type.upper() == 'SELL'])
        hold_decisions = len([e for e in explanations if e.decision_type.upper() == 'HOLD'])
        
        # Confidence statistics
        confidences = [e.confidence for e in explanations]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        # Model type distribution
        model_types = {}
        for exp in explanations:
            model_type = exp.model_type
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        # Attention analysis availability
        with_attention = len([e for e in explanations if e.attention_data is not None])
        
        return {
            'total_decisions': total_decisions,
            'decision_distribution': {
                'buy': buy_decisions,
                'sell': sell_decisions,
                'hold': hold_decisions
            },
            'confidence_statistics': {
                'average': avg_confidence,
                'minimum': min_confidence,
                'maximum': max_confidence
            },
            'model_type_distribution': model_types,
            'attention_analysis_coverage': with_attention / total_decisions if total_decisions > 0 else 0.0,
            'symbols_traded': list(set(e.symbol for e in explanations))
        }
    
    def _analyze_mifid_compliance(self, explanations: List[TradingExplanation]) -> Dict[str, Any]:
        """Analyze MiFID II compliance requirements."""
        analysis = {
            'explanation_coverage': 0.0,
            'risk_disclosure_coverage': 0.0,
            'model_transparency_coverage': 0.0,
            'client_communication_ready': 0.0,
            'issues': []
        }
        
        if not explanations:
            return analysis
        
        total = len(explanations)
        
        # Check explanation coverage
        with_explanations = len([e for e in explanations if e.explanation_data is not None])
        analysis['explanation_coverage'] = with_explanations / total
        
        # Check risk disclosure
        with_risk_info = len([
            e for e in explanations 
            if 'risk_factors' in e.metadata or 'regulatory_compliance' in e.metadata
        ])
        analysis['risk_disclosure_coverage'] = with_risk_info / total
        
        # Check model transparency
        with_transparency = len([
            e for e in explanations
            if e.attention_data is not None or 'interpretable_signals' in e.metadata
        ])
        analysis['model_transparency_coverage'] = with_transparency / total
        
        # Check client communication readiness
        client_ready = len([
            e for e in explanations
            if e.confidence >= 0.5 and e.explanation_data is not None
        ])
        analysis['client_communication_ready'] = client_ready / total
        
        # Identify issues
        if analysis['explanation_coverage'] < 0.95:
            analysis['issues'].append('Insufficient explanation coverage')
        if analysis['model_transparency_coverage'] < 0.80:
            analysis['issues'].append('Limited model transparency')
        if analysis['risk_disclosure_coverage'] < 0.90:
            analysis['issues'].append('Inadequate risk disclosure')
        
        return analysis
    
    def _analyze_sec_compliance(self, explanations: List[TradingExplanation]) -> Dict[str, Any]:
        """Analyze SEC compliance requirements."""
        analysis = {
            'risk_control_coverage': 0.0,
            'audit_trail_coverage': 0.0,
            'explanation_availability': 0.0,
            'issues': []
        }
        
        if not explanations:
            return analysis
        
        total = len(explanations)
        
        # Check risk controls
        with_risk_controls = len([
            e for e in explanations
            if e.confidence < 1.0 and 'risk_factors' in e.metadata
        ])
        analysis['risk_control_coverage'] = with_risk_controls / total
        
        # Check audit trail
        with_audit_info = len([
            e for e in explanations
            if e.decision_id and e.timestamp and e.explanation_data
        ])
        analysis['audit_trail_coverage'] = with_audit_info / total
        
        # Check explanation availability
        with_explanations = len([
            e for e in explanations
            if e.explanation_data is not None
        ])
        analysis['explanation_availability'] = with_explanations / total
        
        # Identify issues
        if analysis['audit_trail_coverage'] < 0.98:
            analysis['issues'].append('Incomplete audit trail coverage')
        if analysis['risk_control_coverage'] < 0.85:
            analysis['issues'].append('Insufficient risk control documentation')
        
        return analysis
    
    def _analyze_fca_compliance(self, explanations: List[TradingExplanation]) -> Dict[str, Any]:
        """Analyze FCA compliance requirements."""
        analysis = {
            'governance_coverage': 0.0,
            'transparency_coverage': 0.0,
            'oversight_coverage': 0.0,
            'issues': []
        }
        
        if not explanations:
            return analysis
        
        total = len(explanations)
        
        # Check governance
        with_governance = len([
            e for e in explanations
            if e.confidence >= 0.3 and e.explanation_data is not None
        ])
        analysis['governance_coverage'] = with_governance / total
        
        # Check transparency
        with_transparency = len([
            e for e in explanations
            if e.attention_data is not None or 'model_transparency' in e.metadata
        ])
        analysis['transparency_coverage'] = with_transparency / total
        
        # Check oversight
        with_oversight = len([
            e for e in explanations
            if 'regulatory_compliance' in e.metadata
        ])
        analysis['oversight_coverage'] = with_oversight / total
        
        # Identify issues
        if analysis['governance_coverage'] < 0.90:
            analysis['issues'].append('Governance framework gaps')
        if analysis['transparency_coverage'] < 0.75:
            analysis['issues'].append('Limited transparency mechanisms')
        
        return analysis
    
    def _assess_mifid_compliance(
        self, 
        analysis: Dict[str, Any], 
        summary_stats: Dict[str, Any]
    ) -> str:
        """Assess overall MiFID II compliance status."""
        issues = analysis.get('issues', [])
        
        if not issues and all(
            analysis.get(key, 0) >= 0.85 
            for key in ['explanation_coverage', 'risk_disclosure_coverage', 'model_transparency_coverage']
        ):
            return 'COMPLIANT'
        elif len(issues) <= 1 and analysis.get('explanation_coverage', 0) >= 0.75:
            return 'MOSTLY_COMPLIANT'
        else:
            return 'NON_COMPLIANT'
    
    def _assess_sec_compliance(
        self,
        analysis: Dict[str, Any],
        summary_stats: Dict[str, Any]
    ) -> str:
        """Assess overall SEC compliance status."""
        issues = analysis.get('issues', [])
        
        if not issues and analysis.get('audit_trail_coverage', 0) >= 0.95:
            return 'COMPLIANT'
        elif len(issues) <= 1:
            return 'MOSTLY_COMPLIANT'
        else:
            return 'NON_COMPLIANT'
    
    def _assess_fca_compliance(
        self,
        analysis: Dict[str, Any],
        summary_stats: Dict[str, Any]
    ) -> str:
        """Assess overall FCA compliance status."""
        issues = analysis.get('issues', [])
        
        if not issues and analysis.get('governance_coverage', 0) >= 0.85:
            return 'COMPLIANT'
        elif len(issues) <= 1:
            return 'MOSTLY_COMPLIANT'
        else:
            return 'NON_COMPLIANT'
    
    def _generate_mifid_recommendations(
        self,
        analysis: Dict[str, Any],
        compliance_status: str
    ) -> List[str]:
        """Generate MiFID II recommendations."""
        recommendations = []
        
        if analysis.get('explanation_coverage', 0) < 0.95:
            recommendations.append("Improve explanation coverage for all trading decisions")
        
        if analysis.get('model_transparency_coverage', 0) < 0.80:
            recommendations.append("Enhance model transparency documentation")
        
        if analysis.get('risk_disclosure_coverage', 0) < 0.90:
            recommendations.append("Strengthen risk disclosure mechanisms")
        
        if compliance_status == 'NON_COMPLIANT':
            recommendations.append("Conduct comprehensive compliance review")
        
        return recommendations
    
    def _generate_sec_recommendations(
        self,
        analysis: Dict[str, Any],
        compliance_status: str
    ) -> List[str]:
        """Generate SEC recommendations."""
        recommendations = []
        
        if analysis.get('audit_trail_coverage', 0) < 0.98:
            recommendations.append("Strengthen audit trail documentation")
        
        if analysis.get('risk_control_coverage', 0) < 0.85:
            recommendations.append("Enhance risk control mechanisms")
        
        return recommendations
    
    def _generate_fca_recommendations(
        self,
        analysis: Dict[str, Any],
        compliance_status: str
    ) -> List[str]:
        """Generate FCA recommendations."""
        recommendations = []
        
        if analysis.get('governance_coverage', 0) < 0.90:
            recommendations.append("Strengthen governance framework")
        
        if analysis.get('transparency_coverage', 0) < 0.75:
            recommendations.append("Improve transparency mechanisms")
        
        return recommendations
    
    def _determine_client_sophistication(self, client_profile: Optional[Dict[str, Any]]) -> str:
        """Determine client sophistication level."""
        if not client_profile:
            return 'retail'
        
        # Simple heuristics for demonstration
        if client_profile.get('professional_client', False):
            return 'professional'
        elif client_profile.get('institutional_client', False):
            return 'institutional'
        else:
            return 'retail'
    
    def _generate_client_decision_summary(
        self,
        explanation: TradingExplanation,
        sophistication_level: str
    ) -> str:
        """Generate client-friendly decision summary."""
        template = self.client_templates[sophistication_level]['decision_format']
        
        return template.format(
            decision_type=explanation.decision_type.lower(),
            symbol=explanation.symbol,
            confidence=f"{explanation.confidence:.0%}"
        )
    
    def _generate_client_rationale(
        self,
        explanation: TradingExplanation,
        sophistication_level: str
    ) -> str:
        """Generate client-friendly rationale."""
        if sophistication_level == 'retail':
            return "Our analysis indicates favorable market conditions for this trade."
        elif sophistication_level == 'professional':
            return f"Model analysis with {explanation.confidence:.0%} confidence based on market indicators."
        else:  # institutional
            feature_count = len(explanation.explanation_data.feature_importance) if explanation.explanation_data else 0
            return f"Transformer analysis of {feature_count} features with attention-based importance weighting."
    
    def _extract_client_risk_factors(self, explanation: TradingExplanation) -> List[str]:
        """Extract client-appropriate risk factors."""
        base_risks = [
            "Market volatility may affect returns",
            "Past performance does not guarantee future results",
            "All trading involves risk of loss"
        ]
        
        # Add specific risks from metadata
        if 'risk_factors' in explanation.metadata:
            specific_risks = explanation.metadata['risk_factors'][:2]  # Limit to 2 additional
            base_risks.extend(specific_risks)
        
        return base_risks
    
    def _generate_confidence_description(self, confidence: float) -> str:
        """Generate confidence level description."""
        if confidence >= 0.8:
            return "High confidence"
        elif confidence >= 0.6:
            return "Moderate confidence"
        elif confidence >= 0.4:
            return "Low confidence"
        else:
            return "Very low confidence"
    
    def _generate_client_model_explanation(
        self,
        explanation: TradingExplanation,
        sophistication_level: str
    ) -> str:
        """Generate client-appropriate model explanation."""
        if sophistication_level == 'retail':
            return "This decision was made by our automated trading system using advanced market analysis."
        elif sophistication_level == 'professional':
            return f"Decision generated by {explanation.model_type} model with explainable AI analysis."
        else:  # institutional
            attention_info = "with attention analysis" if explanation.attention_data else "without attention data"
            return f"Transformer-based model ({explanation.model_type}) {attention_info} and feature importance ranking."
    
    def _generate_regulatory_notices(
        self,
        explanation: TradingExplanation,
        client_profile: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate regulatory notices."""
        notices = [
            "This decision was generated by an automated algorithmic trading system.",
            "The system operates under regulatory oversight and compliance frameworks."
        ]
        
        if explanation.confidence < 0.5:
            notices.append("Low confidence decisions require additional consideration.")
        
        if client_profile and client_profile.get('retail_client', True):
            notices.append("Retail client protections apply to this trading activity.")
        
        return notices
    
    def _archive_report(self, report: RegulatoryReport) -> None:
        """Archive regulatory report."""
        try:
            # Create archive directory structure
            year_dir = self.archive_path / str(report.generated_timestamp.year)
            month_dir = year_dir / f"{report.generated_timestamp.month:02d}"
            month_dir.mkdir(parents=True, exist_ok=True)
            
            # Archive file path
            archive_file = month_dir / f"{report.report_id}.json"
            
            # Convert report to dictionary and save
            report_dict = {
                'report_id': report.report_id,
                'jurisdiction': report.jurisdiction,
                'framework': report.framework,
                'report_type': report.report_type,
                'generated_timestamp': report.generated_timestamp.isoformat(),
                'covered_period_start': report.covered_period_start.isoformat(),
                'covered_period_end': report.covered_period_end.isoformat(),
                'trading_decisions': report.trading_decisions,
                'summary_statistics': report.summary_statistics,
                'compliance_status': report.compliance_status,
                'recommendations': report.recommendations,
                'metadata': report.metadata
            }
            
            with open(archive_file, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            # Update archive index
            self.archived_reports[report.report_id] = str(archive_file)
            
            logger.info(f"Archived report {report.report_id} to {archive_file}")
            
        except Exception as e:
            logger.error(f"Error archiving report {report.report_id}: {str(e)}")
    
    def _load_archived_report(self, report_id: str) -> Optional[RegulatoryReport]:
        """Load report from archive."""
        try:
            archive_path = self.archived_reports.get(report_id)
            if not archive_path or not Path(archive_path).exists():
                return None
            
            with open(archive_path, 'r') as f:
                report_dict = json.load(f)
            
            # Convert back to RegulatoryReport
            return RegulatoryReport(
                report_id=report_dict['report_id'],
                jurisdiction=report_dict['jurisdiction'],
                framework=report_dict['framework'],
                report_type=report_dict['report_type'],
                generated_timestamp=datetime.fromisoformat(report_dict['generated_timestamp']),
                covered_period_start=datetime.fromisoformat(report_dict['covered_period_start']),
                covered_period_end=datetime.fromisoformat(report_dict['covered_period_end']),
                trading_decisions=report_dict['trading_decisions'],
                summary_statistics=report_dict['summary_statistics'],
                compliance_status=report_dict['compliance_status'],
                recommendations=report_dict['recommendations'],
                metadata=report_dict['metadata']
            )
            
        except Exception as e:
            logger.error(f"Error loading archived report {report_id}: {str(e)}")
            return None
    
    def _log_audit_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        """Log audit event."""
        audit_event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        self.audit_log.append(audit_event)
        logger.debug(f"Logged audit event: {event_type}")
    
    def _calculate_explanation_hash(self, explanation: TradingExplanation) -> str:
        """Calculate hash of explanation for audit purposes."""
        # Create deterministic representation
        data_str = f"{explanation.decision_id}_{explanation.timestamp}_{explanation.decision_type}_{explanation.symbol}_{explanation.confidence}"
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _archive_audit_log(self) -> None:
        """Archive audit log when it gets too large."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_file = self.archive_path / f"audit_log_{timestamp}.json"
            
            with open(archive_file, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
            
            # Keep only recent entries
            self.audit_log = self.audit_log[-1000:]  # Keep last 1000 entries
            
            logger.info(f"Archived audit log to {archive_file}")
            
        except Exception as e:
            logger.error(f"Error archiving audit log: {str(e)}")