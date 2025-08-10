"""
GDPR (General Data Protection Regulation) compliance implementation.
Provides comprehensive GDPR compliance features including data sovereignty,
consent management, and data subject rights.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from . import (
    ComplianceManager, ComplianceFramework, DataCategory, 
    ProcessingPurpose, LegalBasis, ConsentRecord, DataSubjectRequest
)

logger = logging.getLogger(__name__)

class GDPRDataSubjectRights(Enum):
    """GDPR Data Subject Rights."""
    ACCESS = "access"                    # Article 15
    RECTIFICATION = "rectification"      # Article 16
    ERASURE = "erasure"                  # Article 17 (Right to be forgotten)
    RESTRICT_PROCESSING = "restrict"     # Article 18
    DATA_PORTABILITY = "portability"     # Article 20
    OBJECT_PROCESSING = "objection"      # Article 21
    AUTOMATED_DECISION = "automated"     # Article 22

class GDPRLawfulBasis(Enum):
    """GDPR Lawful basis for processing (Article 6)."""
    CONSENT = "consent"                  # Article 6(1)(a)
    CONTRACT = "contract"                # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation" # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"   # Article 6(1)(d)
    PUBLIC_TASK = "public_task"          # Article 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests" # Article 6(1)(f)

@dataclass
class GDPRProcessingActivity:
    """GDPR Processing Activity Record (Article 30)."""
    id: str
    controller_name: str
    controller_contact: str
    dpo_contact: Optional[str]
    purposes: List[str]
    data_categories: List[str]
    data_subjects: List[str]
    recipients: List[str]
    third_country_transfers: List[str]
    safeguards: List[str]
    retention_period: str
    security_measures: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class GDPRDataBreach:
    """GDPR Data Breach Record (Article 33)."""
    id: str
    description: str
    categories_affected: List[DataCategory]
    approximate_subjects: int
    likely_consequences: str
    measures_taken: str
    measures_proposed: str
    detected_at: datetime
    reported_at: Optional[datetime] = None
    authority_notified: bool = False
    subjects_notified: bool = False
    high_risk: bool = False
    resolved: bool = False

class GDPRComplianceManager(ComplianceManager):
    """
    GDPR-specific compliance manager implementing all GDPR requirements.
    Extends the base ComplianceManager with GDPR-specific functionality.
    """
    
    def __init__(self, region: str, controller_name: str, dpo_contact: str):
        super().__init__(ComplianceFramework.GDPR, region)
        self.controller_name = controller_name
        self.dpo_contact = dpo_contact
        self.processing_activities: Dict[str, GDPRProcessingActivity] = {}
        self.data_breaches: Dict[str, GDPRDataBreach] = {}
        self.privacy_notices: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized GDPR compliance manager for {controller_name} in {region}")
    
    def register_processing_activity(
        self,
        purpose: str,
        data_categories: List[DataCategory],
        data_subjects: List[str],
        legal_basis: GDPRLawfulBasis,
        recipients: Optional[List[str]] = None,
        third_country_transfers: Optional[List[str]] = None,
        retention_period: str = "As long as necessary for the purpose",
        security_measures: Optional[List[str]] = None
    ) -> GDPRProcessingActivity:
        """Register a processing activity under GDPR Article 30."""
        activity_id = str(uuid.uuid4())
        
        activity = GDPRProcessingActivity(
            id=activity_id,
            controller_name=self.controller_name,
            controller_contact=f"privacy@{self.controller_name.lower().replace(' ', '-')}.com",
            dpo_contact=self.dpo_contact,
            purposes=[purpose],
            data_categories=[cat.value for cat in data_categories],
            data_subjects=data_subjects,
            recipients=recipients or [],
            third_country_transfers=third_country_transfers or [],
            safeguards=self._get_transfer_safeguards(third_country_transfers or []),
            retention_period=retention_period,
            security_measures=security_measures or self._get_default_security_measures()
        )
        
        self.processing_activities[activity_id] = activity
        
        logger.info(f"Registered GDPR processing activity {activity_id} for purpose: {purpose}")
        return activity
    
    def _get_transfer_safeguards(self, transfers: List[str]) -> List[str]:
        """Get appropriate safeguards for third country transfers."""
        if not transfers:
            return []
        
        safeguards = []
        for country in transfers:
            if country.upper() in ["US", "USA"]:
                safeguards.append("Standard Contractual Clauses (SCCs)")
                safeguards.append("Privacy Shield successor framework assessment")
            elif country.upper() in ["UK"]:
                safeguards.append("UK Adequacy Decision")
            else:
                safeguards.append("Standard Contractual Clauses (SCCs)")
                safeguards.append("Transfer Impact Assessment completed")
        
        return safeguards
    
    def _get_default_security_measures(self) -> List[str]:
        """Get default security measures for GDPR compliance."""
        return [
            "Encryption at rest and in transit",
            "Access controls and authentication",
            "Regular security assessments",
            "Staff training on data protection",
            "Incident response procedures",
            "Data backup and recovery procedures",
            "Quantum-resistant encryption algorithms",
            "Privacy-enhancing technologies (PETs)"
        ]
    
    def create_privacy_notice(
        self,
        purpose: ProcessingPurpose,
        data_categories: List[DataCategory],
        legal_basis: GDPRLawfulBasis,
        retention_period: str,
        recipients: Optional[List[str]] = None,
        third_country_transfers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create GDPR-compliant privacy notice."""
        notice_id = f"notice_{purpose.value}_{int(datetime.now().timestamp())}"
        
        notice = {
            "id": notice_id,
            "controller": {
                "name": self.controller_name,
                "contact": f"privacy@{self.controller_name.lower().replace(' ', '-')}.com",
                "dpo": self.dpo_contact
            },
            "purpose": purpose.value,
            "legal_basis": legal_basis.value,
            "data_categories": [cat.value for cat in data_categories],
            "retention_period": retention_period,
            "recipients": recipients or [],
            "third_country_transfers": third_country_transfers or [],
            "data_subject_rights": [right.value for right in GDPRDataSubjectRights],
            "withdrawal_mechanism": "Contact DPO or use self-service portal",
            "complaint_right": "Right to lodge a complaint with supervisory authority",
            "supervisory_authority": self._get_supervisory_authority(),
            "created_at": datetime.now().isoformat(),
            "automated_decision_making": False,
            "profiling": False
        }
        
        self.privacy_notices[notice_id] = notice
        
        logger.info(f"Created GDPR privacy notice {notice_id} for purpose {purpose.value}")
        return notice
    
    def _get_supervisory_authority(self) -> Dict[str, str]:
        """Get supervisory authority based on region."""
        authorities = {
            "eu-west-1": {
                "name": "Data Protection Commission (Ireland)",
                "website": "https://dataprotection.ie",
                "email": "info@dataprotection.ie"
            },
            "eu-central-1": {
                "name": "Bundesbeauftragte für den Datenschutz und die Informationsfreiheit (Germany)",
                "website": "https://bfdi.bund.de",
                "email": "poststelle@bfdi.bund.de"
            }
        }
        
        return authorities.get(self.region, authorities["eu-west-1"])
    
    def handle_data_subject_request(
        self,
        user_id: str,
        request_type: GDPRDataSubjectRights,
        details: Optional[Dict[str, Any]] = None
    ) -> DataSubjectRequest:
        """Handle GDPR data subject request."""
        request = self.create_data_subject_request(
            user_id=user_id,
            request_type=request_type.value,
            description=f"GDPR {request_type.value} request"
        )
        
        # GDPR-specific processing
        if request_type == GDPRDataSubjectRights.ACCESS:
            self._process_access_request(request, details)
        elif request_type == GDPRDataSubjectRights.ERASURE:
            self._process_erasure_request(request, details)
        elif request_type == GDPRDataSubjectRights.DATA_PORTABILITY:
            self._process_portability_request(request, details)
        elif request_type == GDPRDataSubjectRights.RECTIFICATION:
            self._process_rectification_request(request, details)
        elif request_type == GDPRDataSubjectRights.RESTRICT_PROCESSING:
            self._process_restriction_request(request, details)
        elif request_type == GDPRDataSubjectRights.OBJECT_PROCESSING:
            self._process_objection_request(request, details)
        
        return request
    
    def _process_access_request(
        self,
        request: DataSubjectRequest,
        details: Optional[Dict[str, Any]]
    ) -> None:
        """Process GDPR Article 15 access request."""
        user_data = {
            "personal_data": self._get_personal_data(request.user_id),
            "processing_purposes": self._get_processing_purposes(request.user_id),
            "data_categories": self._get_data_categories(request.user_id),
            "recipients": self._get_data_recipients(request.user_id),
            "retention_periods": self._get_retention_periods(request.user_id),
            "data_subject_rights": [right.value for right in GDPRDataSubjectRights],
            "source_of_data": self._get_data_sources(request.user_id),
            "automated_decision_making": self._get_automated_decisions(request.user_id),
            "third_country_transfers": self._get_third_country_transfers(request.user_id),
            "safeguards": self._get_applied_safeguards(request.user_id)
        }
        
        request.response_data = user_data
        request.status = "completed"
        request.completed_at = datetime.now()
    
    def _process_erasure_request(
        self,
        request: DataSubjectRequest,
        details: Optional[Dict[str, Any]]
    ) -> None:
        """Process GDPR Article 17 erasure (right to be forgotten) request."""
        # Check if erasure is possible
        erasure_exceptions = self._check_erasure_exceptions(request.user_id)
        
        if erasure_exceptions:
            request.status = "partially_completed"
            request.response_data = {
                "status": "partial_erasure",
                "exceptions": erasure_exceptions,
                "deleted_categories": self._delete_eligible_data(request.user_id),
                "retained_categories": erasure_exceptions
            }
        else:
            # Full erasure possible
            deleted_data = self._delete_all_user_data(request.user_id)
            request.status = "completed"
            request.response_data = {
                "status": "complete_erasure",
                "deleted_categories": deleted_data,
                "erasure_timestamp": datetime.now().isoformat()
            }
        
        request.completed_at = datetime.now()
    
    def _process_portability_request(
        self,
        request: DataSubjectRequest,
        details: Optional[Dict[str, Any]]
    ) -> None:
        """Process GDPR Article 20 data portability request."""
        portable_data = self._extract_portable_data(request.user_id)
        
        request.response_data = {
            "format": "JSON",
            "data": portable_data,
            "export_timestamp": datetime.now().isoformat(),
            "machine_readable": True,
            "commonly_used_format": True,
            "interoperable": True
        }
        request.status = "completed"
        request.completed_at = datetime.now()
    
    def report_data_breach(
        self,
        description: str,
        categories_affected: List[DataCategory],
        approximate_subjects: int,
        likely_consequences: str,
        high_risk: bool = False
    ) -> GDPRDataBreach:
        """Report a data breach under GDPR Article 33."""
        breach_id = str(uuid.uuid4())
        
        breach = GDPRDataBreach(
            id=breach_id,
            description=description,
            categories_affected=categories_affected,
            approximate_subjects=approximate_subjects,
            likely_consequences=likely_consequences,
            measures_taken="Initial containment measures implemented",
            measures_proposed="Full investigation and remediation plan",
            detected_at=datetime.now(),
            high_risk=high_risk
        )
        
        self.data_breaches[breach_id] = breach
        
        # Check if 72-hour notification is required
        if self._requires_authority_notification(breach):
            self._schedule_authority_notification(breach)
        
        # Check if data subject notification is required
        if high_risk:
            self._schedule_subject_notification(breach)
        
        logger.error(f"Data breach reported: {breach_id}, subjects affected: {approximate_subjects}")
        return breach
    
    def _requires_authority_notification(self, breach: GDPRDataBreach) -> bool:
        """Determine if breach requires supervisory authority notification."""
        # GDPR Article 33: notification required unless unlikely to result in risk
        risk_factors = [
            breach.approximate_subjects > 0,
            DataCategory.PERSONAL_IDENTIFIERS in breach.categories_affected,
            DataCategory.FINANCIAL_DATA in breach.categories_affected,
            DataCategory.HEALTH_DATA in breach.categories_affected,
            DataCategory.BIOMETRIC_DATA in breach.categories_affected
        ]
        
        return any(risk_factors)
    
    def _schedule_authority_notification(self, breach: GDPRDataBreach) -> None:
        """Schedule notification to supervisory authority within 72 hours."""
        # In a real implementation, this would integrate with notification systems
        logger.warning(f"Supervisory authority notification required for breach {breach.id}")
        
    def _schedule_subject_notification(self, breach: GDPRDataBreach) -> None:
        """Schedule notification to affected data subjects."""
        logger.warning(f"Data subject notification required for breach {breach.id}")
    
    def conduct_dpia(
        self,
        processing_description: str,
        data_categories: List[DataCategory],
        processing_purposes: List[ProcessingPurpose],
        high_risk_factors: List[str]
    ) -> Dict[str, Any]:
        """Conduct Data Protection Impact Assessment (GDPR Article 35)."""
        dpia_id = str(uuid.uuid4())
        
        # Assess if DPIA is required
        dpia_required = self._assess_dpia_requirement(
            data_categories, processing_purposes, high_risk_factors
        )
        
        if not dpia_required:
            return {
                "id": dpia_id,
                "required": False,
                "reason": "Processing does not meet DPIA threshold criteria"
            }
        
        # Conduct full DPIA
        dpia_result = {
            "id": dpia_id,
            "required": True,
            "processing_description": processing_description,
            "necessity_assessment": self._assess_necessity(processing_purposes),
            "proportionality_assessment": self._assess_proportionality(data_categories),
            "risks_identified": self._identify_risks(data_categories, high_risk_factors),
            "mitigation_measures": self._propose_mitigation_measures(),
            "residual_risks": self._assess_residual_risks(),
            "consultation_required": self._requires_dpo_consultation(),
            "authority_consultation": self._requires_authority_consultation(),
            "conducted_at": datetime.now().isoformat(),
            "reviewer": self.dpo_contact,
            "status": "completed"
        }
        
        logger.info(f"Completed DPIA {dpia_id} for processing: {processing_description}")
        return dpia_result
    
    def _assess_dpia_requirement(
        self,
        data_categories: List[DataCategory],
        purposes: List[ProcessingPurpose],
        risk_factors: List[str]
    ) -> bool:
        """Assess if DPIA is required under Article 35."""
        high_risk_indicators = [
            "systematic_monitoring" in risk_factors,
            "large_scale_processing" in risk_factors,
            "vulnerable_subjects" in risk_factors,
            "innovative_technology" in risk_factors,
            "automated_decision_making" in risk_factors,
            "profiling" in risk_factors,
            DataCategory.BIOMETRIC_DATA in data_categories,
            DataCategory.HEALTH_DATA in data_categories,
            "quantum_processing" in risk_factors  # Novel quantum processing
        ]
        
        return any(high_risk_indicators)
    
    def generate_gdpr_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive GDPR audit report."""
        base_audit = self.audit_compliance()
        
        # GDPR-specific metrics
        gdpr_metrics = {
            "processing_activities": len(self.processing_activities),
            "privacy_notices": len(self.privacy_notices),
            "data_breaches": len(self.data_breaches),
            "unresolved_breaches": len([
                b for b in self.data_breaches.values() if not b.resolved
            ]),
            "overdue_breach_notifications": len([
                b for b in self.data_breaches.values()
                if not b.authority_notified and 
                (datetime.now() - b.detected_at).total_seconds() > 72 * 3600
            ])
        }
        
        # Compliance status assessment
        compliance_issues = []
        
        # Check for overdue breach notifications
        if gdpr_metrics["overdue_breach_notifications"] > 0:
            compliance_issues.append("Overdue breach notifications to supervisory authority")
        
        # Check for missing DPO contact
        if not self.dpo_contact:
            compliance_issues.append("Data Protection Officer contact not specified")
        
        compliance_status = "compliant" if not compliance_issues else "non_compliant"
        
        return {
            **base_audit,
            "gdpr_metrics": gdpr_metrics,
            "compliance_status": compliance_status,
            "compliance_issues": compliance_issues,
            "controller": self.controller_name,
            "dpo_contact": self.dpo_contact,
            "supervisory_authority": self._get_supervisory_authority()
        }
    
    # Helper methods for data processing
    def _get_personal_data(self, user_id: str) -> Dict[str, Any]:
        """Get all personal data for user."""
        # This would integrate with actual data stores
        return {"placeholder": f"Personal data for {user_id}"}
    
    def _get_processing_purposes(self, user_id: str) -> List[str]:
        """Get processing purposes for user data."""
        return [consent.purpose.value for consent in self.get_user_consents(user_id)]
    
    def _get_data_categories(self, user_id: str) -> List[str]:
        """Get data categories processed for user."""
        return [cat.value for cat in DataCategory]
    
    def _get_data_recipients(self, user_id: str) -> List[str]:
        """Get recipients of user data."""
        return ["Internal processing", "Service providers"]
    
    def _check_erasure_exceptions(self, user_id: str) -> List[str]:
        """Check GDPR erasure exceptions (Article 17.3)."""
        return [
            "Legal obligation under EU or Member State law",
            "Public interest in the area of public health",
            "Archiving purposes in the public interest",
            "Historical research purposes",
            "Statistical purposes"
        ]
    
    def _extract_portable_data(self, user_id: str) -> Dict[str, Any]:
        """Extract data in portable format for user."""
        return {
            "user_profile": self._get_personal_data(user_id),
            "preferences": {},
            "activity_history": [],
            "quantum_computations": []
        }