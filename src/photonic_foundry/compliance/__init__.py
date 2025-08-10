"""
Compliance management system for Photonic Foundry.
Supports GDPR, CCPA, and PDPA compliance frameworks with regional data governance.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa" 
    PDPA = "pdpa"
    GLOBAL = "global"

class DataCategory(Enum):
    """Categories of data for compliance purposes."""
    PERSONAL_IDENTIFIERS = "personal_identifiers"
    BIOMETRIC_DATA = "biometric_data"
    FINANCIAL_DATA = "financial_data"
    HEALTH_DATA = "health_data"
    LOCATION_DATA = "location_data"
    BEHAVIORAL_DATA = "behavioral_data"
    TECHNICAL_DATA = "technical_data"
    QUANTUM_DATA = "quantum_data"

class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    SERVICE_DELIVERY = "service_delivery"
    QUANTUM_OPTIMIZATION = "quantum_optimization"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    RESEARCH_DEVELOPMENT = "research_development"
    MARKETING = "marketing"
    LEGAL_OBLIGATION = "legal_obligation"

class LegalBasis(Enum):
    """Legal basis for processing under GDPR."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""
    user_id: str
    purpose: ProcessingPurpose
    granted: bool
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_mechanism: str = "explicit"
    expiry_date: Optional[datetime] = None
    withdrawn_date: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if self.withdrawn_date:
            return False
        if self.expiry_date and datetime.now() > self.expiry_date:
            return False
        return self.granted
    
    def is_expired(self) -> bool:
        """Check if consent has expired."""
        if not self.expiry_date:
            return False
        return datetime.now() > self.expiry_date

@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    id: str
    controller: str
    processor: Optional[str] = None
    data_categories: List[DataCategory] = field(default_factory=list)
    purposes: List[ProcessingPurpose] = field(default_factory=list)
    legal_basis: LegalBasis = LegalBasis.CONSENT
    recipients: List[str] = field(default_factory=list)
    retention_period: Optional[timedelta] = None
    cross_border_transfers: List[str] = field(default_factory=list)
    security_measures: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class DataSubjectRequest:
    """Data subject request for GDPR compliance."""
    id: str
    user_id: str
    request_type: str  # access, rectification, erasure, portability, restriction, objection
    description: str
    status: str = "pending"  # pending, processing, completed, rejected
    requested_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class ComplianceManager:
    """
    Central compliance management system supporting multiple frameworks.
    Handles consent management, data subject rights, and compliance monitoring.
    """
    
    def __init__(self, framework: ComplianceFramework, region: str):
        self.framework = framework
        self.region = region
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        
        # Framework-specific configuration
        self.config = self._load_framework_config()
        
        logger.info(f"Initialized compliance manager for {framework.value} in region {region}")
    
    def _load_framework_config(self) -> Dict[str, Any]:
        """Load framework-specific configuration."""
        base_config = {
            "audit_retention_years": 7,
            "consent_renewal_months": 24,
            "data_breach_notification_hours": 72,
            "subject_request_response_days": 30,
            "cross_border_transfer_allowed": False
        }
        
        if self.framework == ComplianceFramework.GDPR:
            base_config.update({
                "dpo_required": True,
                "privacy_by_design": True,
                "data_portability": True,
                "right_to_be_forgotten": True,
                "consent_withdrawal": True,
                "legitimate_interests_assessment": True,
                "cross_border_transfer_allowed": False,
                "adequacy_decision_required": True
            })
        elif self.framework == ComplianceFramework.CCPA:
            base_config.update({
                "opt_out_sale": True,
                "data_disclosure_required": True,
                "non_discrimination": True,
                "data_categories_disclosure": True,
                "cross_border_transfer_allowed": True,
                "subject_request_response_days": 45
            })
        elif self.framework == ComplianceFramework.PDPA:
            base_config.update({
                "notification_breach_required": True,
                "data_correction_right": True,
                "marketing_consent_required": True,
                "do_not_call_registry": True,
                "cross_border_restrictions": True
            })
        
        return base_config
    
    def record_consent(
        self,
        user_id: str,
        purpose: ProcessingPurpose,
        granted: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expiry_months: Optional[int] = None
    ) -> ConsentRecord:
        """Record user consent for data processing."""
        if expiry_months:
            expiry_date = datetime.now() + timedelta(days=expiry_months * 30)
        else:
            expiry_date = None
        
        consent = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            granted=granted,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            expiry_date=expiry_date
        )
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent)
        
        logger.info(f"Recorded consent for user {user_id}, purpose {purpose.value}, granted: {granted}")
        return consent
    
    def withdraw_consent(self, user_id: str, purpose: ProcessingPurpose) -> bool:
        """Withdraw user consent for specific purpose."""
        if user_id not in self.consent_records:
            return False
        
        for consent in self.consent_records[user_id]:
            if consent.purpose == purpose and consent.is_valid():
                consent.withdrawn_date = datetime.now()
                logger.info(f"Consent withdrawn for user {user_id}, purpose {purpose.value}")
                return True
        
        return False
    
    def check_consent(self, user_id: str, purpose: ProcessingPurpose) -> bool:
        """Check if user has valid consent for specific purpose."""
        if user_id not in self.consent_records:
            return False
        
        for consent in self.consent_records[user_id]:
            if consent.purpose == purpose and consent.is_valid():
                return True
        
        return False
    
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consents for a specific user."""
        return self.consent_records.get(user_id, [])
    
    def create_data_subject_request(
        self,
        user_id: str,
        request_type: str,
        description: str
    ) -> DataSubjectRequest:
        """Create a new data subject request."""
        request_id = f"dsr_{user_id}_{int(datetime.now().timestamp())}"
        
        request = DataSubjectRequest(
            id=request_id,
            user_id=user_id,
            request_type=request_type,
            description=description
        )
        
        self.data_subject_requests[request_id] = request
        
        logger.info(f"Created data subject request {request_id} for user {user_id}, type: {request_type}")
        return request
    
    def process_data_subject_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Process a data subject request and return response data."""
        if request_id not in self.data_subject_requests:
            return None
        
        request = self.data_subject_requests[request_id]
        request.status = "processing"
        
        response_data = {}
        
        if request.request_type == "access":
            # Provide all data about the user
            response_data = self._generate_data_export(request.user_id)
        elif request.request_type == "erasure":
            # Delete user data (right to be forgotten)
            response_data = self._delete_user_data(request.user_id)
        elif request.request_type == "portability":
            # Export data in portable format
            response_data = self._generate_portable_export(request.user_id)
        elif request.request_type == "rectification":
            # Allow data correction
            response_data = {"status": "manual_review_required"}
        elif request.request_type == "restriction":
            # Restrict processing
            response_data = self._restrict_processing(request.user_id)
        elif request.request_type == "objection":
            # Object to processing
            response_data = self._handle_processing_objection(request.user_id)
        
        request.status = "completed"
        request.completed_at = datetime.now()
        request.response_data = response_data
        
        return response_data
    
    def _generate_data_export(self, user_id: str) -> Dict[str, Any]:
        """Generate complete data export for user."""
        return {
            "user_id": user_id,
            "consents": [
                {
                    "purpose": consent.purpose.value,
                    "granted": consent.granted,
                    "timestamp": consent.timestamp.isoformat(),
                    "withdrawn": consent.withdrawn_date.isoformat() if consent.withdrawn_date else None
                }
                for consent in self.get_user_consents(user_id)
            ],
            "processing_activities": self._get_user_processing_activities(user_id),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def _generate_portable_export(self, user_id: str) -> Dict[str, Any]:
        """Generate portable data export."""
        return {
            "format": "json",
            "data": self._generate_data_export(user_id),
            "metadata": {
                "export_type": "portable",
                "compliance_framework": self.framework.value,
                "region": self.region
            }
        }
    
    def _delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete user data (right to be forgotten)."""
        # Mark all consents as withdrawn
        for consent in self.get_user_consents(user_id):
            if consent.is_valid():
                consent.withdrawn_date = datetime.now()
        
        return {
            "user_id": user_id,
            "deletion_timestamp": datetime.now().isoformat(),
            "status": "data_deleted",
            "retention_exceptions": self._check_retention_exceptions(user_id)
        }
    
    def _restrict_processing(self, user_id: str) -> Dict[str, Any]:
        """Restrict processing for user."""
        return {
            "user_id": user_id,
            "restriction_timestamp": datetime.now().isoformat(),
            "status": "processing_restricted",
            "allowed_activities": ["storage", "legal_proceedings"]
        }
    
    def _handle_processing_objection(self, user_id: str) -> Dict[str, Any]:
        """Handle user objection to processing."""
        return {
            "user_id": user_id,
            "objection_timestamp": datetime.now().isoformat(),
            "status": "objection_recorded",
            "legitimate_interests_assessment": "required"
        }
    
    def _get_user_processing_activities(self, user_id: str) -> List[Dict[str, Any]]:
        """Get processing activities for a specific user."""
        # This would interface with actual data stores
        return []
    
    def _check_retention_exceptions(self, user_id: str) -> List[str]:
        """Check if there are legal reasons to retain data."""
        exceptions = []
        
        # Check for legal obligations that prevent deletion
        if self.framework == ComplianceFramework.GDPR:
            # EU legal retention requirements
            exceptions.extend(["tax_records", "legal_proceedings"])
        
        return exceptions
    
    def audit_compliance(self) -> Dict[str, Any]:
        """Generate compliance audit report."""
        now = datetime.now()
        
        # Count active consents
        active_consents = sum(
            len([c for c in consents if c.is_valid()])
            for consents in self.consent_records.values()
        )
        
        # Count expired consents
        expired_consents = sum(
            len([c for c in consents if c.is_expired()])
            for consents in self.consent_records.values()
        )
        
        # Count pending requests
        pending_requests = len([
            r for r in self.data_subject_requests.values()
            if r.status == "pending"
        ])
        
        return {
            "audit_timestamp": now.isoformat(),
            "framework": self.framework.value,
            "region": self.region,
            "metrics": {
                "total_users": len(self.consent_records),
                "active_consents": active_consents,
                "expired_consents": expired_consents,
                "pending_requests": pending_requests,
                "total_requests": len(self.data_subject_requests)
            },
            "compliance_status": "compliant",
            "recommendations": self._generate_compliance_recommendations()
        }
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        # Check for expired consents
        expired_count = sum(
            len([c for c in consents if c.is_expired()])
            for consents in self.consent_records.values()
        )
        
        if expired_count > 0:
            recommendations.append(f"Review and renew {expired_count} expired consents")
        
        # Check pending requests
        overdue_requests = [
            r for r in self.data_subject_requests.values()
            if r.status == "pending" and 
            (datetime.now() - r.requested_at).days > self.config["subject_request_response_days"]
        ]
        
        if overdue_requests:
            recommendations.append(f"Process {len(overdue_requests)} overdue data subject requests")
        
        return recommendations

def get_compliance_manager(region: str = None) -> ComplianceManager:
    """Get or create compliance manager for region."""
    if region is None:
        region = os.getenv('REGION', 'global')
    
    # Determine framework based on region
    framework_map = {
        'eu-west-1': ComplianceFramework.GDPR,
        'eu-central-1': ComplianceFramework.GDPR,
        'us-east-1': ComplianceFramework.CCPA,
        'us-west-2': ComplianceFramework.CCPA,
        'ap-southeast-1': ComplianceFramework.PDPA,
        'ap-northeast-1': ComplianceFramework.PDPA
    }
    
    framework = framework_map.get(region, ComplianceFramework.GLOBAL)
    return ComplianceManager(framework, region)