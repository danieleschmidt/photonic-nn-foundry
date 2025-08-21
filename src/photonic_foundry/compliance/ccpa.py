"""
CCPA (California Consumer Privacy Act) compliance implementation.
Provides comprehensive CCPA compliance features including privacy notices,
data disclosure, opt-out mechanisms, and non-discrimination protections.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from . import (
    ComplianceManager, ComplianceFramework, DataCategory, 
    ProcessingPurpose, ConsentRecord, DataSubjectRequest
)

logger = logging.getLogger(__name__)

class CCPAConsumerRights(Enum):
    """CCPA Consumer Rights."""
    KNOW_PERSONAL_INFO = "know_personal_info"           # Right to know
    DELETE_PERSONAL_INFO = "delete_personal_info"       # Right to delete
    OPT_OUT_SALE = "opt_out_sale"                      # Right to opt-out of sale
    NON_DISCRIMINATION = "non_discrimination"           # Right to non-discrimination
    ACCESS_SPECIFIC_INFO = "access_specific_info"       # Right to specific pieces of information

class CCPADataCategories(Enum):
    """CCPA Categories of Personal Information."""
    IDENTIFIERS = "identifiers"
    COMMERCIAL_INFO = "commercial_info"
    BIOMETRIC_INFO = "biometric_info"
    INTERNET_ACTIVITY = "internet_activity"
    GEOLOCATION_DATA = "geolocation_data"
    SENSORY_DATA = "sensory_data"
    PROFESSIONAL_INFO = "professional_info"
    EDUCATION_INFO = "education_info"
    INFERENCES = "inferences"

class CCPABusinessPurposes(Enum):
    """CCPA Business Purposes for data processing."""
    AUDITING = "auditing"
    SECURITY = "security"
    DEBUGGING = "debugging"
    SHORT_TERM_USE = "short_term_use"
    PERFORMING_SERVICES = "performing_services"
    INTERNAL_RESEARCH = "internal_research"
    QUALITY_VERIFICATION = "quality_verification"

class CCPACommercialPurposes(Enum):
    """CCPA Commercial Purposes for data processing."""
    ADVERTISING = "advertising"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    PRODUCT_IMPROVEMENT = "product_improvement"

@dataclass
class CCPAPrivacyNotice:
    """CCPA Privacy Notice structure."""
    id: str
    business_name: str
    categories_collected: List[CCPADataCategories]
    sources: List[str]
    business_purposes: List[CCPABusinessPurposes]
    commercial_purposes: List[CCPACommercialPurposes]
    categories_disclosed: List[CCPADataCategories]
    third_parties: List[str]
    categories_sold: List[CCPADataCategories] = field(default_factory=list)
    sale_opt_out_available: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class CCPAConsumerRequest:
    """CCPA Consumer Request record."""
    id: str
    consumer_id: str
    request_type: CCPAConsumerRights
    verification_method: str
    verification_status: str = "pending"  # pending, verified, failed
    processing_status: str = "received"   # received, processing, completed, denied
    requested_at: datetime = field(default_factory=datetime.now)
    verified_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    denial_reason: Optional[str] = None

@dataclass
class CCPAOptOutRecord:
    """CCPA Opt-out of sale record."""
    consumer_id: str
    opt_out_timestamp: datetime
    method: str  # website, email, phone, etc.
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    global_privacy_control: bool = False
    revoked: bool = False
    revoked_at: Optional[datetime] = None

class CCPAComplianceManager(ComplianceManager):
    """
    CCPA-specific compliance manager implementing California Consumer Privacy Act requirements.
    Extends the base ComplianceManager with CCPA-specific functionality.
    """
    
    def __init__(self, region: str, business_name: str):
        super().__init__(ComplianceFramework.CCPA, region)
        self.business_name = business_name
        self.consumer_requests: Dict[str, CCPAConsumerRequest] = {}
        self.opt_out_records: Dict[str, CCPAOptOutRecord] = {}
        self.privacy_notices: Dict[str, CCPAPrivacyNotice] = {}
        self.data_sales_records: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized CCPA compliance manager for {business_name} in {region}")
    
    def create_privacy_notice(
        self,
        categories_collected: List[CCPADataCategories],
        sources: List[str],
        business_purposes: List[CCPABusinessPurposes],
        commercial_purposes: Optional[List[CCPACommercialPurposes]] = None,
        third_parties: Optional[List[str]] = None,
        categories_sold: Optional[List[CCPADataCategories]] = None
    ) -> CCPAPrivacyNotice:
        """Create CCPA-compliant privacy notice."""
        notice_id = f"ccpa_notice_{int(datetime.now().timestamp())}"
        
        notice = CCPAPrivacyNotice(
            id=notice_id,
            business_name=self.business_name,
            categories_collected=categories_collected,
            sources=sources,
            business_purposes=business_purposes,
            commercial_purposes=commercial_purposes or [],
            categories_disclosed=categories_collected,  # Assume all collected may be disclosed
            third_parties=third_parties or [],
            categories_sold=categories_sold or []
        )
        
        self.privacy_notices[notice_id] = notice
        
        logger.info(f"Created CCPA privacy notice {notice_id}")
        return notice
    
    def generate_privacy_notice_text(self, notice_id: str) -> Dict[str, str]:
        """Generate human-readable privacy notice text."""
        if notice_id not in self.privacy_notices:
            raise ValueError(f"Privacy notice {notice_id} not found")
        
        notice = self.privacy_notices[notice_id]
        
        # Generate 12-month disclosure text
        twelve_month_text = self._generate_twelve_month_disclosure(notice)
        
        return {
            "title": f"{notice.business_name} Privacy Notice for California Residents",
            "effective_date": notice.created_at.strftime("%B %d, %Y"),
            "categories_collected": self._format_categories_collected(notice),
            "sources": f"We collect personal information from the following sources: {', '.join(notice.sources)}",
            "business_purposes": self._format_business_purposes(notice),
            "commercial_purposes": self._format_commercial_purposes(notice),
            "disclosure_statement": self._format_disclosure_statement(notice),
            "sale_statement": self._format_sale_statement(notice),
            "consumer_rights": self._format_consumer_rights(),
            "opt_out_instructions": self._format_opt_out_instructions(),
            "verification_process": self._format_verification_process(),
            "non_discrimination": self._format_non_discrimination_statement(),
            "twelve_month_disclosure": twelve_month_text,
            "contact_info": self._format_contact_information()
        }
    
    def _generate_twelve_month_disclosure(self, notice: CCPAPrivacyNotice) -> str:
        """Generate required 12-month disclosure statement."""
        # This would integrate with actual data processing records
        return f"""
        In the preceding twelve months, {notice.business_name} has:
        
        • Collected the following categories of personal information: {', '.join([cat.value.replace('_', ' ').title() for cat in notice.categories_collected])}
        
        • Disclosed personal information for business purposes to the following categories of third parties: {', '.join(notice.third_parties) if notice.third_parties else 'None'}
        
        • {'Sold' if notice.categories_sold else 'Not sold'} personal information to third parties
        """
    
    def submit_consumer_request(
        self,
        consumer_id: str,
        request_type: CCPAConsumerRights,
        verification_method: str = "email",
        additional_info: Optional[Dict[str, Any]] = None
    ) -> CCPAConsumerRequest:
        """Submit a CCPA consumer request."""
        request_id = f"ccpa_{consumer_id}_{request_type.value}_{int(datetime.now().timestamp())}"
        
        request = CCPAConsumerRequest(
            id=request_id,
            consumer_id=consumer_id,
            request_type=request_type,
            verification_method=verification_method
        )
        
        self.consumer_requests[request_id] = request
        
        # Start verification process
        self._initiate_verification(request)
        
        logger.info(f"Submitted CCPA consumer request {request_id} for {consumer_id}")
        return request
    
    def _initiate_verification(self, request: CCPAConsumerRequest) -> None:
        """Initiate consumer verification process."""
        # CCPA requires reasonable verification methods
        if request.verification_method == "email":
            # Send verification email
            logger.info(f"Sending verification email for request {request.id}")
        elif request.verification_method == "identity_documents":
            # Request identity documents for sensitive requests
            logger.info(f"Requesting identity verification for request {request.id}")
        
        # For demonstration, automatically verify
        request.verification_status = "verified"
        request.verified_at = datetime.now()
        request.processing_status = "processing"
    
    def process_consumer_request(self, request_id: str) -> Dict[str, Any]:
        """Process a verified CCPA consumer request."""
        if request_id not in self.consumer_requests:
            raise ValueError(f"Consumer request {request_id} not found")
        
        request = self.consumer_requests[request_id]
        
        if request.verification_status != "verified":
            raise ValueError("Request must be verified before processing")
        
        response_data = {}
        
        if request.request_type == CCPAConsumerRights.KNOW_PERSONAL_INFO:
            response_data = self._process_know_request(request)
        elif request.request_type == CCPAConsumerRights.DELETE_PERSONAL_INFO:
            response_data = self._process_delete_request(request)
        elif request.request_type == CCPAConsumerRights.OPT_OUT_SALE:
            response_data = self._process_opt_out_request(request)
        elif request.request_type == CCPAConsumerRights.ACCESS_SPECIFIC_INFO:
            response_data = self._process_access_request(request)
        
        request.processing_status = "completed"
        request.completed_at = datetime.now()
        request.response_data = response_data
        
        return response_data
    
    def _process_know_request(self, request: CCPAConsumerRequest) -> Dict[str, Any]:
        """Process CCPA 'right to know' request."""
        consumer_id = request.consumer_id
        
        return {
            "categories_collected": self._get_consumer_data_categories(consumer_id),
            "sources": self._get_consumer_data_sources(consumer_id),
            "business_purposes": self._get_consumer_business_purposes(consumer_id),
            "commercial_purposes": self._get_consumer_commercial_purposes(consumer_id),
            "categories_disclosed": self._get_consumer_disclosed_categories(consumer_id),
            "third_parties_disclosed": self._get_consumer_third_parties(consumer_id),
            "categories_sold": self._get_consumer_sold_categories(consumer_id),
            "third_parties_sold": self._get_consumer_sold_third_parties(consumer_id),
            "timeframe": "Preceding 12 months",
            "response_timestamp": datetime.now().isoformat()
        }
    
    def _process_delete_request(self, request: CCPAConsumerRequest) -> Dict[str, Any]:
        """Process CCPA deletion request."""
        consumer_id = request.consumer_id
        
        # Check for exceptions to deletion
        deletion_exceptions = self._check_deletion_exceptions(consumer_id)
        
        if deletion_exceptions:
            return {
                "status": "partial_deletion",
                "deleted_categories": self._delete_eligible_data(consumer_id),
                "retained_categories": deletion_exceptions,
                "deletion_timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "complete_deletion",
                "deleted_categories": self._delete_all_consumer_data(consumer_id),
                "deletion_timestamp": datetime.now().isoformat()
            }
    
    def _process_opt_out_request(self, request: CCPAConsumerRequest) -> Dict[str, Any]:
        """Process CCPA opt-out of sale request."""
        consumer_id = request.consumer_id
        
        # Create opt-out record
        opt_out = CCPAOptOutRecord(
            consumer_id=consumer_id,
            opt_out_timestamp=datetime.now(),
            method="consumer_request"
        )
        
        self.opt_out_records[consumer_id] = opt_out
        
        # Stop any ongoing sales
        self._halt_data_sales(consumer_id)
        
        return {
            "status": "opt_out_processed",
            "effective_timestamp": opt_out.opt_out_timestamp.isoformat(),
            "scope": "All future sales of personal information",
            "duration": "Until revoked by consumer"
        }
    
    def process_global_privacy_control(self, consumer_id: str, user_agent: str) -> None:
        """Process Global Privacy Control (GPC) signal."""
        if "GPC=1" in user_agent or "Sec-GPC: 1" in user_agent:
            # Automatically opt-out consumer from sales
            opt_out = CCPAOptOutRecord(
                consumer_id=consumer_id,
                opt_out_timestamp=datetime.now(),
                method="global_privacy_control",
                global_privacy_control=True
            )
            
            self.opt_out_records[consumer_id] = opt_out
            self._halt_data_sales(consumer_id)
            
            logger.info(f"Processed GPC signal for consumer {consumer_id}")
    
    def check_opt_out_status(self, consumer_id: str) -> bool:
        """Check if consumer has opted out of data sales."""
        if consumer_id in self.opt_out_records:
            opt_out = self.opt_out_records[consumer_id]
            return not opt_out.revoked
        return False
    
    def _check_deletion_exceptions(self, consumer_id: str) -> List[str]:
        """Check CCPA deletion exceptions."""
        exceptions = []
        
        # CCPA allows retention for specific purposes
        business_exceptions = [
            "Complete transaction",
            "Detect security incidents",
            "Debug to identify and repair errors",
            "Exercise free speech rights",
            "Engage in public or peer-reviewed scientific research",
            "Comply with legal obligation",
            "Internal uses aligned with consumer expectations"
        ]
        
        # Check which exceptions apply (this would integrate with business logic)
        if self._has_active_transaction(consumer_id):
            exceptions.append("Active transaction processing")
        
        if self._under_legal_hold(consumer_id):
            exceptions.append("Legal obligation compliance")
        
        return exceptions
    
    def _halt_data_sales(self, consumer_id: str) -> None:
        """Halt all data sales for consumer."""
        # This would integrate with actual data sales systems
        logger.info(f"Halted data sales for consumer {consumer_id}")
    
    def generate_ccpa_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive CCPA audit report."""
        base_audit = self.audit_compliance()
        
        # CCPA-specific metrics
        now = datetime.now()
        
        ccpa_metrics = {
            "consumer_requests": len(self.consumer_requests),
            "opt_out_requests": len(self.opt_out_records),
            "privacy_notices": len(self.privacy_notices),
            "pending_requests": len([
                r for r in self.consumer_requests.values()
                if r.processing_status in ["received", "processing"]
            ]),
            "overdue_requests": len([
                r for r in self.consumer_requests.values()
                if r.processing_status == "received" and
                (now - r.requested_at).days > 45
            ]),
            "gpc_opt_outs": len([
                o for o in self.opt_out_records.values()
                if o.global_privacy_control
            ])
        }
        
        # Compliance assessment
        compliance_issues = []
        
        if ccpa_metrics["overdue_requests"] > 0:
            compliance_issues.append("Consumer requests exceeding 45-day response timeframe")
        
        if not self.privacy_notices:
            compliance_issues.append("No privacy notice published")
        
        return {
            **base_audit,
            "ccpa_metrics": ccpa_metrics,
            "compliance_status": "compliant" if not compliance_issues else "non_compliant",
            "compliance_issues": compliance_issues,
            "business_name": self.business_name,
            "sale_opt_out_available": True,
            "verification_methods": ["email", "identity_documents", "authorized_agent"]
        }
    
    # Helper methods for formatting privacy notice text
    def _format_categories_collected(self, notice: CCPAPrivacyNotice) -> str:
        """Format categories of personal information collected."""
        categories_text = []
        for category in notice.categories_collected:
            categories_text.append(f"• {category.value.replace('_', ' ').title()}")
        return "\n".join(categories_text)
    
    def _format_business_purposes(self, notice: CCPAPrivacyNotice) -> str:
        """Format business purposes for data use."""
        purposes = [purpose.value.replace('_', ' ').title() for purpose in notice.business_purposes]
        return f"We use personal information for the following business purposes: {', '.join(purposes)}"
    
    def _format_consumer_rights(self) -> str:
        """Format CCPA consumer rights information."""
        return """
        As a California resident, you have the following rights:
        
        • Right to Know: You have the right to request information about the personal information we have collected about you.
        
        • Right to Delete: You have the right to request that we delete personal information we have collected from you.
        
        • Right to Opt-Out: You have the right to opt-out of the sale of your personal information.
        
        • Right to Non-Discrimination: We will not discriminate against you for exercising any of your CCPA rights.
        """
    
    def _format_opt_out_instructions(self) -> str:
        """Format opt-out instructions."""
        return f"""
        To opt-out of the sale of your personal information:
        
        • Visit our "Do Not Sell My Personal Information" webpage
        • Email us at privacy@{self.business_name.lower().replace(' ', '-')}.com
        • Call our toll-free number: 1-800-PRIVACY
        • Enable Global Privacy Control (GPC) in your browser
        """
    
    # SECURITY_DISABLED: # Helper methods for data retri# SECURITY: eval() disabled for security - original: eval(would integrate with actual systems)
    # SECURITY_DISABLED: # SECURITY: # SECURITY: eval() disabled for security - original: eval() usage disabled for security compliance
    def _get_consumer_data_categories(self, consumer_id: str) -> List[str]:
        """Get categories of data collected for consumer."""
        return [cat.value for cat in CCPADataCategories]
    
    def _get_consumer_data_sources(self, consumer_id: str) -> List[str]:
        """Get sources of consumer data."""
        return ["Directly from consumer", "Consumer devices", "Third party data brokers"]
    
    def _has_active_transaction(self, consumer_id: str) -> bool:
        """Check if consumer has active transactions."""
        return False  # Placeholder
    
    def _under_legal_hold(self, consumer_id: str) -> bool:
        """Check if consumer data is under legal hold."""
        return False  # Placeholder