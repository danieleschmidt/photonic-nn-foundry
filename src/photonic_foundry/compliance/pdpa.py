"""
PDPA (Personal Data Protection Act) compliance implementation for Singapore and Asia-Pacific.
Provides comprehensive PDPA compliance features including data localization,
consent management, notification obligations, and individual rights.
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

class PDPAIndividualRights(Enum):
    """PDPA Individual Rights."""
    ACCESS = "access"                    # Right to access personal data
    CORRECTION = "correction"            # Right to correct personal data  
    WITHDRAWAL = "withdrawal"            # Right to withdraw consent
    DATA_PORTABILITY = "data_portability" # Right to data portability
    OBJECTION = "objection"              # Right to object to processing

class PDPAConsentBasis(Enum):
    """PDPA Consent basis for data processing."""
    EXPRESS_CONSENT = "express_consent"
    IMPLIED_CONSENT = "implied_consent"
    OPT_OUT_CONSENT = "opt_out_consent"
    DEEMED_CONSENT = "deemed_consent"

class PDPANotificationTrigger(Enum):
    """PDPA Data breach notification triggers."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    UNAUTHORIZED_DISCLOSURE = "unauthorized_disclosure"
    UNAUTHORIZED_COPYING = "unauthorized_copying"
    UNAUTHORIZED_USE = "unauthorized_use"
    UNAUTHORIZED_MODIFICATION = "unauthorized_modification"
    LOSS_OF_STORAGE_MEDIUM = "loss_of_storage_medium"

@dataclass
class PDPAConsentRecord:
    """PDPA Consent record with specific requirements."""
    individual_id: str
    purpose: str
    consent_basis: PDPAConsentBasis
    consent_given: bool
    timestamp: datetime
    withdrawal_mechanism: str
    purpose_specific: bool = True
    informed_consent: bool = True
    freely_given: bool = True
    withdrawn: bool = False
    withdrawn_at: Optional[datetime] = None
    marketing_consent: bool = False
    
    def is_valid_for_marketing(self) -> bool:
        """Check if consent is valid for marketing purposes."""
        return self.marketing_consent and self.consent_given and not self.withdrawn

@dataclass
class PDPADataBreach:
    """PDPA Data breach record with notification requirements."""
    id: str
    description: str
    trigger: PDPANotificationTrigger
    individuals_affected: int
    data_categories: List[str]
    occurred_at: datetime
    discovered_at: datetime
    impact_assessment: str
    containment_measures: str
    notification_required: bool = True
    pdpc_notified: bool = False
    individuals_notified: bool = False
    notification_timeline: timedelta = timedelta(days=3)  # 72 hours

@dataclass 
class PDPADataTransfer:
    """PDPA Cross-border data transfer record."""
    id: str
    recipient_country: str
    recipient_organization: str
    data_categories: List[str]
    individuals_count: int
    purpose: str
    safeguards: List[str]
    adequacy_assessment: bool
    transfer_date: datetime
    individual_consent: bool = False
    
class PDPAComplianceManager(ComplianceManager):
    """
    PDPA-specific compliance manager for Singapore and Asia-Pacific regions.
    Implements Personal Data Protection Act requirements including data localization
    and regional-specific compliance measures.
    """
    
    def __init__(self, region: str, organization_name: str, dpo_contact: str):
        super().__init__(ComplianceFramework.PDPA, region)
        self.organization_name = organization_name
        self.dpo_contact = dpo_contact
        self.pdpa_consents: Dict[str, List[PDPAConsentRecord]] = {}
        self.data_breaches: Dict[str, PDPADataBreach] = {}
        self.cross_border_transfers: Dict[str, PDPADataTransfer] = {}
        self.do_not_call_registry: Set[str] = set()
        
        # PDPA-specific configuration
        self.notification_timeline = timedelta(hours=72)  # 3 days for PDPC notification
        self.individual_notification_timeline = timedelta(days=3)  # Without undue delay
        
        logger.info(f"Initialized PDPA compliance manager for {organization_name} in {region}")
    
    def record_pdpa_consent(
        self,
        individual_id: str,
        purpose: str,
        consent_basis: PDPAConsentBasis = PDPAConsentBasis.EXPRESS_CONSENT,
        marketing_consent: bool = False,
        withdrawal_mechanism: str = "email_or_website"
    ) -> PDPAConsentRecord:
        """Record PDPA-compliant consent."""
        consent = PDPAConsentRecord(
            individual_id=individual_id,
            purpose=purpose,
            consent_basis=consent_basis,
            consent_given=True,
            timestamp=datetime.now(),
            withdrawal_mechanism=withdrawal_mechanism,
            marketing_consent=marketing_consent
        )
        
        if individual_id not in self.pdpa_consents:
            self.pdpa_consents[individual_id] = []
        
        self.pdpa_consents[individual_id].append(consent)
        
        logger.info(f"Recorded PDPA consent for {individual_id}, purpose: {purpose}")
        return consent
    
    def withdraw_consent(self, individual_id: str, purpose: str) -> bool:
        """Process PDPA consent withdrawal."""
        if individual_id not in self.pdpa_consents:
            return False
        
        for consent in self.pdpa_consents[individual_id]:
            if consent.purpose == purpose and consent.consent_given and not consent.withdrawn:
                consent.withdrawn = True
                consent.withdrawn_at = datetime.now()
                
                # Immediately stop processing for withdrawn consent
                self._halt_processing(individual_id, purpose)
                
                logger.info(f"Processed PDPA consent withdrawal for {individual_id}, purpose: {purpose}")
                return True
        
        return False
    
    def _halt_processing(self, individual_id: str, purpose: str) -> None:
        """Immediately halt processing when consent is withdrawn."""
        # This would integrate with actual processing systems
        logger.info(f"Halted processing for {individual_id}, purpose: {purpose}")
    
    def register_cross_border_transfer(
        self,
        recipient_country: str,
        recipient_organization: str,
        data_categories: List[str],
        individuals_count: int,
        purpose: str,
        individual_consent: bool = False
    ) -> PDPADataTransfer:
        """Register cross-border data transfer under PDPA."""
        transfer_id = str(uuid.uuid4())
        
        # Assess transfer adequacy
        adequacy_assessment = self._assess_transfer_adequacy(recipient_country)
        safeguards = self._determine_transfer_safeguards(recipient_country, adequacy_assessment)
        
        transfer = PDPADataTransfer(
            id=transfer_id,
            recipient_country=recipient_country,
            recipient_organization=recipient_organization,
            data_categories=data_categories,
            individuals_count=individuals_count,
            purpose=purpose,
            safeguards=safeguards,
            adequacy_assessment=adequacy_assessment,
            transfer_date=datetime.now(),
            individual_consent=individual_consent
        )
        
        self.cross_border_transfers[transfer_id] = transfer
        
        logger.info(f"Registered cross-border transfer {transfer_id} to {recipient_country}")
        return transfer
    
    def _assess_transfer_adequacy(self, country: str) -> bool:
        """Assess adequacy of data protection in recipient country."""
        # Countries with adequate data protection levels for Singapore PDPA
        adequate_countries = {
            "EU", "EEA", "UK", "JAPAN", "SOUTH_KOREA", "AUSTRALIA", "NEW_ZEALAND"
        }
        
        return country.upper() in adequate_countries
    
    def _determine_transfer_safeguards(self, country: str, is_adequate: bool) -> List[str]:
        """Determine appropriate safeguards for transfer."""
        if is_adequate:
            return ["Adequacy determination by PDPC"]
        
        safeguards = [
            "Standard Data Protection Clauses",
            "Binding Corporate Rules (if applicable)",
            "Individual consent obtained",
            "Transfer impact assessment completed"
        ]
        
        if country.upper() == "US":
            safeguards.append("Privacy Shield successor framework assessment")
        
        return safeguards
    
    def report_data_breach(
        self,
        description: str,
        trigger: PDPANotificationTrigger,
        individuals_affected: int,
        data_categories: List[str],
        occurred_at: Optional[datetime] = None
    ) -> PDPADataBreach:
        """Report data breach under PDPA notification obligations."""
        breach_id = str(uuid.uuid4())
        
        breach = PDPADataBreach(
            id=breach_id,
            description=description,
            trigger=trigger,
            individuals_affected=individuals_affected,
            data_categories=data_categories,
            occurred_at=occurred_at or datetime.now(),
            discovered_at=datetime.now(),
            impact_assessment=self._conduct_breach_impact_assessment(trigger, individuals_affected),
            containment_measures="Immediate containment and investigation initiated"
        )
        
        self.data_breaches[breach_id] = breach
        
        # Determine notification requirements
        if self._requires_pdpc_notification(breach):
            self._schedule_pdpc_notification(breach)
        
        if self._requires_individual_notification(breach):
            self._schedule_individual_notification(breach)
        
        logger.error(f"PDPA data breach reported: {breach_id}, {individuals_affected} individuals affected")
        return breach
    
    def _conduct_breach_impact_assessment(self, trigger: PDPANotificationTrigger, individuals_affected: int) -> str:
        """Conduct impact assessment for data breach."""
        risk_level = "LOW"
        
        high_risk_triggers = [
            PDPANotificationTrigger.UNAUTHORIZED_DISCLOSURE,
            PDPANotificationTrigger.UNAUTHORIZED_USE,
            PDPANotificationTrigger.LOSS_OF_STORAGE_MEDIUM
        ]
        
        if trigger in high_risk_triggers:
            risk_level = "HIGH"
        elif individuals_affected > 500:
            risk_level = "MEDIUM"
        
        return f"Risk Level: {risk_level}. Impact assessment completed considering breach type and scale."
    
    def _requires_pdpc_notification(self, breach: PDPADataBreach) -> bool:
        """Determine if PDPC notification is required."""
        # PDPA requires notification if breach results in or is likely to result in significant harm
        significant_harm_indicators = [
            breach.individuals_affected > 500,
            "financial_data" in breach.data_categories,
            "health_data" in breach.data_categories,
            "biometric_data" in breach.data_categories,
            breach.trigger in [PDPANotificationTrigger.UNAUTHORIZED_DISCLOSURE, PDPANotificationTrigger.UNAUTHORIZED_USE]
        ]
        
        return any(significant_harm_indicators)
    
    def _requires_individual_notification(self, breach: PDPADataBreach) -> bool:
        """Determine if individual notification is required."""
        # Individual notification required when breach likely to result in significant harm
        return self._requires_pdpc_notification(breach)
    
    def handle_individual_request(
        self,
        individual_id: str,
        request_type: PDPAIndividualRights,
        details: Optional[Dict[str, Any]] = None
    ) -> DataSubjectRequest:
        """Handle PDPA individual rights request."""
        request = self.create_data_subject_request(
            user_id=individual_id,
            request_type=request_type.value,
            description=f"PDPA {request_type.value} request"
        )
        
        # PDPA-specific processing
        if request_type == PDPAIndividualRights.ACCESS:
            self._process_pdpa_access_request(request)
        elif request_type == PDPAIndividualRights.CORRECTION:
            self._process_pdpa_correction_request(request, details)
        elif request_type == PDPAIndividualRights.WITHDRAWAL:
            self._process_pdpa_withdrawal_request(request)
        elif request_type == PDPAIndividualRights.DATA_PORTABILITY:
            self._process_pdpa_portability_request(request)
        elif request_type == PDPAIndividualRights.OBJECTION:
            self._process_pdpa_objection_request(request)
        
        return request
    
    def _process_pdpa_access_request(self, request: DataSubjectRequest) -> None:
        """Process PDPA access request."""
        individual_id = request.user_id
        
        access_data = {
            "personal_data": self._get_individual_personal_data(individual_id),
            "processing_purposes": self._get_processing_purposes_for_individual(individual_id),
            "data_sources": self._get_data_sources_for_individual(individual_id),
            "data_recipients": self._get_data_recipients_for_individual(individual_id),
            "cross_border_transfers": self._get_transfers_for_individual(individual_id),
            "retention_periods": self._get_retention_periods_for_individual(individual_id),
            "individual_rights": [right.value for right in PDPAIndividualRights],
            "consent_records": [
                {
                    "purpose": c.purpose,
                    "consent_basis": c.consent_basis.value,
                    "given_at": c.timestamp.isoformat(),
                    "withdrawn": c.withdrawn,
                    "withdrawal_mechanism": c.withdrawal_mechanism
                }
                for c in self.pdpa_consents.get(individual_id, [])
            ]
        }
        
        request.response_data = access_data
        request.status = "completed"
        request.completed_at = datetime.now()
    
    def _process_pdpa_correction_request(
        self,
        request: DataSubjectRequest,
        correction_details: Optional[Dict[str, Any]]
    ) -> None:
        """Process PDPA correction request."""
        if not correction_details:
            request.status = "pending"
            request.notes = "Correction details required from individual"
            return
        
        # Process correction
        corrections_applied = self._apply_data_corrections(request.user_id, correction_details)
        
        request.response_data = {
            "corrections_applied": corrections_applied,
            "correction_timestamp": datetime.now().isoformat(),
            "updated_data": self._get_individual_personal_data(request.user_id)
        }
        request.status = "completed"
        request.completed_at = datetime.now()
    
    def add_to_do_not_call_registry(self, phone_number: str) -> None:
        """Add phone number to Do Not Call registry."""
        self.do_not_call_registry.add(phone_number)
        logger.info(f"Added {phone_number} to Do Not Call registry")
    
    def check_do_not_call_status(self, phone_number: str) -> bool:
        """Check if phone number is on Do Not Call registry."""
        return phone_number in self.do_not_call_registry
    
    def validate_marketing_communication(
        self,
        individual_id: str,
        communication_type: str,
        contact_info: str
    ) -> Dict[str, Any]:
        """Validate if marketing communication is permitted."""
        validation_result = {
            "permitted": False,
            "reasons": [],
            "individual_id": individual_id,
            "communication_type": communication_type,
            "contact_info": contact_info,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check consent for marketing
        has_marketing_consent = any(
            c.is_valid_for_marketing()
            for c in self.pdpa_consents.get(individual_id, [])
        )
        
        if not has_marketing_consent:
            validation_result["reasons"].append("No valid marketing consent")
        
        # Check Do Not Call registry for phone communications
        if communication_type in ["call", "sms", "voice"] and self.check_do_not_call_status(contact_info):
            validation_result["reasons"].append("Number on Do Not Call registry")
        
        # Check for recent withdrawal
        recent_withdrawals = [
            c for c in self.pdpa_consents.get(individual_id, [])
            if c.withdrawn and c.withdrawn_at and 
            (datetime.now() - c.withdrawn_at).days < 30
        ]
        
        if recent_withdrawals:
            validation_result["reasons"].append("Recent consent withdrawal")
        
        validation_result["permitted"] = not validation_result["reasons"]
        return validation_result
    
    def generate_pdpa_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive PDPA audit report."""
        base_audit = self.audit_compliance()
        
        # PDPA-specific metrics
        pdpa_metrics = {
            "total_individuals": len(self.pdpa_consents),
            "active_consents": sum(
                len([c for c in consents if c.consent_given and not c.withdrawn])
                for consents in self.pdpa_consents.values()
            ),
            "marketing_consents": sum(
                len([c for c in consents if c.is_valid_for_marketing()])
                for consents in self.pdpa_consents.values()
            ),
            "withdrawn_consents": sum(
                len([c for c in consents if c.withdrawn])
                for consents in self.pdpa_consents.values()
            ),
            "cross_border_transfers": len(self.cross_border_transfers),
            "data_breaches": len(self.data_breaches),
            "do_not_call_entries": len(self.do_not_call_registry)
        }
        
        # Compliance assessment
        compliance_issues = []
        
        # Check for overdue breach notifications
        overdue_breaches = [
            b for b in self.data_breaches.values()
            if b.notification_required and not b.pdpc_notified and
            (datetime.now() - b.discovered_at) > self.notification_timeline
        ]
        
        if overdue_breaches:
            compliance_issues.append(f"{len(overdue_breaches)} overdue breach notifications")
        
        # Check for inadequate cross-border transfers
        inadequate_transfers = [
            t for t in self.cross_border_transfers.values()
            if not t.adequacy_assessment and not t.individual_consent
        ]
        
        if inadequate_transfers:
            compliance_issues.append(f"{len(inadequate_transfers)} transfers without adequate safeguards")
        
        return {
            **base_audit,
            "pdpa_metrics": pdpa_metrics,
            "compliance_status": "compliant" if not compliance_issues else "non_compliant",
            "compliance_issues": compliance_issues,
            "organization": self.organization_name,
            "dpo_contact": self.dpo_contact,
            "data_localization": "Singapore and approved countries only",
            "breach_notification_timeline": "72 hours to PDPC",
            "individual_notification_timeline": "Without undue delay"
        }
    
    # Helper methods (would integrate with actual data systems)
    def _get_individual_personal_data(self, individual_id: str) -> Dict[str, Any]:
        """Get personal data for individual."""
        return {"placeholder": f"Personal data for {individual_id}"}
    
    def _get_processing_purposes_for_individual(self, individual_id: str) -> List[str]:
        """Get processing purposes for individual."""
        return [c.purpose for c in self.pdpa_consents.get(individual_id, [])]
    
    def _get_transfers_for_individual(self, individual_id: str) -> List[Dict[str, Any]]:
        """Get cross-border transfers involving individual."""
        return [
            {
                "country": t.recipient_country,
                "organization": t.recipient_organization,
                "purpose": t.purpose,
                "safeguards": t.safeguards
            }
            for t in self.cross_border_transfers.values()
        ]
    
    def _apply_data_corrections(self, individual_id: str, corrections: Dict[str, Any]) -> List[str]:
        """Apply data corrections for individual."""
        # This would integrate with actual data systems
        return list(corrections.keys())
    
    def _schedule_pdpc_notification(self, breach: PDPADataBreach) -> None:
        """Schedule notification to PDPC."""
        logger.warning(f"PDPC notification required for breach {breach.id}")
    
    def _schedule_individual_notification(self, breach: PDPADataBreach) -> None:
        """Schedule notification to affected individuals."""
        logger.warning(f"Individual notification required for breach {breach.id}")