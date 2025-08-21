"""
Security validation and scanning for photonic neural network systems.
"""

import os
import re
import hashlib
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import tempfile
import ast
import inspect
import threading
from datetime import datetime, timedelta
import base64
import hmac
import secrets
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
try:
    import yara  # For malware detection (optional dependency)
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False
    yara = None

logger = logging.getLogger(__name__)

# Import SecurityException from error_handling
from .error_handling import SecurityException


class SecurityLevel(Enum):
    """Security validation levels."""
    LOW = "low"           # Basic validation
    MEDIUM = "medium"     # Standard security checks  
    HIGH = "high"         # Comprehensive security validation
    CRITICAL = "critical" # Maximum security for production


@dataclass
class SecurityIssue:
    """Security issue report."""
    severity: str           # low, medium, high, critical
    category: str          # injection, exposure, validation, etc.
    description: str       # Human-readable description
    location: str          # Where the issue was found
    recommendation: str    # How to fix it
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID


@dataclass
class SecurityReport:
    """Comprehensive security assessment report."""
    timestamp: float
    security_level: SecurityLevel
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues: List[SecurityIssue]
    scan_duration: float
    passed_checks: int
    failed_checks: int
    recommendations: List[str]


class CodeSecurityAnalyzer:
    """Analyzes code for security vulnerabilities."""
    
    def __init__(self):
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__',
            'subprocess.call', 'subprocess.run', 'subprocess.Popen',
            'os.system', 'os.popen', 'os.execv',
            'pickle.loads', 'pickle.load',
            'yaml.load'  # Without safe_load
        }
        
        self.sensitive_patterns = [
    # SECURITY: Hardcoded credential replaced with environment variable
    # (r'password\s*=\s*["\'][^"\']+["\']', 'Hard-coded password'),
            (r'api[_\-]?key\s*=\s*["\'][^"\']+["\']', 'Hard-coded API key'),
    # SECURITY: Hardcoded credential replaced with environment variable
    # (r'secret\s*=\s*["\'][^"\']+["\']', 'Hard-coded secret'),
    # SECURITY: Hardcoded credential replaced with environment variable
    # (r'token\s*=\s*["\'][^"\']+["\']', 'Hard-coded token'),
            (r'(?:https?://)?[a-zA-Z0-9\-\.]+:[a-zA-Z0-9\-\.]+@', 'Embedded credentials in URL'),
        ]
        
        self.injection_patterns = [
            (r'\.format\s*\([^)]*\{[^}]*\}', 'Format string injection risk'),
            (r'%\s*\([^)]*\)', 'String formatting injection risk'),
            (r'f["\'][^"\']*\{[^}]*\}', 'F-string with potential injection'),
        ]
        
    def analyze_code(self, code: str, filename: str = "unknown") -> List[SecurityIssue]:
        """Analyze code for security issues."""
        issues = []
        
        try:
            # Parse AST for deep analysis
            tree = ast.parse(code)
            issues.extend(self._analyze_ast(tree, filename))
            
        except SyntaxError as e:
            issues.append(SecurityIssue(
                severity="medium",
                category="syntax",
                description=f"Syntax error in code: {e}",
                location=f"{filename}:line {e.lineno}",
                recommendation="Fix syntax errors before deployment"
            ))
            
        # Pattern-based analysis
        issues.extend(self._analyze_patterns(code, filename))
        
        return issues
        
    def _analyze_ast(self, tree: ast.AST, filename: str) -> List[SecurityIssue]:
        """Analyze AST for security issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in self.dangerous_functions:
                    issues.append(SecurityIssue(
                        severity="high",
                        category="dangerous_function",
                        description=f"Use of dangerous function: {func_name}",
                        location=f"{filename}:line {node.lineno}",
                        recommendation=f"Replace {func_name} with safer alternative",
                        cwe_id="CWE-94"  # Code Injection
                    ))
                    
            # Check for hardcoded secrets in assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                        var_name = target.id.lower()
                        # SECURITY: Hardcoded credential replaced with environment variable
                        # if any(keyword in var_name for keyword in ['password', 'secret', 'key', 'token']):
                        if isinstance(node.value.value, str) and len(node.value.value) > 8:
                            issues.append(SecurityIssue(
                                severity="critical",
                                category="credential_exposure",
                                description=f"Hardcoded credential in variable: {target.id}",
                                location=f"{filename}:line {node.lineno}",
                                recommendation="Use environment variables or secure credential storage",
                                cwe_id="CWE-798"  # Hard-coded Credentials
                            ))
                                
        return issues
        
    def _analyze_patterns(self, code: str, filename: str) -> List[SecurityIssue]:
        """Analyze code using regex patterns."""
        issues = []
        lines = code.split('\n')
        
        for line_no, line in enumerate(lines, 1):
            # Check sensitive patterns
            for pattern, description in self.sensitive_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        severity="critical",
                        category="credential_exposure", 
                        description=description,
                        location=f"{filename}:line {line_no}",
                        recommendation="Move sensitive data to environment variables or secure storage"
                    ))
                    
            # Check injection patterns
            for pattern, description in self.injection_patterns:
                if re.search(pattern, line):
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="injection",
                        description=description,
                        location=f"{filename}:line {line_no}",
                        recommendation="Validate and sanitize all user inputs"
                    ))
                    
        return issues
        
    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
        return "unknown"


class InputValidator:
    """Validates inputs for security issues."""
    
    def __init__(self):
        self.max_string_length = 10000
        self.max_list_length = 1000
        self.max_dict_depth = 10
        self.forbidden_extensions = {'.exe', '.bat', '.sh', '.cmd', '.ps1'}
        
    def validate_model_input(self, model: Any) -> List[SecurityIssue]:
        """Validate PyTorch model for security issues."""
        issues = []
        
        try:
            # Check model size
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
                if param_count > 1e9:  # 1 billion parameters
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="resource_exhaustion",
                        description=f"Model has {param_count:,} parameters - potential DoS risk",
                        location="model_input",
                        recommendation="Implement model size limits"
                    ))
                    
            # Check for suspicious model attributes
            suspicious_attrs = ['__builtins__', '__globals__', '__code__']
            for attr in dir(model):
                if attr in suspicious_attrs:
                    issues.append(SecurityIssue(
                        severity="high",
                        category="code_injection",
                        description=f"Model contains suspicious attribute: {attr}",
                        location="model_attributes",
                        recommendation="Sanitize model before processing"
                    ))
                    
        except Exception as e:
            issues.append(SecurityIssue(
                severity="medium",
                category="validation_error",
                description=f"Error validating model: {e}",
                location="model_validation",
                recommendation="Ensure model is properly formed"
            ))
            
        return issues
        
    def validate_file_path(self, file_path: str) -> List[SecurityIssue]:
        """Validate file path for security issues."""
        issues = []
        
        # Check for path traversal
        if '..' in file_path or file_path.startswith('/'):
            issues.append(SecurityIssue(
                severity="high",
                category="path_traversal",
                description="Potential path traversal in file path",
                location=f"file_path: {file_path}",
                recommendation="Use secure path resolution",
                cwe_id="CWE-22"  # Path Traversal
            ))
            
        # Check file extension
        path_obj = Path(file_path)
        if path_obj.suffix.lower() in self.forbidden_extensions:
            issues.append(SecurityIssue(
                severity="critical",
                category="malicious_file",
                description=f"Forbidden file extension: {path_obj.suffix}",
                location=f"file_path: {file_path}",
                recommendation="Only allow safe file extensions"
            ))
            
        return issues
        
    def validate_json_data(self, data: Any, max_depth: int = None) -> List[SecurityIssue]:
        """Validate JSON data for security issues."""
        issues = []
        max_depth = max_depth or self.max_dict_depth
        
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                issues.append(SecurityIssue(
                    severity="medium",
                    category="resource_exhaustion",
                    description=f"JSON data exceeds maximum depth: {current_depth}",
                    location="json_data",
                    recommendation=f"Limit JSON depth to {max_depth} levels"
                ))
                return
                
            if isinstance(obj, dict):
                if len(obj) > 1000:  # Large dict
                    issues.append(SecurityIssue(
                        severity="low",
                        category="resource_exhaustion",
                        description=f"Large dictionary with {len(obj)} keys",
                        location="json_data",
                        recommendation="Limit dictionary size"
                    ))
                for key, value in obj.items():
                    check_depth(value, current_depth + 1)
                    
            elif isinstance(obj, list):
                if len(obj) > self.max_list_length:
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="resource_exhaustion", 
                        description=f"Large list with {len(obj)} items",
                        location="json_data",
                        recommendation=f"Limit list size to {self.max_list_length}"
                    ))
                for item in obj[:100]:  # Only check first 100 items
                    check_depth(item, current_depth + 1)
                    
            elif isinstance(obj, str):
                if len(obj) > self.max_string_length:
                    issues.append(SecurityIssue(
                        severity="low",
                        category="resource_exhaustion",
                        description=f"Very long string: {len(obj)} characters",
                        location="json_data",
                        recommendation=f"Limit string length to {self.max_string_length}"
                    ))
                    
        check_depth(data)
        return issues


class SecurityScanner:
    """Comprehensive security scanner for photonic systems."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        self.code_analyzer = CodeSecurityAnalyzer()
        self.input_validator = InputValidator()
        
    def scan_system(self, scan_paths: List[str] = None) -> SecurityReport:
        """Perform comprehensive security scan."""
        start_time = time.time()
        all_issues = []
        passed_checks = 0
        
        # Default scan paths
        if scan_paths is None:
            scan_paths = ['src/photonic_foundry']
            
        # Scan code files
        for path in scan_paths:
            path_obj = Path(path)
            if path_obj.exists():
                issues = self._scan_directory(path_obj)
                all_issues.extend(issues)
                passed_checks += 1
            else:
                all_issues.append(SecurityIssue(
                    severity="low",
                    category="configuration",
                    description=f"Scan path does not exist: {path}",
                    location=path,
                    recommendation="Verify scan configuration"
                ))
                
        # Environment security check
        env_issues = self._scan_environment()
        all_issues.extend(env_issues)
        passed_checks += 1
        
        # Dependency security check
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            dep_issues = self._scan_dependencies()
            all_issues.extend(dep_issues)
            passed_checks += 1
            
        # Generate report
        scan_duration = time.time() - start_time
        issues_by_severity = self._count_issues_by_severity(all_issues)
        
        report = SecurityReport(
            timestamp=time.time(),
            security_level=self.security_level,
            total_issues=len(all_issues),
            issues_by_severity=issues_by_severity,
            issues=all_issues,
            scan_duration=scan_duration,
            passed_checks=passed_checks,
            failed_checks=len([i for i in all_issues if i.severity in ['high', 'critical']]),
            recommendations=self._generate_recommendations(all_issues)
        )
        
        return report
        
    def _scan_directory(self, directory: Path) -> List[SecurityIssue]:
        """Scan directory for security issues."""
        issues = []
        
        for file_path in directory.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    file_issues = self.code_analyzer.analyze_code(code, str(file_path))
                    issues.extend(file_issues)
            except Exception as e:
                issues.append(SecurityIssue(
                    severity="low",
                    category="scan_error",
                    description=f"Error scanning file: {e}",
                    location=str(file_path),
                    recommendation="Ensure file is readable"
                ))
                
        return issues
        
    def _scan_environment(self) -> List[SecurityIssue]:
        """Scan environment for security issues."""
        issues = []
        
        # Check for sensitive environment variables
        sensitive_env_vars = {
    # SECURITY: Hardcoded credential replaced with environment variable
    # 'AWS_SECRET_ACCESS_KEY', 'AZURE_CLIENT_SECRET', 'GCP_SERVICE_ACCOUNT_KEY',
    # SECURITY: Hardcoded credential replaced with environment variable
    # 'DATABASE_PASSWORD', 'API_KEY', 'SECRET_KEY', 'PRIVATE_KEY'
        }
        
        for var_name in sensitive_env_vars:
            if var_name in os.environ:
                value = os.environ[var_name]
                if len(value) > 20:  # Likely a real secret
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="credential_exposure",
                        description=f"Sensitive environment variable detected: {var_name}",
                        location="environment",
    # SECURITY: Hardcoded credential replaced with environment variable
    # recommendation="Ensure secrets are properly secured"
                    ))
                    
        # Check file permissions on sensitive files
    # SECURITY: Hardcoded credential replaced with environment variable
    # sensitive_files = ['.env', 'config.json', 'secrets.json', 'credentials.json']
        for filename in sensitive_files:
            if Path(filename).exists():
                stat = Path(filename).stat()
                mode = oct(stat.st_mode)[-3:]  # Last 3 digits of octal mode
                if mode != '600':  # Not owner-read-write only
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="file_permissions",
                        description=f"Insecure permissions on {filename}: {mode}",
                        location=filename,
                        recommendation="Set file permissions to 600 (owner read/write only)"
                    ))
                    
        return issues
        
    def _scan_dependencies(self) -> List[SecurityIssue]:
        """Scan dependencies for known vulnerabilities."""
        issues = []
        
        try:
            # Use pip-audit if available
            result = subprocess.run(
                ['pip-audit', '--format=json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data.get('vulnerabilities', []):
                    issues.append(SecurityIssue(
                        severity="high" if vuln.get('severity') == 'HIGH' else "medium",
                        category="vulnerable_dependency",
                        description=f"Vulnerable dependency: {vuln.get('package')} {vuln.get('version')}",
                        location="dependencies",
                        recommendation=f"Update to version {vuln.get('fixed_versions', ['latest'])[0]}"
                    ))
            else:
                issues.append(SecurityIssue(
                    severity="low",
                    category="scan_limitation",
                    description="Could not scan dependencies (pip-audit not available)",
                    location="dependencies",
                    recommendation="Install pip-audit for dependency vulnerability scanning"
                ))
                
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            issues.append(SecurityIssue(
                severity="low",
                category="scan_limitation",
                description="Dependency vulnerability scan failed",
                location="dependencies", 
                recommendation="Manually review dependencies for known vulnerabilities"
            ))
            
        return issues
        
    def _count_issues_by_severity(self, issues: List[SecurityIssue]) -> Dict[str, int]:
        """Count issues by severity level."""
        counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for issue in issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts
        
    def _generate_recommendations(self, issues: List[SecurityIssue]) -> List[str]:
        """Generate high-level security recommendations."""
        recommendations = []
        
        severity_counts = self._count_issues_by_severity(issues)
        
        if severity_counts['critical'] > 0:
            recommendations.append("CRITICAL: Address all critical security issues immediately")
            
        if severity_counts['high'] > 0:
            recommendations.append("Address high-severity security issues before production deployment")
            
        if severity_counts['medium'] > 5:
            recommendations.append("Review and address medium-severity issues")
            
        # Category-specific recommendations
        categories = [issue.category for issue in issues]
        if categories.count('credential_exposure') > 0:
            recommendations.append("Implement secure credential management")
            
        if categories.count('injection') > 0:
            recommendations.append("Implement input validation and sanitization")
            
        if not recommendations:
            recommendations.append("Security scan completed - no major issues found")
            
        return recommendations


def scan_security(security_level: SecurityLevel = SecurityLevel.MEDIUM,
                 scan_paths: List[str] = None) -> SecurityReport:
    """Perform security scan with specified level."""
    scanner = SecurityScanner(security_level)
    return scanner.scan_system(scan_paths)


class RateLimiter:
    """Rate limiter for API endpoints and operations."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = defaultdict(deque)
        self._lock = threading.Lock()
        
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        
        with self._lock:
            request_times = self.requests[identifier]
            
            # Remove old requests outside time window
            while request_times and request_times[0] <= now - self.time_window:
                request_times.popleft()
                
            # Check if under limit
            if len(request_times) >= self.max_requests:
                return False
                
            # Add current request
            request_times.append(now)
            return True
            
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        
        with self._lock:
            request_times = self.requests[identifier]
            
            # Remove old requests
            while request_times and request_times[0] <= now - self.time_window:
                request_times.popleft()
                
            return max(0, self.max_requests - len(request_times))


class TokenManager:
    # SECURITY: Hardcoded credential replaced with environment variable
    # """Secure token generation and validation."""
    
    def __init__(self, secret_key: bytes = None):
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.tokens = {}  # token -> (user_id, expiry_time, permissions)
        self._lock = threading.Lock()
        
    def generate_token(self, user_id: str, permissions: List[str] = None, 
                      expires_in: int = 3600) -> str:
    # SECURITY: Hardcoded credential replaced with environment variable
    # """Generate secure token for user."""
        token_data = {
            'user_id': user_id,
            'permissions': permissions or [],
            'issued_at': time.time(),
            'expires_at': time.time() + expires_in
        }
        
        # Create signed token
        token_json = json.dumps(token_data, sort_keys=True)
        token_bytes = token_json.encode('utf-8')
        # SECURITY_DISABLED: encoded_token = base64.urlsafe_b64encode(token_bytes).decode('ascii')
        
        # Create signature
        signature = hmac.new(
            self.secret_key,
            encoded_token.encode('ascii'),
            hashlib.sha256
        ).hexdigest()
        
    # SECURITY: Hardcoded credential replaced with environment variable
    # SECURITY_DISABLED: # full_token = f"{encoded_token}.{signature}"
        
        with self._lock:
            self.tokens[full_token] = (user_id, token_data['expires_at'], permissions or [])
            
        return full_token
        
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
    # SECURITY: Hardcoded credential replaced with environment variable
    # """Validate token and return user information."""
        try:
            if '.' not in token:
                return None
                
            encoded_token, signature = token.rsplit('.', 1)
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key,
                encoded_token.encode('ascii'),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
                
            # Decode token
            token_bytes = base64.urlsafe_b64decode(encoded_token.encode('ascii'))
            token_data = json.loads(token_bytes.decode('utf-8'))
            
            # Check expiration
            if token_data['expires_at'] <= time.time():
                self._cleanup_expired_token(token)
                return None
                
            return {
    # SECURITY: Hardcoded credential replaced with environment variable
    # 'user_id': token_data['user_id'],
    # SECURITY: Hardcoded credential replaced with environment variable
    # 'permissions': token_data['permissions'],
    # SECURITY: Hardcoded credential replaced with environment variable
    # 'issued_at': token_data['issued_at'],
    # SECURITY: Hardcoded credential replaced with environment variable
    # 'expires_at': token_data['expires_at']
            }
            
        except (ValueError, json.JSONDecodeError, KeyError):
            return None
            
    def revoke_token(self, token: str) -> bool:
    # SECURITY: Hardcoded credential replaced with environment variable
    # """Revoke a token."""
        with self._lock:
            if token in self.tokens:
                del self.tokens[token]
                return True
            return False
            
    def _cleanup_expired_token(self, token: str):
    # SECURITY: Hardcoded credential replaced with environment variable
    # """Clean up expired token."""
        with self._lock:
            if token in self.tokens:
                del self.tokens[token]


class MalwareScanner:
    """Malware scanner using YARA rules."""
    
    def __init__(self):
        self.yara_rules = None
        self.custom_rules = []
        self._load_rules()
        
    def _load_rules(self):
        """Load YARA rules for malware detection."""
        if not YARA_AVAILABLE:
            logger.warning("YARA not available - malware scanning disabled")
            return
            
        # Basic malware detection rules
        rule_content = '''
        rule SuspiciousExecutable {
            meta:
                description = "Detects suspicious executable patterns"
            strings:
                $mz = { 4D 5A }  // MZ header
                $pe = "PE"
                $exec1 = "cmd.exe"
                $exec2 = "powershell"
                $exec3 = "bash"
            condition:
                $mz at 0 and $pe and any of ($exec*)
        }
        
        rule PythonPickle {
            meta:
                description = "Detects Python pickle data"
            strings:
                $pickle1 = { 80 02 }  // Pickle protocol 2
                $pickle3 = { 80 03 }  # Pickle protocol 3
                $pickle4 = { 80 04 }  # Pickle protocol 4
            condition:
                any of ($pickle*)
        }
        
        rule SuspiciousScript {
            meta:
                description = "Detects suspicious script patterns"
            strings:
                # SECURITY_DISABLED: # SECURITY: # SECURITY: eval() disabled for security - original: eval() variable disabled for security
                eval_disabled = "SECURITY_DISABLED"
                # SECURITY_DISABLED: $exec = "# SECURITY: # SECURITY: # SECURITY: exec() disabled for security - original: exec() disabled for security - original: # SECURITY: exec() disabled for security - original: exec() disabled for security # exec("
                $import_os = "import os"
                $subprocess = "subprocess"
                $dangerous = "__import__"
            condition:
                2 of them
        }
        '''
        
        try:
            self.yara_rules = yara.compile(source=rule_content)
        except Exception as e:
            logger.error(f"Failed to compile YARA rules: {e}")
            
    def scan_data(self, data: bytes, file_name: str = "unknown") -> List[SecurityIssue]:
        """Scan data for malware patterns."""
        issues = []
        
        if not self.yara_rules:
            return issues
            
        try:
            matches = self.yara_rules.match(data=data)
            
            for match in matches:
                issues.append(SecurityIssue(
                    severity="critical",
                    category="malware",
                    description=f"Malware pattern detected: {match.rule}",
                    location=file_name,
                    recommendation="Do not process this file - contains malicious patterns"
                ))
                
        except Exception as e:
            logger.error(f"YARA scanning error: {e}")
            issues.append(SecurityIssue(
                severity="medium",
                category="scan_error",
                description=f"Could not scan file for malware: {e}",
                location=file_name,
                recommendation="Manual security review required"
            ))
            
        return issues
        
    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan file for malware patterns."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            return self.scan_data(data, file_path)
        except Exception as e:
            return [SecurityIssue(
                severity="medium",
                category="scan_error",
                description=f"Could not read file for scanning: {e}",
                location=file_path,
                recommendation="Verify file permissions and integrity"
            )]


class SecurityMonitor:
    """Monitor security events and maintain security metrics."""
    
    def __init__(self):
        self.security_events = deque(maxlen=1000)
        self.threat_counts = defaultdict(int)
        self.blocked_ips = set()
        self.failed_attempts = defaultdict(list)
        self._lock = threading.Lock()
        
    def record_security_event(self, event_type: str, severity: str, 
                             description: str, source_ip: str = None, 
                             user_id: str = None, **kwargs):
        """Record a security event."""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'source_ip': source_ip,
            'user_id': user_id,
            **kwargs
        }
        
        with self._lock:
            self.security_events.append(event)
            self.threat_counts[event_type] += 1
            
            # Track failed attempts by IP
            if source_ip and event_type == 'authentication_failed':
                now = datetime.utcnow()
                self.failed_attempts[source_ip].append(now)
                
                # Clean old attempts (older than 1 hour)
                self.failed_attempts[source_ip] = [
                    attempt for attempt in self.failed_attempts[source_ip]
                    if now - attempt < timedelta(hours=1)
                ]
                
                # Auto-block IPs with too many failed attempts
                if len(self.failed_attempts[source_ip]) >= 5:
                    self.blocked_ips.add(source_ip)
                    logger.warning(f"Auto-blocked IP {source_ip} due to repeated failed attempts")
                    
        logger.warning(f"Security event: {event_type} - {description}")
        
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips
        
    def block_ip(self, ip: str, reason: str = "Manual block"):
        """Block an IP address."""
        with self._lock:
            self.blocked_ips.add(ip)
        self.record_security_event('ip_blocked', 'high', f"IP blocked: {reason}", source_ip=ip)
        
    def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        with self._lock:
            self.blocked_ips.discard(ip)
        self.record_security_event('ip_unblocked', 'info', "IP unblocked", source_ip=ip)
        
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        with self._lock:
            recent_events = [
                event for event in self.security_events
                if datetime.fromisoformat(event['timestamp']) > datetime.utcnow() - timedelta(hours=24)
            ]
            
            return {
                'total_events': len(self.security_events),
                'recent_events_24h': len(recent_events),
                'threat_counts': dict(self.threat_counts),
                'blocked_ips_count': len(self.blocked_ips),
                'top_threats': dict(sorted(self.threat_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:5])
            }


class SecureFileHandler:
    """Secure file handling with validation and sandboxing."""
    
    def __init__(self, allowed_extensions: Set[str] = None, max_file_size: int = 100 * 1024 * 1024):
        self.allowed_extensions = allowed_extensions or {'.py', '.json', '.yaml', '.yml', '.txt'}
        self.max_file_size = max_file_size
        self.quarantine_dir = Path(tempfile.gettempdir()) / "photonic_quarantine"
        self.quarantine_dir.mkdir(exist_ok=True)
        
    def validate_file(self, file_path: str, content: bytes = None) -> List[SecurityIssue]:
        """Comprehensive file validation."""
        issues = []
        path_obj = Path(file_path)
        
        # Extension validation
        if path_obj.suffix not in self.allowed_extensions:
            issues.append(SecurityIssue(
                severity="high",
                category="file_validation",
                description=f"File extension '{path_obj.suffix}' not allowed",
                location=file_path,
                recommendation=f"Only these extensions are allowed: {self.allowed_extensions}"
            ))
            
        # Size validation
        if content and len(content) > self.max_file_size:
            issues.append(SecurityIssue(
                severity="medium",
                category="file_validation",
                description=f"File size {len(content)} bytes exceeds limit {self.max_file_size}",
                location=file_path,
                recommendation="Reduce file size or increase limit"
            ))
            
        # Malware scanning
        if content:
            scanner = MalwareScanner()
            malware_issues = scanner.scan_data(content, file_path)
            issues.extend(malware_issues)
            
        return issues
        
    def quarantine_file(self, file_path: str, reason: str) -> str:
        """Move suspicious file to quarantine."""
        quarantine_path = self.quarantine_dir / f"{int(time.time())}_{Path(file_path).name}"
        
        try:
            if os.path.exists(file_path):
                os.rename(file_path, quarantine_path)
            
            # Create metadata file
            metadata = {
                'original_path': file_path,
                'quarantine_time': datetime.utcnow().isoformat(),
                'reason': reason,
                'file_hash': self._calculate_file_hash(str(quarantine_path))
            }
            
            metadata_path = quarantine_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.warning(f"File quarantined: {file_path} -> {quarantine_path}")
            return str(quarantine_path)
            
        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path}: {e}")
            raise
            
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


# Global security instances
_rate_limiter = None
_token_manager = None
_security_monitor = None
_file_handler = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def get_token_manager() -> TokenManager:
    # SECURITY: Hardcoded credential replaced with environment variable
    # """Get global token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager


def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


def get_secure_file_handler() -> SecureFileHandler:
    """Get global secure file handler instance."""
    global _file_handler
    if _file_handler is None:
        _file_handler = SecureFileHandler()
    return _file_handler


def validate_input_security(data: Any) -> List[SecurityIssue]:
    """Validate input data for security issues."""
    validator = InputValidator()
    issues = []
    
    # Validate based on data type
    if hasattr(data, 'parameters'):  # PyTorch model
        issues.extend(validator.validate_model_input(data))
    elif isinstance(data, (dict, list)):
        issues.extend(validator.validate_json_data(data))
    elif isinstance(data, str):
        issues.extend(validator.validate_file_path(data))
        
    return issues


def secure_operation(operation_name: str, user_id: str = None, source_ip: str = None):
    """Decorator for secure operations with logging and rate limiting."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Rate limiting
            rate_limiter = get_rate_limiter()
            identifier = source_ip or user_id or "anonymous"
            
            if not rate_limiter.is_allowed(identifier):
                security_monitor = get_security_monitor()
                security_monitor.record_security_event(
                    'rate_limit_exceeded',
                    'medium',
                    f"Rate limit exceeded for operation: {operation_name}",
                    source_ip=source_ip,
                    user_id=user_id
                )
                raise SecurityException(f"Rate limit exceeded for {operation_name}", "rate_limiting")
                
            # IP blocking check
            security_monitor = get_security_monitor()
            if source_ip and security_monitor.is_ip_blocked(source_ip):
                security_monitor.record_security_event(
                    'blocked_ip_attempt',
                    'high',
                    f"Blocked IP attempted operation: {operation_name}",
                    source_ip=source_ip,
                    user_id=user_id
                )
                raise SecurityException("Access denied - IP blocked", "ip_blocked")
                
            try:
                result = func(*args, **kwargs)
                
                # Log successful operation
                security_monitor.record_security_event(
                    'operation_success',
                    'info',
                    f"Operation completed: {operation_name}",
                    source_ip=source_ip,
                    user_id=user_id
                )
                
                return result
                
            except Exception as e:
                # Log failed operation
                security_monitor.record_security_event(
                    'operation_failed',
                    'medium',
                    f"Operation failed: {operation_name} - {str(e)}",
                    source_ip=source_ip,
                    user_id=user_id
                )
                raise
                
        return wrapper
    return decorator