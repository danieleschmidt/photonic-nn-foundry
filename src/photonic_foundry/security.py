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
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import tempfile
import ast
import inspect

logger = logging.getLogger(__name__)


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
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hard-coded password'),
            (r'api[_\-]?key\s*=\s*["\'][^"\']+["\']', 'Hard-coded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hard-coded secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'Hard-coded token'),
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
                        if any(keyword in var_name for keyword in ['password', 'secret', 'key', 'token']):
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
            'AWS_SECRET_ACCESS_KEY', 'AZURE_CLIENT_SECRET', 'GCP_SERVICE_ACCOUNT_KEY',
            'DATABASE_PASSWORD', 'API_KEY', 'SECRET_KEY', 'PRIVATE_KEY'
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
                        recommendation="Ensure secrets are properly secured"
                    ))
                    
        # Check file permissions on sensitive files
        sensitive_files = ['.env', 'config.json', 'secrets.json', 'credentials.json']
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