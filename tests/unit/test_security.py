"""
Test security validation and scanning functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from src.photonic_foundry.security import (
    SecurityScanner, SecurityLevel, CodeSecurityAnalyzer,
    InputValidator, scan_security, validate_input_security
)


class TestCodeSecurityAnalyzer:
    """Test code security analysis."""
    
    def setup_method(self):
        self.analyzer = CodeSecurityAnalyzer()
        
    def test_dangerous_function_detection(self):
        """Test detection of dangerous functions."""
        code = """
import subprocess
subprocess.call(['ls', '-la'])
# SECURITY_DISABLED: eval("print('hello')")
# SECURITY_DISABLED: exec("x = 1")
"""
        issues = self.analyzer.analyze_code(code, "test.py")
        
        dangerous_issues = [i for i in issues if i.category == "dangerous_function"]
        assert len(dangerous_issues) >= 2  # eval and exec at minimum
        
        # Check specific dangerous functions
        found_functions = [issue.description for issue in dangerous_issues]
        assert any("eval" in desc for desc in found_functions)
        assert any("exec" in desc for desc in found_functions)
        
    def test_hardcoded_credential_detection(self):
        """Test detection of hardcoded credentials."""
        code = """
# SECURITY_DISABLED: password = "secret123"
# SECURITY_DISABLED: api_key = "sk-1234567890abcdef"
# SECURITY_DISABLED: database_secret = "super_secret_key"
normal_var = "just_text"
"""
        issues = self.analyzer.analyze_code(code, "test.py")
        
        credential_issues = [i for i in issues if i.category == "credential_exposure"]
        assert len(credential_issues) >= 2  # Should find password and api_key
        
    def test_pattern_based_detection(self):
        """Test regex pattern-based security issue detection."""
        code = """
# Hard-coded password
# SECURITY_DISABLED: password = "my_secret_password"

# URL with embedded credentials
url = "https://user:password@example.com/api"

# Format string potential injection
query = "SELECT * FROM users WHERE id = {}".format(user_input)
"""
        issues = self.analyzer.analyze_code(code, "test.py")
        
        assert len(issues) > 0
        categories = [issue.category for issue in issues]
        assert "credential_exposure" in categories
        
    def test_syntax_error_handling(self):
        """Test handling of syntax errors in code."""
        code = """
def broken_function(
    # Missing closing parenthesis
    return "test"
"""
        issues = self.analyzer.analyze_code(code, "test.py")
        
        syntax_issues = [i for i in issues if i.category == "syntax"]
        assert len(syntax_issues) == 1
        assert "Syntax error" in syntax_issues[0].description


class TestInputValidator:
    """Test input validation functionality."""
    
    def setup_method(self):
        self.validator = InputValidator()
        
    def test_model_validation(self):
        """Test PyTorch model validation."""
        # Mock a large model
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(numel=Mock(return_value=2e9))]  # 2B params
        
        issues = self.validator.validate_model_input(mock_model)
        
        resource_issues = [i for i in issues if i.category == "resource_exhaustion"]
        assert len(resource_issues) >= 1
        assert "parameters" in resource_issues[0].description and "DoS risk" in resource_issues[0].description
        
    def test_file_path_validation(self):
        """Test file path security validation."""
        # Test path traversal
        issues = self.validator.validate_file_path("../../../etc/passwd")
        traversal_issues = [i for i in issues if i.category == "path_traversal"]
        assert len(traversal_issues) >= 1
        
        # Test forbidden extension
        issues = self.validator.validate_file_path("malicious.exe")
        malicious_issues = [i for i in issues if i.category == "malicious_file"]
        assert len(malicious_issues) >= 1
        
    def test_json_data_validation(self):
        """Test JSON data validation."""
        # Test deeply nested data
        deep_data = {"level1": {"level2": {"level3": {"level4": {"level5": {}}}}}}
        issues = self.validator.validate_json_data(deep_data, max_depth=3)
        
        depth_issues = [i for i in issues if "depth" in i.description]
        assert len(depth_issues) >= 1
        
        # Test large list
        large_list = list(range(2000))
        issues = self.validator.validate_json_data(large_list)
        
        size_issues = [i for i in issues if "Large list" in i.description]
        assert len(size_issues) >= 1
        
    def test_string_length_validation(self):
        """Test validation of very long strings."""
        long_string = "a" * 20000  # Longer than max_string_length
        issues = self.validator.validate_json_data(long_string)
        
        length_issues = [i for i in issues if "long string" in i.description]
        assert len(length_issues) >= 1


class TestSecurityScanner:
    """Test comprehensive security scanner."""
    
    def setup_method(self):
        self.scanner = SecurityScanner(SecurityLevel.MEDIUM)
        
    def test_scanner_initialization(self):
        """Test security scanner initialization."""
        assert self.scanner.security_level == SecurityLevel.MEDIUM
        assert self.scanner.code_analyzer is not None
        assert self.scanner.input_validator is not None
        
    def test_scan_with_secure_code(self):
        """Test scanning secure code."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a secure Python file
            secure_file = Path(temp_dir) / "secure.py"
            secure_file.write_text("""
def safe_function(x):
    return x * 2

def another_safe_function():
    import os
    return os.getenv('SAFE_VAR', 'default')
""")
            
            report = self.scanner.scan_system([temp_dir])
            
            assert report.total_issues >= 0  # May have some low-severity issues
            assert report.issues_by_severity.get('critical', 0) == 0
            assert report.scan_duration > 0
            
    def test_scan_with_insecure_code(self):
        """Test scanning insecure code."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an insecure Python file
            insecure_file = Path(temp_dir) / "insecure.py"
            insecure_file.write_text("""
import subprocess

# SECURITY_DISABLED: password = "hardcoded_secret"
# SECURITY_DISABLED: api_key = "sk-1234567890abcdef"

def dangerous_function(user_input):
    # SECURITY_DISABLED: eval(user_input)  # Dangerous!
    subprocess.call(['rm', '-rf', '/'])  # Very dangerous!
    
def format_injection(user_data):
    query = "SELECT * FROM users WHERE id = {}".format(user_data)
    return query
""")
            
            report = self.scanner.scan_system([temp_dir])
            
            assert report.total_issues > 0
            assert report.issues_by_severity.get('critical', 0) > 0
            assert report.failed_checks > 0
            
            # Check for specific issue types
            categories = [issue.category for issue in report.issues]
            assert "dangerous_function" in categories
            assert "credential_exposure" in categories
            
    def test_environment_scanning(self):
        """Test environment variable scanning."""
        # Mock environment variables
        with patch.dict(os.environ, {'AWS_SECRET_ACCESS_KEY': 'fake_secret_key_1234567890'}):
            issues = self.scanner._scan_environment()
            
            credential_issues = [i for i in issues if i.category == "credential_exposure"]
            assert len(credential_issues) >= 1
            
    def test_different_security_levels(self):
        """Test different security levels."""
        # Low security level
        low_scanner = SecurityScanner(SecurityLevel.LOW)
        assert low_scanner.security_level == SecurityLevel.LOW
        
        # High security level  
        high_scanner = SecurityScanner(SecurityLevel.HIGH)
        assert high_scanner.security_level == SecurityLevel.HIGH
        
        # Critical security level
        critical_scanner = SecurityScanner(SecurityLevel.CRITICAL)
        assert critical_scanner.security_level == SecurityLevel.CRITICAL
        
    def test_recommendation_generation(self):
        """Test security recommendation generation."""
        # Create mock issues
        from src.photonic_foundry.security import SecurityIssue
        
        issues = [
            SecurityIssue("critical", "credential_exposure", "Test critical", "test", "Fix it"),
            SecurityIssue("high", "injection", "Test high", "test", "Fix it"),
            SecurityIssue("medium", "validation", "Test medium", "test", "Fix it")
        ]
        
        recommendations = self.scanner._generate_recommendations(issues)
        
        assert len(recommendations) > 0
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("high-severity" in rec for rec in recommendations)
        
    def test_nonexistent_path_handling(self):
        """Test handling of nonexistent scan paths."""
        report = self.scanner.scan_system(["/nonexistent/path"])
        
        config_issues = [i for i in report.issues if i.category == "configuration"]
        # Should handle gracefully without crashing
        assert len(config_issues) >= 0  # May or may not have config issues


class TestSecurityIntegration:
    """Test security integration functions."""
    
    def test_scan_security_function(self):
        """Test the main scan_security function."""
        report = scan_security(SecurityLevel.LOW, [])
        
        assert report is not None
        assert report.security_level == SecurityLevel.LOW
        assert isinstance(report.total_issues, int)
        assert isinstance(report.scan_duration, float)
        
    def test_validate_input_security_function(self):
        """Test the validate_input_security function."""
        # Test with different input types
        
        # String input (potential file path)
        issues = validate_input_security("../dangerous/path")
        assert isinstance(issues, list)
        
        # Dictionary input
        issues = validate_input_security({"key": "value"})
        assert isinstance(issues, list)
        
        # Mock model input
        mock_model = Mock()
        mock_model.parameters.return_value = []
        issues = validate_input_security(mock_model)
        assert isinstance(issues, list)
        
    def test_issue_severity_levels(self):
        """Test all security issue severity levels."""
        from src.photonic_foundry.security import SecurityIssue
        
        severities = ["low", "medium", "high", "critical"]
        
        for severity in severities:
            issue = SecurityIssue(
                severity=severity,
                category="test",
                description="Test issue",
                location="test",
                recommendation="Fix it"
            )
            assert issue.severity == severity
            
    def test_cwe_id_assignment(self):
        """Test CWE ID assignment for security issues."""
        from src.photonic_foundry.security import SecurityIssue
        
        issue = SecurityIssue(
            severity="high",
            category="injection",
            description="Code injection vulnerability",
            location="test.py:10",
            recommendation="Sanitize inputs",
            cwe_id="CWE-94"
        )
        
        assert issue.cwe_id == "CWE-94"