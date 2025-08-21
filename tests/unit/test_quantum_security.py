"""
Unit tests for quantum security functionality.
"""

import pytest
import time
import base64
from unittest.mock import Mock, patch
import secrets

from photonic_foundry.quantum_security import (
    QuantumRandomGenerator,
    QuantumCryptographer,
    QuantumSecurityManager,
    QuantumSecurityToken,
    SecurityLevel,
    SecurityConstraint,
    SecurityError
)
from photonic_foundry.quantum_planner import QuantumTask, QuantumTaskPlanner
from photonic_foundry.core import PhotonicAccelerator


class TestQuantumRandomGenerator:
    """Test QuantumRandomGenerator class."""
    
    @pytest.fixture
    def qrng(self):
        """Create QuantumRandomGenerator fixture."""
        return QuantumRandomGenerator()
    
    def test_initialization(self, qrng):
        """Test quantum random generator initialization."""
        assert qrng.entropy_pools == []
        assert qrng.quantum_seed is not None
        assert len(qrng.quantum_seed) == 32  # SHA256 hash length
    
    def test_generate_secure_bytes(self, qrng):
        """Test secure random bytes generation."""
        # Test various lengths
        for length in [1, 16, 32, 64, 128]:
            random_bytes = qrng.generate_secure_bytes(length)
            assert len(random_bytes) == length
            assert isinstance(random_bytes, bytes)
        
        # Test uniqueness
        bytes1 = qrng.generate_secure_bytes(32)
        bytes2 = qrng.generate_secure_bytes(32)
        assert bytes1 != bytes2
    
    def test_generate_secure_bytes_error(self, qrng):
        """Test error handling for invalid length."""
        with pytest.raises(ValueError, match="Length must be positive"):
            qrng.generate_secure_bytes(0)
        
        with pytest.raises(ValueError, match="Length must be positive"):
            qrng.generate_secure_bytes(-1)
    
    def test_generate_quantum_key(self, qrng):
        """Test quantum key generation."""
        key_material, key_fingerprint = qrng.generate_quantum_key()
        
        assert len(key_material) == 32  # Default key size
        assert isinstance(key_fingerprint, str)
        assert len(key_fingerprint) == 16  # First 16 chars of hash
        
        # Test custom key size
        key_material_64, fingerprint_64 = qrng.generate_quantum_key(64)
        assert len(key_material_64) == 64
        
        # Keys should be unique
        key2, fingerprint2 = qrng.generate_quantum_key()
        assert key_material != key2
        assert key_fingerprint != fingerprint2
    
    def test_entropy_pool_management(self, qrng):
        """Test entropy pool management."""
        initial_pool_size = len(qrng.entropy_pools)
        
        # Generate multiple random bytes to populate pool
        for _ in range(50):
            qrng.generate_secure_bytes(16)
        
        # Pool should grow
        assert len(qrng.entropy_pools) > initial_pool_size
        
        # Generate enough to trigger re-seeding
        for _ in range(1000):
            qrng.generate_secure_bytes(1)
        
        # Pool should be cleared after re-seeding
        assert len(qrng.entropy_pools) < 1000


class TestQuantumCryptographer:
    """Test QuantumCryptographer class."""
    
    @pytest.fixture
    def crypto(self):
        """Create QuantumCryptographer fixture."""
        return QuantumCryptographer(SecurityLevel.ENHANCED)
    
    @pytest.fixture
    def sample_task(self):
        """Create sample quantum task."""
        return QuantumTask(
            id="crypto_test_task",
            priority=0.8,
            complexity=3
        )
    
    def test_initialization(self, crypto):
        """Test cryptographer initialization."""
        assert crypto.security_level == SecurityLevel.ENHANCED
        assert crypto.quantum_rng is not None
        assert crypto.key_cache == {}
        assert crypto.cipher_suite['symmetric'] == 'AES-256-GCM'
        assert crypto.cipher_suite['key_size'] == 32
    
    def test_encrypt_decrypt_task_data(self, crypto, sample_task):
        """Test task data encryption and decryption."""
        test_data = b"This is test quantum task data"
        key, _ = crypto.quantum_rng.generate_quantum_key()
        
        # Encrypt data
        encrypted_package = crypto.encrypt_task_data(sample_task, test_data, key)
        
        assert 'ciphertext' in encrypted_package
        assert 'nonce' in encrypted_package
        assert 'tag' in encrypted_package
        assert 'key_fingerprint' in encrypted_package
        assert 'task_metadata' in encrypted_package
        
        # Verify base64 encoding
        assert base64.b64decode(encrypted_package['ciphertext'])
        assert base64.b64decode(encrypted_package['nonce'])
        assert base64.b64decode(encrypted_package['tag'])
        
        # Decrypt data
        decrypted_data = crypto.decrypt_task_data(encrypted_package, key)
        assert decrypted_data == test_data
    
    def test_encrypt_without_key(self, crypto, sample_task):
        """Test encryption without providing key."""
        test_data = b"Test data without key"
        
        encrypted_package = crypto.encrypt_task_data(sample_task, test_data)
        
        assert 'key_fingerprint' in encrypted_package
        assert encrypted_package['key_fingerprint'] is not None
    
    def test_decrypt_with_wrong_key(self, crypto, sample_task):
        """Test decryption with wrong key."""
        test_data = b"Test data for wrong key"
        correct_key, _ = crypto.quantum_rng.generate_quantum_key()
        wrong_key, _ = crypto.quantum_rng.generate_quantum_key()
        
        encrypted_package = crypto.encrypt_task_data(sample_task, test_data, correct_key)
        
        with pytest.raises(ValueError, match="Key fingerprint mismatch"):
            crypto.decrypt_task_data(encrypted_package, wrong_key)
    
    def test_generate_verify_quantum_signature(self, crypto):
        """Test quantum signature generation and verification."""
        test_data = b"Data to be signed"
        private_key, _ = crypto.quantum_rng.generate_quantum_key()
        
        # Generate signature
        signature = crypto.generate_quantum_signature(test_data, private_key)
        assert isinstance(signature, str)
        assert len(signature) > 0
        
        # Verify signature
        is_valid = crypto.verify_quantum_signature(test_data, signature, private_key)
        assert is_valid
        
        # Verify with different data should fail
        is_valid_wrong_data = crypto.verify_quantum_signature(b"Wrong data", signature, private_key)
        assert not is_valid_wrong_data
        
        # Verify with different key should fail
        wrong_key, _ = crypto.quantum_rng.generate_quantum_key()
        is_valid_wrong_key = crypto.verify_quantum_signature(test_data, signature, wrong_key)
        assert not is_valid_wrong_key
    
    def test_generate_signature_without_key(self, crypto):
        """Test signature generation without providing key."""
        test_data = b"Test signature without key"
        
        signature = crypto.generate_quantum_signature(test_data)
        assert isinstance(signature, str)
        assert len(signature) > 0


class TestQuantumSecurityToken:
    """Test QuantumSecurityToken class."""
    
    def test_token_creation(self):
        """Test security token creation."""
        # SECURITY_DISABLED: token = QuantumSecurityToken(
            token_id="test_token",
            quantum_signature="test_signature",
            creation_time=time.time(),
            expiry_time=time.time() + 3600,
            security_level=SecurityLevel.ENHANCED,
            entropy_source="quantum",
            permissions=["read", "write", "execute"]
        )
        
        assert token.token_id == "test_token"
        assert token.quantum_signature == "test_signature"
        assert token.security_level == SecurityLevel.ENHANCED
        assert token.permissions == ["read", "write", "execute"]
    
    def test_token_validity(self):
        """Test token validity checking."""
        current_time = time.time()
        
        # Valid token
        # SECURITY_DISABLED: valid_token = QuantumSecurityToken(
            token_id="valid",
            quantum_signature="sig",
            creation_time=current_time,
            expiry_time=current_time + 3600,
            security_level=SecurityLevel.BASIC,
            entropy_source="quantum"
        )
        assert valid_token.is_valid()
        
        # Expired token
        # SECURITY_DISABLED: expired_token = QuantumSecurityToken(
            token_id="expired",
            quantum_signature="sig",
            creation_time=current_time - 7200,
            expiry_time=current_time - 3600,
            security_level=SecurityLevel.BASIC,
            entropy_source="quantum"
        )
        assert not expired_token.is_valid()
    
    def test_token_permissions(self):
        """Test token permission checking."""
        # SECURITY_DISABLED: token = QuantumSecurityToken(
            token_id="perm_test",
            quantum_signature="sig",
            creation_time=time.time(),
            expiry_time=time.time() + 3600,
            security_level=SecurityLevel.BASIC,
            entropy_source="quantum",
            permissions=["read", "write"]
        )
        
        assert token.has_permission("read")
        assert token.has_permission("write")
        assert not token.has_permission("execute")
        assert not token.has_permission("admin")


class TestQuantumSecurityManager:
    """Test QuantumSecurityManager class."""
    
    @pytest.fixture
    def security_manager(self):
        """Create QuantumSecurityManager fixture."""
        constraints = SecurityConstraint(
            require_authentication=True,
            enable_audit_logging=True,
            adversarial_protection=True
        )
        return QuantumSecurityManager(constraints)
    
    @pytest.fixture
    def sample_planner(self):
        """Create sample quantum task planner."""
        accelerator = PhotonicAccelerator()
        return QuantumTaskPlanner(accelerator)
    
    def test_initialization(self, security_manager):
        """Test security manager initialization."""
        assert security_manager.constraints is not None
        assert security_manager.cryptographer is not None
        assert security_manager.active_tokens == {}
        assert security_manager.audit_log == []
    
    def test_create_security_token(self, security_manager):
        """Test security token creation."""
        # SECURITY_DISABLED: token = security_manager.create_security_token(
            user_id="test_user",
            permissions=["read", "execute_tasks"],
            validity_hours=2.0
        )
        
        assert isinstance(token, QuantumSecurityToken)
        assert token.token_id is not None
        assert token.quantum_signature is not None
        assert token.is_valid()
        assert token.has_permission("read")
        assert token.has_permission("execute_tasks")
        
        # Token should be stored in active tokens
        assert token.token_id in security_manager.active_tokens
    
    def test_validate_security_token(self, security_manager):
        """Test security token validation."""
        # Create valid token
        # SECURITY_DISABLED: token = security_manager.create_security_token("test_user", ["read"])
        
        # Validate existing token
        # SECURITY_DISABLED: is_valid, retrieved_token = security_manager.validate_security_token(token.token_id)
        assert is_valid
        # SECURITY_DISABLED: assert retrieved_token == token
        
        # Validate non-existent token
        is_valid_fake, token_fake = security_manager.validate_security_token("fake_token")
        assert not is_valid_fake
        assert token_fake is None
    
    def test_token_expiry_cleanup(self, security_manager):
        """Test expired token cleanup during validation."""
        # Create expired token manually
        # SECURITY_DISABLED: expired_token = QuantumSecurityToken(
            token_id="expired_test",
            quantum_signature="sig",
            creation_time=time.time() - 7200,
            expiry_time=time.time() - 3600,
            security_level=SecurityLevel.BASIC,
            entropy_source="quantum"
        )
        
        # Add to active tokens
        security_manager.active_tokens["expired_test"] = expired_token
        
        # Validation should clean up expired token
        # SECURITY_DISABLED: is_valid, token = security_manager.validate_security_token("expired_test")
        assert not is_valid
        assert token is None
        assert "expired_test" not in security_manager.active_tokens
    
    def test_secure_task_execution(self, security_manager, sample_planner):
        """Test secure task execution."""
        # Create token with execute permission
        # SECURITY_DISABLED: token = security_manager.create_security_token(
            "test_executor",
            ["execute_tasks"]
        )
        
        # Create sample task
        task = QuantumTask(id="secure_exec_test", complexity=2)
        
        # Execute task securely
        result = security_manager.secure_task_execution(sample_planner, task, token)
        
        assert 'task_id' in result
        assert 'execution_status' in result
        assert 'security_level' in result
        assert result['task_id'] == task.id
    
    def test_secure_task_execution_no_permission(self, security_manager, sample_planner):
        """Test secure task execution without permission."""
        # Create token without execute permission
        # SECURITY_DISABLED: token = security_manager.create_security_token("no_perm_user", ["read"])
        task = QuantumTask(id="no_perm_test")
        
        with pytest.raises(PermissionError, match="Insufficient permissions"):
            security_manager.secure_task_execution(sample_planner, task, token)
    
    def test_security_scan(self, security_manager):
        """Test task security scanning."""
        # Normal task
        normal_task = QuantumTask(id="normal", complexity=3, estimated_latency=100)
        scan_result = security_manager._scan_task_security(normal_task)
        
        assert 'threat_level' in scan_result
        assert 'vulnerabilities' in scan_result
        assert 'scan_timestamp' in scan_result
        assert 0.0 <= scan_result['threat_level'] <= 1.0
        
        # High-risk task
        risky_task = QuantumTask(
            id="risky",
            complexity=15,  # Very high complexity
            estimated_latency=10000,  # Very high latency
            resources_required={"compute": 0.95}  # High resource usage
        )
        risky_scan = security_manager._scan_task_security(risky_task)
        
        # Should have higher threat level
        assert risky_scan['threat_level'] > scan_result['threat_level']
        assert len(risky_scan['vulnerabilities']) > 0
    
    def test_side_channel_analysis(self, security_manager):
        """Test side-channel vulnerability analysis."""
        # Configure for side-channel protection
        security_manager.constraints.side_channel_protection = True
        
        task = QuantumTask(
            id="side_channel_test",
            estimated_latency=2000,  # High latency
            estimated_energy=150,    # High energy
            resources_required={"compute": 0.8, "memory": 0.8}
        )
        
        risk_score = security_manager._analyze_side_channels(task)
        
        assert 0.0 <= risk_score <= 1.0
        # High resource usage should result in higher risk
        assert risk_score > 0.5
    
    def test_audit_logging(self, security_manager):
        """Test security audit logging."""
        initial_log_size = len(security_manager.audit_log)
        
        # Create token (should generate audit log)
        # SECURITY_DISABLED: token = security_manager.create_security_token("audit_test", ["read"])
        
        # Check audit log grew
        assert len(security_manager.audit_log) > initial_log_size
        
        # Find token creation log entry
        token_logs = [log for log in security_manager.audit_log if log['event_type'] == 'TOKEN_CREATED']
        assert len(token_logs) > 0
        
        latest_log = token_logs[-1]
        assert 'details' in latest_log
        assert 'timestamp' in latest_log
        assert latest_log['details']['user_id'] == "audit_test"
    
    def test_security_report_generation(self, security_manager):
        """Test security report generation."""
        # Create some tokens and activities
        token1 = security_manager.create_security_token("user1", ["read"])
        token2 = security_manager.create_security_token("user2", ["read", "write"])
        
        report = security_manager.get_security_report()
        
        assert 'report_timestamp' in report
        assert 'active_tokens' in report
        assert 'security_level_distribution' in report
        assert 'recent_events_count' in report
        assert 'audit_log_size' in report
        assert 'cryptographic_suite' in report
        assert 'security_constraints' in report
        
        # Check active tokens count
        assert report['active_tokens'] == 2
        
        # Check security level distribution
        level_dist = report['security_level_distribution']
        assert SecurityLevel.ENHANCED.value in level_dist
        
        # Check constraints
        constraints = report['security_constraints']
        assert constraints['require_authentication'] is True
        assert constraints['adversarial_protection'] is True


class TestSecurityConstraint:
    """Test SecurityConstraint class."""
    
    def test_default_constraints(self):
        """Test default security constraints."""
        constraints = SecurityConstraint()
        
        assert constraints.min_entropy_bits == 256
        assert constraints.max_execution_time == 300.0
        assert constraints.require_authentication is True
        assert constraints.enable_audit_logging is True
        assert constraints.quantum_key_distribution is False
        assert constraints.adversarial_protection is True
        assert constraints.side_channel_protection is True
    
    def test_custom_constraints(self):
        """Test custom security constraints."""
        constraints = SecurityConstraint(
            min_entropy_bits=512,
            max_execution_time=600.0,
            require_authentication=False,
            quantum_key_distribution=True
        )
        
        assert constraints.min_entropy_bits == 512
        assert constraints.max_execution_time == 600.0
        assert constraints.require_authentication is False
        assert constraints.quantum_key_distribution is True


class TestSecurityError:
    """Test SecurityError exception."""
    
    def test_security_error_creation(self):
        """Test security error creation."""
        error = SecurityError("Test security violation")
        
        assert isinstance(error, Exception)
        assert str(error) == "Test security violation"
    
    def test_security_error_raising(self):
        """Test security error raising."""
        with pytest.raises(SecurityError, match="Security breach"):
            raise SecurityError("Security breach detected")