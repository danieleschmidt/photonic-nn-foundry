# Security Guidelines

This document outlines security practices and guidelines for developing, deploying, and maintaining photonic-nn-foundry.

## Security Principles

### 1. Defense in Depth
Multiple layers of security controls to protect against various threat vectors.

### 2. Least Privilege
Grant minimum necessary permissions for users, processes, and services.

### 3. Fail Secure
System should fail to a secure state when errors occur.

### 4. Security by Design
Security considerations integrated throughout the development lifecycle.

## Code Security

### Input Validation

```python
from pydantic import BaseModel, validator, Field
from pathlib import Path
import re

class ModelInputValidator(BaseModel):
    model_path: str = Field(..., description="Path to PyTorch model file")
    output_dir: str = Field(..., description="Output directory path")
    config_name: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
    
    @validator('model_path')
    def validate_model_path(cls, v):
        # Prevent directory traversal
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid model path")
        
        # Check file extension
        if not v.endswith(('.pth', '.pt', '.onnx')):
            raise ValueError("Unsupported model format")
        
        # Validate path exists and is readable
        path = Path(v)
        if not path.exists() or not path.is_file():
            raise ValueError("Model file not found")
            
        return v
    
    @validator('output_dir')
    def validate_output_dir(cls, v):
        # Sanitize output directory path
        sanitized = re.sub(r'[^\w\-_./]', '', v)
        if sanitized != v:
            raise ValueError("Invalid characters in output directory")
        
        # Prevent directory traversal
        if '..' in v:
            raise ValueError("Directory traversal not allowed")
            
        return sanitized
```

### Secure File Handling

```python
import os
import tempfile
from pathlib import Path
from typing import Optional

class SecureFileHandler:
    """Secure file operations with validation and sandboxing"""
    
    def __init__(self, base_dir: str, max_file_size: int = 100 * 1024 * 1024):
        self.base_dir = Path(base_dir).resolve()
        self.max_file_size = max_file_size
        
        # Ensure base directory exists and is secure
        self.base_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.base_dir, 0o750)
    
    def validate_path(self, file_path: str) -> Path:
        """Validate and resolve file path within base directory"""
        path = Path(file_path).resolve()
        
        # Check if path is within base directory
        try:
            path.relative_to(self.base_dir)
        except ValueError:
            raise SecurityError("Path outside allowed directory")
        
        return path
    
    def safe_read(self, file_path: str) -> bytes:
        """Safely read file with size and path validation"""
        path = self.validate_path(file_path)
        
        # Check file size
        if path.stat().st_size > self.max_file_size:
            raise SecurityError("File size exceeds maximum allowed")
        
        # Read file securely
        with open(path, 'rb') as f:
            return f.read()
    
    def safe_write(self, file_path: str, content: bytes) -> None:
        """Safely write file with validation"""
        path = self.validate_path(file_path)
        
        # Check content size
        if len(content) > self.max_file_size:
            raise SecurityError("Content size exceeds maximum allowed")
        
        # Write to temporary file first, then move
        with tempfile.NamedTemporaryFile(
            dir=path.parent, 
            delete=False,
            mode='wb'
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Atomic move
        os.rename(tmp_path, path)
        os.chmod(path, 0o640)

class SecurityError(Exception):
    """Custom security exception"""
    pass
```

### SQL Injection Prevention

```python
from sqlalchemy import text
from sqlalchemy.orm import Session

class SecureQueryBuilder:
    """Secure database query builder"""
    
    @staticmethod
    def find_models_by_user(session: Session, user_id: int, model_type: str):
        # Use parameterized queries
        query = text("""
            SELECT id, name, created_at 
            FROM models 
            WHERE user_id = :user_id 
            AND model_type = :model_type
            ORDER BY created_at DESC
        """)
        
        return session.execute(
            query, 
            {"user_id": user_id, "model_type": model_type}
        ).fetchall()
```

## Authentication and Authorization

### JWT Token Security

```python
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict

class SecureTokenManager:
    """Secure JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        if len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=1)
    
    def generate_token(self, user_id: int, permissions: list) -> str:
        """Generate secure JWT token"""
        now = datetime.utcnow()
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'iat': now,
            'exp': now + self.token_expiry,
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={
                    'require_exp': True,
                    'require_iat': True,
                    'verify_exp': True
                }
            )
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token_id: str) -> None:
        """Add token to revocation list"""
        # Implementation depends on storage backend
        pass
```

### Role-Based Access Control

```python
from enum import Enum
from functools import wraps
from typing import List

class Permission(Enum):
    READ_MODELS = "read_models"
    WRITE_MODELS = "write_models"
    DELETE_MODELS = "delete_models"
    ADMIN_ACCESS = "admin_access"

class Role(Enum):
    USER = [Permission.READ_MODELS]
    DEVELOPER = [Permission.READ_MODELS, Permission.WRITE_MODELS]
    ADMIN = [Permission.READ_MODELS, Permission.WRITE_MODELS, 
             Permission.DELETE_MODELS, Permission.ADMIN_ACCESS]

def require_permission(required_permission: Permission):
    """Decorator for permission-based access control"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get user permissions from context
            user_permissions = get_current_user_permissions()
            
            if required_permission not in user_permissions:
                raise PermissionError("Insufficient permissions")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_permission(Permission.WRITE_MODELS)
def create_model(model_data: dict):
    """Create new model - requires write permission"""
    pass
```

## Container Security

### Dockerfile Security Best Practices

```dockerfile
# Use specific version tags, not 'latest'
FROM python:3.10.12-slim

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/app/.local/bin:$PATH

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        dumb-init && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies as root, then switch to non-root
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements-prod.txt && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser config/ config/

# Switch to non-root user
USER appuser

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Set secure defaults
CMD ["python", "-m", "photonic_foundry.cli"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"
```

### Container Runtime Security

```yaml
# Kubernetes SecurityContext
apiVersion: v1
kind: Pod
metadata:
  name: photonic-foundry
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
    
  containers:
  - name: photonic-foundry
    image: photonic-foundry:latest
    
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
      runAsNonRoot: true
      runAsUser: 1000
      runAsGroup: 1000
    
    resources:
      limits:
        memory: "1Gi"
        cpu: "500m"
      requests:
        memory: "256Mi" 
        cpu: "100m"
    
    volumeMounts:
    - name: tmp-volume
      mountPath: /tmp
    - name: data-volume
      mountPath: /data
      readOnly: false
  
  volumes:
  - name: tmp-volume
    emptyDir: {}
  - name: data-volume
    persistentVolumeClaim:
      claimName: data-pvc
```

## Network Security

### TLS Configuration

```python
import ssl
from flask import Flask
from werkzeug.serving import WSGIRequestHandler

class SecureWSGIRequestHandler(WSGIRequestHandler):
    """Custom request handler with security headers"""
    
    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Strict-Transport-Security', 
                        'max-age=31536000; includeSubDomains')
        self.send_header('Content-Security-Policy', 
                        "default-src 'self'")
        super().end_headers()

def create_secure_app():
    app = Flask(__name__)
    
    # Configure SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_ciphers('ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:'
                       'ECDH+AES128:DH+AES:RSA+AESGCM:RSA+AES:!aNULL:'
                       '!MD5:!DSS')
    
    return app, context
```

### Network Policies

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: photonic-foundry-network-policy
  namespace: photonic-foundry
spec:
  podSelector:
    matchLabels:
      app: photonic-foundry
  
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-system
    ports:
    - protocol: TCP
      port: 8080
  
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database-system
    ports:
    - protocol: TCP
      port: 5432
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
```

## Secrets Management

### Environment-based Secrets

```python
import os
from typing import Optional
from cryptography.fernet import Fernet

class SecretManager:
    """Secure secret management"""
    
    def __init__(self):
        self.encryption_key = os.environ.get('ENCRYPTION_KEY')
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode())
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve secret with optional decryption"""
        encrypted_value = os.environ.get(f"{secret_name}_ENCRYPTED")
        if encrypted_value and self.cipher:
            return self.cipher.decrypt(encrypted_value.encode()).decode()
        
        return os.environ.get(secret_name)
    
    def set_secret(self, secret_name: str, value: str) -> None:
        """Store encrypted secret"""
        if self.cipher:
            encrypted_value = self.cipher.encrypt(value.encode()).decode()
            os.environ[f"{secret_name}_ENCRYPTED"] = encrypted_value
        else:
            os.environ[secret_name] = value
```

## Monitoring and Incident Response

### Security Logging

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class SecurityLogger:
    """Structured security event logging"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event with structured data"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': 'HIGH' if event_type in ['unauthorized_access', 'injection_attempt'] else 'MEDIUM',
            'details': details
        }
        
        self.logger.warning(json.dumps(event))
    
    def log_failed_auth(self, user_id: str, ip_address: str):
        """Log authentication failure"""
        self.log_security_event('failed_authentication', {
            'user_id': user_id,
            'ip_address': ip_address,
            'action': 'login_attempt'
        })
    
    def log_suspicious_activity(self, user_id: str, activity: str, details: Dict):
        """Log suspicious user activity"""
        self.log_security_event('suspicious_activity', {
            'user_id': user_id,
            'activity': activity,
            'details': details
        })
```

### Intrusion Detection

```python
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List

class IntrusionDetector:
    """Simple intrusion detection system"""
    
    def __init__(self, max_attempts: int = 5, time_window: int = 300):
        self.max_attempts = max_attempts
        self.time_window = timedelta(seconds=time_window)
        self.failed_attempts: Dict[str, deque] = defaultdict(deque)
        self.blocked_ips: Dict[str, datetime] = {}
    
    def record_failed_attempt(self, ip_address: str) -> bool:
        """Record failed attempt and check if IP should be blocked"""
        now = datetime.utcnow()
        
        # Clean old attempts
        while (self.failed_attempts[ip_address] and 
               self.failed_attempts[ip_address][0] < now - self.time_window):
            self.failed_attempts[ip_address].popleft()
        
        # Add current attempt
        self.failed_attempts[ip_address].append(now)
        
        # Block if too many attempts
        if len(self.failed_attempts[ip_address]) >= self.max_attempts:
            self.blocked_ips[ip_address] = now + timedelta(hours=1)
            return True
        
        return False
    
    def is_blocked(self, ip_address: str) -> bool:
        """Check if IP address is currently blocked"""
        if ip_address in self.blocked_ips:
            if datetime.utcnow() < self.blocked_ips[ip_address]:
                return True
            else:
                del self.blocked_ips[ip_address]
        
        return False
```

## Vulnerability Management

### Dependency Scanning

```python
# requirements-security.txt
safety>=2.0.0
bandit>=1.7.0
pip-audit>=2.0.0

# Security scanning script
#!/bin/bash
set -e

echo "Running security scans..."

# Check for known vulnerabilities
echo "Checking dependencies with Safety..."
safety check --json --output safety-report.json

echo "Checking dependencies with pip-audit..."
pip-audit --format=json --output pip-audit-report.json

echo "Running Bandit code analysis..."
bandit -r src/ -f json -o bandit-report.json

echo "Security scan complete. Check reports for issues."
```

### Regular Security Updates

```yaml
# .github/workflows/security-updates.yml
name: Security Updates

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  security-updates:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Update dependencies
      run: |
        pip install --upgrade pip
        pip install pip-upgrader
        pip-upgrade requirements.txt --skip-package-installation
    
    - name: Run security tests
      run: |
        pip install -r requirements-dev.txt
        python -m pytest tests/security/
    
    - name: Create pull request
      uses: peter-evans/create-pull-request@v5
      with:
        title: "Security: Update dependencies"
        body: |
          Automated security updates for dependencies.
          
          Please review changes before merging.
        branch: security/dependency-updates
        delete-branch: true
```

## Security Checklist

### Development Phase
- [ ] Input validation implemented
- [ ] Output encoding/escaping in place
- [ ] Authentication and authorization configured
- [ ] Secure coding practices followed
- [ ] Security tests written
- [ ] Code review for security issues
- [ ] Dependency vulnerability scan clean

### Deployment Phase
- [ ] Secrets properly managed
- [ ] TLS/SSL configured correctly
- [ ] Network security policies applied
- [ ] Container security hardening done
- [ ] Monitoring and logging configured
- [ ] Backup and recovery procedures tested
- [ ] Incident response plan documented

### Operations Phase
- [ ] Regular security updates applied
- [ ] Vulnerability scans scheduled
- [ ] Access logs monitored
- [ ] Security metrics tracked
- [ ] Incident response procedures tested
- [ ] Security training completed
- [ ] Compliance requirements met

This security guide provides comprehensive coverage for securing photonic-nn-foundry throughout its lifecycle, from development to production deployment.