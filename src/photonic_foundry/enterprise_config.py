"""
Enterprise-grade configuration management system.

This module provides comprehensive configuration management capabilities including:
- Hierarchical configuration with environment overrides
- Dynamic configuration updates and hot-reloading
- Configuration validation and schema enforcement
- Secure configuration storage and encryption
- Configuration versioning and rollback
- Environment-specific configuration management
- Configuration templates and inheritance
- Audit logging and change tracking
"""

import time
import threading
import os
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import (Dict, List, Any, Optional, Callable, Union, Tuple, Set, 
                   Protocol, TypeVar, Generic, Type, get_type_hints)
import logging
import numpy as np
from pathlib import Path
import pickle
import hashlib
import weakref
import gc
import traceback
from collections import defaultdict, deque, OrderedDict
from enum import Enum
import copy
from datetime import datetime, timedelta
import math
import statistics
from contextlib import contextmanager
from threading import RLock, Condition, Event
from abc import ABC, abstractmethod
import uuid
import sqlite3
import base64
import zlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jsonschema
from jsonschema import validate, ValidationError
import tempfile
import shutil
import fcntl
import signal
import socket
import platform

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass


class ValidationError(ConfigurationError):
    """Configuration validation error."""
    pass


class SecurityError(ConfigurationError):
    """Configuration security error."""
    pass


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class ConfigSource(Enum):
    """Configuration source types."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"


class ConfigSecurity(Enum):
    """Configuration security levels."""
    NONE = "none"
    BASIC = "basic"
    ENCRYPTED = "encrypted"
    VAULT = "vault"


@dataclass
class ConfigMetadata:
    """Metadata for configuration entries."""
    source: ConfigSource
    format: ConfigFormat
    timestamp: datetime
    version: str
    checksum: str
    encrypted: bool = False
    schema_validated: bool = False
    environment: Optional[Environment] = None
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source': self.source.value,
            'format': self.format.value,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version,
            'checksum': self.checksum,
            'encrypted': self.encrypted,
            'schema_validated': self.schema_validated,
            'environment': self.environment.value if self.environment else None,
            'tags': list(self.tags)
        }


@dataclass
class ConfigChangeEvent:
    """Configuration change event."""
    event_id: str
    config_key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    user: str
    environment: Environment
    change_type: str  # create, update, delete
    validation_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigSchema:
    """Configuration schema definition and validation."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.validator = jsonschema.Draft7Validator(schema)
        
    def validate(self, config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration against schema."""
        errors = []
        
        try:
            validate(instance=config_data, schema=self.schema)
            return True, []
        except ValidationError as e:
            errors.append(str(e))
            
        # Collect all validation errors
        for error in self.validator.iter_errors(config_data):
            errors.append(f"Path: {'.'.join(str(x) for x in error.absolute_path)}, Error: {error.message}")
            
        return len(errors) == 0, errors
        
    def get_default_values(self) -> Dict[str, Any]:
        """Extract default values from schema."""
        defaults = {}
        
        def extract_defaults(schema_part: Dict[str, Any], path: str = ""):
            if isinstance(schema_part, dict):
                if 'default' in schema_part:
                    if path:
                        defaults[path] = schema_part['default']
                        
                if 'properties' in schema_part:
                    for prop_name, prop_schema in schema_part['properties'].items():
                        new_path = f"{path}.{prop_name}" if path else prop_name
                        extract_defaults(prop_schema, new_path)
                        
        extract_defaults(self.schema)
        return defaults


class SecurityManager:
    """Manages configuration security and encryption."""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key
        self.encryption_key = None
        self._initialize_encryption()
        
    def _initialize_encryption(self):
        """Initialize encryption capabilities."""
        if self.master_key:
            # Derive encryption key from master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'photonic_foundry_salt',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            self.encryption_key = Fernet(key)
            
    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value."""
        if not self.encryption_key:
            raise SecurityError("Encryption not initialized")
            
        return self.encryption_key.encrypt(value.encode()).decode()
        
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        if not self.encryption_key:
            raise SecurityError("Encryption not initialized")
            
        return self.encryption_key.decrypt(encrypted_value.encode()).decode()
        
    def is_sensitive_key(self, key: str) -> bool:
        """Check if configuration key contains sensitive data."""
        sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'credential',
            'api_key', 'auth', 'cert', 'private', 'passphrase'
        ]
        
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in sensitive_patterns)
        
    def sanitize_for_logging(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration data for logging."""
        sanitized = {}
        
        for key, value in config_data.items():
            if self.is_sensitive_key(key):
                if isinstance(value, str) and len(value) > 4:
                    sanitized[key] = f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
                else:
                    sanitized[key] = "***HIDDEN***"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_for_logging(value)
            else:
                sanitized[key] = value
                
        return sanitized


class ConfigurationStore:
    """Persistent storage for configuration data."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.connection = None
        self._lock = RLock()
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database schema."""
        with self._lock:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Configuration table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configurations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    version TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    active BOOLEAN DEFAULT TRUE,
                    UNIQUE(key, environment, version)
                )
            ''')
            
            # Change history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    config_key TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    change_type TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    validation_passed BOOLEAN NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Schema definitions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schemas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    schema_definition TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_config_key_env ON configurations(key, environment)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_timestamp ON config_history(timestamp)')
            
            self.connection.commit()
            
    def store_configuration(self, key: str, value: Any, environment: Environment,
                          version: str, metadata: ConfigMetadata) -> bool:
        """Store configuration in database."""
        try:
            with self._lock:
                cursor = self.connection.cursor()
                
                # Serialize value and metadata
                serialized_value = json.dumps(value) if not isinstance(value, str) else value
                serialized_metadata = json.dumps(metadata.to_dict())
                
                current_time = time.time()
                
                # Deactivate previous versions
                cursor.execute('''
                    UPDATE configurations 
                    SET active = FALSE 
                    WHERE key = ? AND environment = ? AND active = TRUE
                ''', (key, environment.value))
                
                # Insert new configuration
                cursor.execute('''
                    INSERT INTO configurations 
                    (key, value, environment, version, metadata, created_at, updated_at, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, TRUE)
                ''', (key, serialized_value, environment.value, version, 
                      serialized_metadata, current_time, current_time))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store configuration: {e}")
            return False
            
    def get_configuration(self, key: str, environment: Environment, 
                         version: Optional[str] = None) -> Optional[Tuple[Any, ConfigMetadata]]:
        """Get configuration from database."""
        with self._lock:
            cursor = self.connection.cursor()
            
            if version:
                cursor.execute('''
                    SELECT value, metadata FROM configurations
                    WHERE key = ? AND environment = ? AND version = ?
                ''', (key, environment.value, version))
            else:
                cursor.execute('''
                    SELECT value, metadata FROM configurations
                    WHERE key = ? AND environment = ? AND active = TRUE
                    ORDER BY updated_at DESC LIMIT 1
                ''', (key, environment.value))
                
            row = cursor.fetchone()
            if not row:
                return None
                
            value, metadata_json = row
            
            # Deserialize
            try:
                deserialized_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                deserialized_value = value
                
            metadata_dict = json.loads(metadata_json)
            metadata = ConfigMetadata(
                source=ConfigSource(metadata_dict['source']),
                format=ConfigFormat(metadata_dict['format']),
                timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
                version=metadata_dict['version'],
                checksum=metadata_dict['checksum'],
                encrypted=metadata_dict.get('encrypted', False),
                schema_validated=metadata_dict.get('schema_validated', False),
                environment=Environment(metadata_dict['environment']) if metadata_dict.get('environment') else None,
                tags=set(metadata_dict.get('tags', []))
            )
            
            return deserialized_value, metadata
            
    def record_change(self, change_event: ConfigChangeEvent) -> bool:
        """Record configuration change in history."""
        try:
            with self._lock:
                cursor = self.connection.cursor()
                
                cursor.execute('''
                    INSERT INTO config_history
                    (event_id, config_key, old_value, new_value, change_type, 
                     user_name, environment, timestamp, validation_passed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    change_event.event_id,
                    change_event.config_key,
                    json.dumps(change_event.old_value) if change_event.old_value is not None else None,
                    json.dumps(change_event.new_value) if change_event.new_value is not None else None,
                    change_event.change_type,
                    change_event.user,
                    change_event.environment.value,
                    change_event.timestamp.timestamp(),
                    change_event.validation_passed,
                    json.dumps(change_event.metadata)
                ))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to record configuration change: {e}")
            return False
            
    def get_configuration_history(self, key: str, environment: Environment,
                                 limit: int = 50) -> List[ConfigChangeEvent]:
        """Get configuration change history."""
        with self._lock:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT event_id, config_key, old_value, new_value, change_type,
                       user_name, environment, timestamp, validation_passed, metadata
                FROM config_history
                WHERE config_key = ? AND environment = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (key, environment.value, limit))
            
            history = []
            for row in cursor.fetchall():
                event_id, config_key, old_value, new_value, change_type, user_name, env, timestamp, validation_passed, metadata = row
                
                # Deserialize values
                old_val = json.loads(old_value) if old_value else None
                new_val = json.loads(new_value) if new_value else None
                meta = json.loads(metadata) if metadata else {}
                
                event = ConfigChangeEvent(
                    event_id=event_id,
                    config_key=config_key,
                    old_value=old_val,
                    new_value=new_val,
                    timestamp=datetime.fromtimestamp(timestamp),
                    user=user_name,
                    environment=Environment(env),
                    change_type=change_type,
                    validation_passed=bool(validation_passed),
                    metadata=meta
                )
                history.append(event)
                
            return history
            
    def list_configurations(self, environment: Environment) -> List[str]:
        """List all configuration keys for an environment."""
        with self._lock:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT DISTINCT key FROM configurations
                WHERE environment = ? AND active = TRUE
                ORDER BY key
            ''', (environment.value,))
            
            return [row[0] for row in cursor.fetchall()]
            
    def store_schema(self, name: str, schema_definition: Dict[str, Any], version: str) -> bool:
        """Store configuration schema."""
        try:
            with self._lock:
                cursor = self.connection.cursor()
                
                # Deactivate previous versions
                cursor.execute('UPDATE schemas SET active = FALSE WHERE name = ?', (name,))
                
                # Insert new schema
                cursor.execute('''
                    INSERT INTO schemas (name, schema_definition, version, created_at, active)
                    VALUES (?, ?, ?, ?, TRUE)
                ''', (name, json.dumps(schema_definition), version, time.time()))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store schema: {e}")
            return False
            
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration schema."""
        with self._lock:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT schema_definition FROM schemas
                WHERE name = ? AND active = TRUE
                ORDER BY created_at DESC LIMIT 1
            ''', (name,))
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None


class ConfigurationTemplate:
    """Configuration template with inheritance and variable substitution."""
    
    def __init__(self, template_data: Dict[str, Any]):
        self.template_data = template_data
        self.variables = {}
        
    def set_variables(self, variables: Dict[str, Any]):
        """Set template variables."""
        self.variables.update(variables)
        
    def render(self, additional_vars: Dict[str, Any] = None) -> Dict[str, Any]:
        """Render template with variable substitution."""
        all_vars = {**self.variables, **(additional_vars or {})}
        return self._substitute_variables(copy.deepcopy(self.template_data), all_vars)
        
    def _substitute_variables(self, data: Any, variables: Dict[str, Any]) -> Any:
        """Recursively substitute variables in data structure."""
        if isinstance(data, str):
            # Simple variable substitution: ${variable_name}
            import re
            pattern = r'\\$\\{([^}]+)\\}'
            
            def replace_var(match):
                var_name = match.group(1)
                return str(variables.get(var_name, match.group(0)))
                
            return re.sub(pattern, replace_var, data)
            
        elif isinstance(data, dict):
            return {key: self._substitute_variables(value, variables) 
                   for key, value in data.items()}
                   
        elif isinstance(data, list):
            return [self._substitute_variables(item, variables) for item in data]
            
        else:
            return data


class ConfigurationManager:
    """Main configuration management system."""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT,
                 config_dir: str = None, master_key: str = None,
                 enable_hot_reload: bool = True, db_path: str = None):
        
        self.environment = environment
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.enable_hot_reload = enable_hot_reload
        
        # Core components
        self.security_manager = SecurityManager(master_key)
        self.store = ConfigurationStore(db_path)
        
        # Configuration data
        self.configurations: Dict[str, Any] = {}
        self.metadata: Dict[str, ConfigMetadata] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.templates: Dict[str, ConfigurationTemplate] = {}
        
        # Hot reload monitoring
        self.file_watchers: Dict[str, float] = {}  # file -> last_modified_time
        self.change_listeners: List[Callable[[str, Any, Any], None]] = []
        
        # Control
        self._lock = RLock()
        self._hot_reload_thread = None
        self._running = False
        
        # Load initial configurations
        self._load_initial_configurations()
        
        if enable_hot_reload:
            self.start_hot_reload_monitoring()
            
    def start_hot_reload_monitoring(self):
        """Start hot reload monitoring."""
        if self._running:
            return
            
        self._running = True
        self._hot_reload_thread = threading.Thread(
            target=self._hot_reload_loop,
            daemon=True,
            name="config-hot-reload"
        )
        self._hot_reload_thread.start()
        logger.info("Configuration hot reload monitoring started")
        
    def stop_hot_reload_monitoring(self):
        """Stop hot reload monitoring."""
        self._running = False
        if self._hot_reload_thread:
            self._hot_reload_thread.join(timeout=5)
        logger.info("Configuration hot reload monitoring stopped")
        
    def register_schema(self, name: str, schema: Dict[str, Any], version: str = "1.0"):
        """Register configuration schema."""
        with self._lock:
            self.schemas[name] = ConfigSchema(schema)
            self.store.store_schema(name, schema, version)
            logger.info(f"Registered schema: {name} v{version}")
            
    def register_template(self, name: str, template_data: Dict[str, Any]):
        """Register configuration template."""
        with self._lock:
            self.templates[name] = ConfigurationTemplate(template_data)
            logger.info(f"Registered template: {name}")
            
    def set_configuration(self, key: str, value: Any, user: str = "system",
                         validate_schema: bool = True, encrypt_if_sensitive: bool = True) -> bool:
        """Set configuration value."""
        with self._lock:
            try:
                # Validate against schema if available
                if validate_schema and key in self.schemas:
                    is_valid, errors = self.schemas[key].validate({key: value})
                    if not is_valid:
                        raise ValidationError(f"Validation failed for {key}: {errors}")
                        
                # Encrypt if sensitive and encryption is enabled
                processed_value = value
                is_encrypted = False
                
                if (encrypt_if_sensitive and 
                    self.security_manager.encryption_key and 
                    self.security_manager.is_sensitive_key(key) and 
                    isinstance(value, str)):
                    
                    processed_value = self.security_manager.encrypt_value(value)
                    is_encrypted = True
                    
                # Create metadata
                metadata = ConfigMetadata(
                    source=ConfigSource.MEMORY,
                    format=ConfigFormat.JSON,
                    timestamp=datetime.now(),
                    version=str(int(time.time())),
                    checksum=hashlib.md5(str(value).encode()).hexdigest(),
                    encrypted=is_encrypted,
                    schema_validated=validate_schema and key in self.schemas,
                    environment=self.environment
                )
                
                # Record change event
                old_value = self.configurations.get(key)
                change_event = ConfigChangeEvent(
                    event_id=str(uuid.uuid4()),
                    config_key=key,
                    old_value=old_value,
                    new_value=value,
                    timestamp=datetime.now(),
                    user=user,
                    environment=self.environment,
                    change_type="update" if key in self.configurations else "create",
                    validation_passed=True
                )
                
                # Update in-memory configuration
                self.configurations[key] = processed_value
                self.metadata[key] = metadata
                
                # Persist to storage
                self.store.store_configuration(key, processed_value, self.environment, 
                                             metadata.version, metadata)
                self.store.record_change(change_event)
                
                # Notify listeners
                self._notify_change_listeners(key, old_value, value)
                
                logger.info(f"Configuration updated: {key} (encrypted: {is_encrypted})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set configuration {key}: {e}")
                return False
                
    def get_configuration(self, key: str, default: Any = None, decrypt_if_encrypted: bool = True) -> Any:
        """Get configuration value."""
        with self._lock:
            if key not in self.configurations:
                # Try loading from storage
                stored = self.store.get_configuration(key, self.environment)
                if stored:
                    value, metadata = stored
                    self.configurations[key] = value
                    self.metadata[key] = metadata
                else:
                    return default
                    
            value = self.configurations[key]
            metadata = self.metadata.get(key)
            
            # Decrypt if necessary
            if (decrypt_if_encrypted and 
                metadata and metadata.encrypted and 
                self.security_manager.encryption_key):
                try:
                    value = self.security_manager.decrypt_value(value)
                except Exception as e:
                    logger.error(f"Failed to decrypt configuration {key}: {e}")
                    return default
                    
            return value
            
    def get_configuration_with_metadata(self, key: str, default: Any = None) -> Tuple[Any, Optional[ConfigMetadata]]:
        """Get configuration value with metadata."""
        value = self.get_configuration(key, default)
        metadata = self.metadata.get(key)
        return value, metadata
        
    def delete_configuration(self, key: str, user: str = "system") -> bool:
        """Delete configuration."""
        with self._lock:
            if key not in self.configurations:
                return False
                
            old_value = self.configurations[key]
            
            # Record change event
            change_event = ConfigChangeEvent(
                event_id=str(uuid.uuid4()),
                config_key=key,
                old_value=old_value,
                new_value=None,
                timestamp=datetime.now(),
                user=user,
                environment=self.environment,
                change_type="delete",
                validation_passed=True
            )
            
            # Remove from memory
            del self.configurations[key]
            self.metadata.pop(key, None)
            
            # Record change in storage (but don't delete from storage for audit trail)
            self.store.record_change(change_event)
            
            # Notify listeners
            self._notify_change_listeners(key, old_value, None)
            
            logger.info(f"Configuration deleted: {key}")
            return True
            
    def load_from_file(self, file_path: str, merge: bool = True, validate: bool = True) -> bool:
        """Load configuration from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Configuration file not found: {file_path}")
                return False
                
            # Determine format from extension
            format_map = {
                '.json': ConfigFormat.JSON,
                '.yaml': ConfigFormat.YAML,
                '.yml': ConfigFormat.YAML,
                '.toml': ConfigFormat.TOML,
                '.ini': ConfigFormat.INI,
                '.env': ConfigFormat.ENV
            }
            
            config_format = format_map.get(path.suffix.lower(), ConfigFormat.JSON)
            
            # Load data
            with open(path, 'r', encoding='utf-8') as f:
                if config_format == ConfigFormat.JSON:
                    data = json.load(f)
                elif config_format == ConfigFormat.YAML:
                    data = yaml.safe_load(f)
                else:
                    raise ConfigurationError(f"Unsupported format: {config_format}")
                    
            # Validate structure
            if not isinstance(data, dict):
                raise ConfigurationError("Configuration file must contain a JSON object")
                
            # Load configurations
            with self._lock:
                if not merge:
                    self.configurations.clear()
                    self.metadata.clear()
                    
                for key, value in data.items():
                    self.set_configuration(key, value, user="file_loader", validate_schema=validate)
                    
                # Track file for hot reload
                if self.enable_hot_reload:
                    self.file_watchers[str(path.absolute())] = path.stat().st_mtime
                    
            logger.info(f"Loaded configuration from {file_path} ({len(data)} keys)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            return False
            
    def save_to_file(self, file_path: str, keys: Optional[List[str]] = None, 
                    include_metadata: bool = False) -> bool:
        """Save configuration to file."""
        try:
            path = Path(file_path)
            config_format = ConfigFormat.JSON
            
            # Determine format from extension
            if path.suffix.lower() in ['.yaml', '.yml']:
                config_format = ConfigFormat.YAML
                
            # Prepare data
            with self._lock:
                if keys:
                    data = {k: self.get_configuration(k) for k in keys if k in self.configurations}
                else:
                    data = {k: self.get_configuration(k) for k in self.configurations}
                    
                # Sanitize sensitive data for export
                sanitized_data = self.security_manager.sanitize_for_logging(data)
                
                # Add metadata if requested
                if include_metadata:
                    export_data = {
                        'configuration': sanitized_data,
                        'metadata': {
                            k: self.metadata[k].to_dict() 
                            for k in sanitized_data.keys() 
                            if k in self.metadata
                        },
                        'export_info': {
                            'timestamp': datetime.now().isoformat(),
                            'environment': self.environment.value,
                            'exported_by': 'configuration_manager'
                        }
                    }
                else:
                    export_data = sanitized_data
                    
            # Write file
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                if config_format == ConfigFormat.YAML:
                    yaml.safe_dump(export_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(export_data, f, indent=2, default=str)
                    
            logger.info(f"Saved configuration to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            return False
            
    def create_from_template(self, template_name: str, variables: Dict[str, Any],
                           config_key: str, user: str = "template") -> bool:
        """Create configuration from template."""
        with self._lock:
            if template_name not in self.templates:
                logger.error(f"Template not found: {template_name}")
                return False
                
            try:
                template = self.templates[template_name]
                rendered_config = template.render(variables)
                
                # Set the rendered configuration
                return self.set_configuration(config_key, rendered_config, user=user)
                
            except Exception as e:
                logger.error(f"Failed to create configuration from template {template_name}: {e}")
                return False
                
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """Add configuration change listener."""
        self.change_listeners.append(listener)
        
    def remove_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """Remove configuration change listener."""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
            
    def get_configuration_history(self, key: str, limit: int = 50) -> List[ConfigChangeEvent]:
        """Get configuration change history."""
        return self.store.get_configuration_history(key, self.environment, limit)
        
    def rollback_configuration(self, key: str, target_event_id: str, user: str = "system") -> bool:
        """Rollback configuration to a previous state."""
        history = self.get_configuration_history(key)
        
        target_event = None
        for event in history:
            if event.event_id == target_event_id:
                target_event = event
                break
                
        if not target_event:
            logger.error(f"Target event not found: {target_event_id}")
            return False
            
        # Rollback to the old value from the target event
        if target_event.change_type == "delete":
            # If target was a delete, restore the old value
            return self.set_configuration(key, target_event.old_value, user=user)
        else:
            # For create/update, use the old value (state before the change)
            if target_event.old_value is not None:
                return self.set_configuration(key, target_event.old_value, user=user)
            else:
                # If there was no old value, delete the configuration
                return self.delete_configuration(key, user=user)
                
    def validate_all_configurations(self) -> Dict[str, List[str]]:
        """Validate all configurations against their schemas."""
        validation_results = {}
        
        with self._lock:
            for key in self.configurations:
                if key in self.schemas:
                    value = self.get_configuration(key)
                    is_valid, errors = self.schemas[key].validate({key: value})
                    if not is_valid:
                        validation_results[key] = errors
                        
        return validation_results
        
    def get_environment_configurations(self) -> Dict[str, Any]:
        """Get all configurations for current environment."""
        with self._lock:
            # Get decrypted configurations
            return {key: self.get_configuration(key) for key in self.configurations}
            
    def export_configuration_report(self) -> Dict[str, Any]:
        """Export comprehensive configuration report."""
        with self._lock:
            configs = self.get_environment_configurations()
            sanitized_configs = self.security_manager.sanitize_for_logging(configs)
            
            validation_results = self.validate_all_configurations()
            
            # Get statistics
            total_configs = len(configs)
            encrypted_configs = sum(1 for meta in self.metadata.values() if meta.encrypted)
            validated_configs = sum(1 for meta in self.metadata.values() if meta.schema_validated)
            
            return {
                'report_generated': datetime.now().isoformat(),
                'environment': self.environment.value,
                'statistics': {
                    'total_configurations': total_configs,
                    'encrypted_configurations': encrypted_configs,
                    'schema_validated_configurations': validated_configs,
                    'registered_schemas': len(self.schemas),
                    'registered_templates': len(self.templates)
                },
                'configurations': sanitized_configs,
                'metadata_summary': {
                    key: {
                        'source': meta.source.value,
                        'format': meta.format.value,
                        'encrypted': meta.encrypted,
                        'schema_validated': meta.schema_validated,
                        'last_updated': meta.timestamp.isoformat()
                    }
                    for key, meta in self.metadata.items()
                },
                'validation_results': validation_results,
                'hot_reload_enabled': self.enable_hot_reload,
                'monitored_files': list(self.file_watchers.keys())
            }
            
    def _load_initial_configurations(self):
        """Load initial configurations from config directory."""
        if not self.config_dir.exists():
            logger.info(f"Config directory does not exist: {self.config_dir}")
            return
            
        # Load environment-specific configurations
        env_config_file = self.config_dir / f"{self.environment.value}.json"
        if env_config_file.exists():
            self.load_from_file(str(env_config_file))
            
        # Load general configuration file
        general_config_file = self.config_dir / "config.json"
        if general_config_file.exists():
            self.load_from_file(str(general_config_file), merge=True)
            
        # Load YAML files
        for yaml_file in self.config_dir.glob("*.yaml"):
            if yaml_file.stem == self.environment.value or yaml_file.stem == "config":
                self.load_from_file(str(yaml_file), merge=True)
                
    def _hot_reload_loop(self):
        """Hot reload monitoring loop."""
        while self._running:
            try:
                files_changed = []
                
                with self._lock:
                    for file_path, last_mtime in list(self.file_watchers.items()):
                        try:
                            current_mtime = Path(file_path).stat().st_mtime
                            if current_mtime > last_mtime:
                                files_changed.append(file_path)
                                self.file_watchers[file_path] = current_mtime
                        except (OSError, IOError):
                            # File might have been deleted
                            continue
                            
                # Reload changed files
                for file_path in files_changed:
                    logger.info(f"Hot reloading configuration from: {file_path}")
                    self.load_from_file(file_path, merge=True, validate=True)
                    
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Hot reload monitoring error: {e}")
                time.sleep(5)
                
    def _notify_change_listeners(self, key: str, old_value: Any, new_value: Any):
        """Notify all change listeners."""
        for listener in self.change_listeners:
            try:
                listener(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Configuration change listener error: {e}")


# Global configuration manager instance
_config_manager = None


def get_config_manager(environment: Environment = None, **kwargs) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        env = environment or Environment.DEVELOPMENT
        _config_manager = ConfigurationManager(environment=env, **kwargs)
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    manager = get_config_manager()
    return manager.get_configuration(key, default)


def set_config(key: str, value: Any, **kwargs) -> bool:
    """Set configuration value."""
    manager = get_config_manager()
    return manager.set_configuration(key, value, **kwargs)


def register_config_schema(name: str, schema: Dict[str, Any], version: str = "1.0"):
    """Register configuration schema."""
    manager = get_config_manager()
    manager.register_schema(name, schema, version)


def add_config_change_listener(listener: Callable[[str, Any, Any], None]):
    """Add configuration change listener."""
    manager = get_config_manager()
    manager.add_change_listener(listener)


def load_config_from_file(file_path: str, merge: bool = True) -> bool:
    """Load configuration from file."""
    manager = get_config_manager()
    return manager.load_from_file(file_path, merge)


def export_config_report() -> Dict[str, Any]:
    """Export configuration report."""
    manager = get_config_manager()
    return manager.export_configuration_report()