"""
Initial database schema migration for Photonic Neural Network Foundry.

Migration: 001_initial_schema
Created: 2025-08-03
Description: Create initial tables for circuits, models, and metrics
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def upgrade(db_connection) -> Dict[str, Any]:
    """
    Apply the migration to upgrade the database schema.
    
    Args:
        db_connection: Database connection object
        
    Returns:
        Migration result information
    """
    logger.info("Applying migration 001_initial_schema...")
    
    migration_sql = [
        # Circuits table
        """
        CREATE TABLE IF NOT EXISTS circuits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            model_hash TEXT NOT NULL,
            circuit_data TEXT NOT NULL,  -- JSON blob
            verilog_code TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        # Circuit metrics table
        """
        CREATE TABLE IF NOT EXISTS circuit_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit_id INTEGER NOT NULL,
            energy_per_op REAL NOT NULL,
            latency REAL NOT NULL,
            area REAL NOT NULL,
            power REAL NOT NULL,
            throughput REAL NOT NULL,
            accuracy REAL NOT NULL,
            loss REAL DEFAULT 0.0,
            crosstalk REAL DEFAULT -30.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (circuit_id) REFERENCES circuits (id) ON DELETE CASCADE
        );
        """,
        
        # Model analysis table
        """
        CREATE TABLE IF NOT EXISTS model_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_hash TEXT NOT NULL,
            analysis_data TEXT NOT NULL,  -- JSON blob
            compatibility_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        # Optimization results table
        """
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit_id INTEGER NOT NULL,
            optimization_type TEXT NOT NULL,
            parameters TEXT NOT NULL,  -- JSON blob
            results TEXT NOT NULL,     -- JSON blob
            improvement_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (circuit_id) REFERENCES circuits (id) ON DELETE CASCADE
        );
        """,
        
        # Benchmark comparisons table
        """
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            model_type TEXT NOT NULL,
            dataset TEXT,
            electronic_metrics TEXT NOT NULL,  -- JSON blob
            photonic_metrics TEXT NOT NULL,    -- JSON blob
            speedup_factor REAL,
            energy_efficiency REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        # System configuration table
        """
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key TEXT NOT NULL UNIQUE,
            config_value TEXT NOT NULL,
            config_type TEXT DEFAULT 'string',
            description TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        # Cache metadata table
        """
        CREATE TABLE IF NOT EXISTS cache_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT NOT NULL UNIQUE,
            cache_type TEXT NOT NULL,
            size_bytes INTEGER,
            hit_count INTEGER DEFAULT 0,
            miss_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]
    
    # Create indexes
    index_sql = [
        "CREATE INDEX IF NOT EXISTS idx_circuits_name ON circuits (name);",
        "CREATE INDEX IF NOT EXISTS idx_circuits_model_hash ON circuits (model_hash);",
        "CREATE INDEX IF NOT EXISTS idx_circuits_created_at ON circuits (created_at);",
        "CREATE INDEX IF NOT EXISTS idx_circuit_metrics_circuit_id ON circuit_metrics (circuit_id);",
        "CREATE INDEX IF NOT EXISTS idx_model_analysis_hash ON model_analysis (model_hash);",
        "CREATE INDEX IF NOT EXISTS idx_optimization_circuit_id ON optimization_results (circuit_id);",
        "CREATE INDEX IF NOT EXISTS idx_benchmarks_name ON benchmarks (name);",
        "CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config (config_key);",
        "CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_metadata (cache_key);"
    ]
    
    executed_statements = []
    
    try:
        cursor = db_connection.cursor()
        
        # Execute table creation
        for sql in migration_sql:
            cursor.execute(sql)
            executed_statements.append(sql.strip().split('\n')[0])
            
        # Execute index creation
        for sql in index_sql:
            cursor.execute(sql)
            executed_statements.append(sql.strip())
            
        # Insert initial configuration
        initial_config = [
            ('schema_version', '001', 'string', 'Current database schema version'),
            ('last_migration', '001_initial_schema', 'string', 'Last applied migration'),
            ('created_at', '2025-08-03T00:00:00Z', 'timestamp', 'Database creation timestamp'),
            ('default_pdk', 'skywater130', 'string', 'Default process design kit'),
            ('default_wavelength', '1550', 'number', 'Default operating wavelength (nm)'),
            ('optimization_enabled', 'true', 'boolean', 'Enable circuit optimization by default')
        ]
        
        for key, value, type_name, description in initial_config:
            cursor.execute(
                "INSERT OR IGNORE INTO system_config (config_key, config_value, config_type, description) VALUES (?, ?, ?, ?)",
                (key, value, type_name, description)
            )
            executed_statements.append(f"Config: {key} = {value}")
            
        db_connection.commit()
        
        logger.info("Migration 001_initial_schema completed successfully")
        
        return {
            'migration_id': '001_initial_schema',
            'status': 'success',
            'statements_executed': len(executed_statements),
            'details': executed_statements
        }
        
    except Exception as e:
        db_connection.rollback()
        logger.error(f"Migration 001_initial_schema failed: {e}")
        raise


def downgrade(db_connection) -> Dict[str, Any]:
    """
    Apply the migration to downgrade the database schema.
    
    Args:
        db_connection: Database connection object
        
    Returns:
        Migration result information
    """
    logger.info("Downgrading migration 001_initial_schema...")
    
    # Drop tables in reverse dependency order
    downgrade_sql = [
        "DROP TABLE IF EXISTS cache_metadata;",
        "DROP TABLE IF EXISTS system_config;",
        "DROP TABLE IF EXISTS benchmarks;",
        "DROP TABLE IF EXISTS optimization_results;",
        "DROP TABLE IF EXISTS model_analysis;",
        "DROP TABLE IF EXISTS circuit_metrics;",
        "DROP TABLE IF EXISTS circuits;"
    ]
    
    executed_statements = []
    
    try:
        cursor = db_connection.cursor()
        
        for sql in downgrade_sql:
            cursor.execute(sql)
            executed_statements.append(sql.strip())
            
        db_connection.commit()
        
        logger.info("Migration 001_initial_schema downgraded successfully")
        
        return {
            'migration_id': '001_initial_schema',
            'status': 'downgraded',
            'statements_executed': len(executed_statements),
            'details': executed_statements
        }
        
    except Exception as e:
        db_connection.rollback()
        logger.error(f"Migration 001_initial_schema downgrade failed: {e}")
        raise


def validate(db_connection) -> Dict[str, Any]:
    """
    Validate that the migration was applied correctly.
    
    Args:
        db_connection: Database connection object
        
    Returns:
        Validation result information
    """
    logger.info("Validating migration 001_initial_schema...")
    
    expected_tables = [
        'circuits',
        'circuit_metrics', 
        'model_analysis',
        'optimization_results',
        'benchmarks',
        'system_config',
        'cache_metadata'
    ]
    
    validation_results = {
        'migration_id': '001_initial_schema',
        'status': 'valid',
        'tables_found': [],
        'tables_missing': [],
        'indexes_validated': 0,
        'config_entries': 0
    }
    
    try:
        cursor = db_connection.cursor()
        
        # Check for expected tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        for table in expected_tables:
            if table in existing_tables:
                validation_results['tables_found'].append(table)
            else:
                validation_results['tables_missing'].append(table)
                
        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';")
        indexes = cursor.fetchall()
        validation_results['indexes_validated'] = len(indexes)
        
        # Check configuration entries
        cursor.execute("SELECT COUNT(*) FROM system_config;")
        config_count = cursor.fetchone()[0]
        validation_results['config_entries'] = config_count
        
        # Determine overall status
        if validation_results['tables_missing']:
            validation_results['status'] = 'invalid'
            
        logger.info(f"Migration validation complete: {validation_results['status']}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Migration validation failed: {e}")
        validation_results['status'] = 'error'
        validation_results['error'] = str(e)
        return validation_results


# Migration metadata
MIGRATION_INFO = {
    'id': '001_initial_schema',
    'description': 'Create initial database schema for photonic circuits and models',
    'version': '0.1.0',
    'dependencies': [],
    'reversible': True,
    'created_at': '2025-08-03T00:00:00Z'
}