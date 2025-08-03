"""
Database connection and management.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from contextlib import contextmanager
import threading
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_path: str = "circuits.db"
    cache_enabled: bool = True
    cache_dir: str = ".cache/circuits"
    max_cached_circuits: int = 1000
    connection_timeout: int = 30
    enable_wal_mode: bool = True


class DatabaseManager:
    """Manages SQLite database connections and operations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self.cache_dir = Path(config.cache_dir)
        self._local = threading.local()
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            # Enable WAL mode for better concurrency
            if self.config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
                
            # Create tables
            self._create_tables(conn)
            
            logger.info(f"Database initialized at {self.db_path}")
            
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables."""
        
        # Circuits table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS circuits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                model_hash TEXT NOT NULL,
                circuit_data TEXT NOT NULL,
                verilog_code TEXT,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1
            )
        """)
        
        # Components table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS components (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                pdk TEXT NOT NULL,
                component_data TEXT NOT NULL,
                verilog_template TEXT,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, type, pdk)
            )
        """)
        
        # Simulation results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS simulation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                circuit_id INTEGER NOT NULL,
                input_data TEXT NOT NULL,
                output_data TEXT NOT NULL,
                simulation_config TEXT,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (circuit_id) REFERENCES circuits (id)
            )
        """)
        
        # PDK information table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pdks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                config_data TEXT NOT NULL,
                component_library TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Performance metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                circuit_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                unit TEXT,
                measurement_config TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (circuit_id) REFERENCES circuits (id)
            )
        """)
        
        # Create indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_circuits_name ON circuits (name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_circuits_hash ON circuits (model_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_components_type_pdk ON components (type, pdk)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_simulation_circuit ON simulation_results (circuit_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_circuit ON performance_metrics (circuit_id)")
        
        conn.commit()
        
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=self.config.connection_timeout,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            # Connection remains open for reuse within the thread
            pass
            
    def close_connection(self):
        """Close thread-local connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
            
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute SELECT query and return results."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
            
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query and return affected rows."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
            
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT query and return last row ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid
            
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            stats = {}
            
            # Table row counts
            tables = ['circuits', 'components', 'simulation_results', 'pdks', 'performance_metrics']
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
                
            # Database size
            cursor = conn.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats['database_size_bytes'] = page_count * page_size
            
            # Cache statistics
            stats['cache_enabled'] = self.config.cache_enabled
            stats['cache_dir'] = str(self.cache_dir)
            if self.cache_dir.exists():
                cache_files = list(self.cache_dir.glob("*.json"))
                stats['cached_circuits'] = len(cache_files)
            else:
                stats['cached_circuits'] = 0
                
            return stats
            
    def vacuum_database(self):
        """Optimize database by running VACUUM."""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuum completed")
            
    def backup_database(self, backup_path: str):
        """Create database backup."""
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.get_connection() as conn:
            with sqlite3.connect(backup_path) as backup_conn:
                conn.backup(backup_conn)
                
        logger.info(f"Database backed up to {backup_path}")
        
    def restore_database(self, backup_path: str):
        """Restore database from backup."""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
        # Close existing connections
        self.close_connection()
        
        # Replace database file
        if self.db_path.exists():
            self.db_path.rename(f"{self.db_path}.old")
            
        with sqlite3.connect(backup_path) as backup_conn:
            with sqlite3.connect(self.db_path) as conn:
                backup_conn.backup(conn)
                
        logger.info(f"Database restored from {backup_path}")


# Global database instance
_database_manager: Optional[DatabaseManager] = None


def get_database(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Get global database manager instance."""
    global _database_manager
    
    if _database_manager is None:
        if config is None:
            # Load config from environment variables
            config = DatabaseConfig(
                db_path=os.getenv('CIRCUIT_DB_PATH', 'circuits.db'),
                cache_enabled=os.getenv('CIRCUIT_CACHE_ENABLED', 'true').lower() == 'true',
                cache_dir=os.getenv('CIRCUIT_CACHE_DIR', '.cache/circuits'),
                max_cached_circuits=int(os.getenv('MAX_CACHED_CIRCUITS', '1000'))
            )
            
        _database_manager = DatabaseManager(config)
        
    return _database_manager


def close_database():
    """Close global database connection."""
    global _database_manager
    if _database_manager:
        _database_manager.close_connection()
        _database_manager = None