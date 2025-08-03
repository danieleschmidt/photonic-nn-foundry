"""
Repository pattern implementations for data access.
"""

from typing import List, Optional, Dict, Any
import logging
from .connection import DatabaseManager, get_database
from .models import CircuitModel, ComponentModel, SimulationResult, PDKModel, PerformanceMetric

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common database operations."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_database()


class CircuitRepository(BaseRepository):
    """Repository for circuit data operations."""
    
    def save(self, circuit: CircuitModel) -> int:
        """Save circuit to database."""
        data = circuit.to_dict()
        
        # Check if circuit exists
        existing = self.find_by_name(circuit.name)
        
        if existing:
            # Update existing circuit
            query = """
                UPDATE circuits 
                SET model_hash = ?, circuit_data = ?, verilog_code = ?, 
                    metrics = ?, updated_at = ?, version = ?
                WHERE name = ?
            """
            params = (
                data['model_hash'], data['circuit_data'], data['verilog_code'],
                data['metrics'], data['updated_at'], data['version'], circuit.name
            )
            self.db.execute_update(query, params)
            return existing.get('id')
        else:
            # Insert new circuit
            query = """
                INSERT INTO circuits (name, model_hash, circuit_data, verilog_code, 
                                    metrics, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                data['name'], data['model_hash'], data['circuit_data'],
                data['verilog_code'], data['metrics'], data['created_at'],
                data['updated_at'], data['version']
            )
            return self.db.execute_insert(query, params)
            
    def find_by_id(self, circuit_id: int) -> Optional[CircuitModel]:
        """Find circuit by ID."""
        query = "SELECT * FROM circuits WHERE id = ?"
        results = self.db.execute_query(query, (circuit_id,))
        
        if results:
            row_data = dict(results[0])
            return CircuitModel.from_dict(row_data)
        return None
        
    def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find circuit by name."""
        query = "SELECT * FROM circuits WHERE name = ?"
        results = self.db.execute_query(query, (name,))
        
        if results:
            return dict(results[0])
        return None
        
    def find_by_hash(self, model_hash: str) -> Optional[CircuitModel]:
        """Find circuit by model hash."""
        query = "SELECT * FROM circuits WHERE model_hash = ?"
        results = self.db.execute_query(query, (model_hash,))
        
        if results:
            row_data = dict(results[0])
            return CircuitModel.from_dict(row_data)
        return None
        
    def list_all(self, limit: int = 100, offset: int = 0) -> List[CircuitModel]:
        """List all circuits with pagination."""
        query = """
            SELECT * FROM circuits 
            ORDER BY updated_at DESC 
            LIMIT ? OFFSET ?
        """
        results = self.db.execute_query(query, (limit, offset))
        
        circuits = []
        for row in results:
            row_data = dict(row)
            circuits.append(CircuitModel.from_dict(row_data))
            
        return circuits
        
    def search_by_name_pattern(self, pattern: str) -> List[CircuitModel]:
        """Search circuits by name pattern."""
        query = "SELECT * FROM circuits WHERE name LIKE ? ORDER BY name"
        results = self.db.execute_query(query, (f"%{pattern}%",))
        
        circuits = []
        for row in results:
            row_data = dict(row)
            circuits.append(CircuitModel.from_dict(row_data))
            
        return circuits
        
    def delete_by_name(self, name: str) -> bool:
        """Delete circuit by name."""
        affected = self.db.execute_update("DELETE FROM circuits WHERE name = ?", (name,))
        return affected > 0
        
    def get_circuit_stats(self) -> Dict[str, Any]:
        """Get circuit statistics."""
        stats = {}
        
        # Total circuits
        result = self.db.execute_query("SELECT COUNT(*) as count FROM circuits")
        stats['total_circuits'] = result[0]['count']
        
        # Circuits with Verilog
        result = self.db.execute_query(
            "SELECT COUNT(*) as count FROM circuits WHERE verilog_code IS NOT NULL"
        )
        stats['circuits_with_verilog'] = result[0]['count']
        
        # Average component count
        result = self.db.execute_query("""
            SELECT AVG(json_extract(circuit_data, '$.total_components')) as avg_components
            FROM circuits
        """)
        stats['avg_components'] = result[0]['avg_components'] or 0
        
        # Most recent update
        result = self.db.execute_query(
            "SELECT MAX(updated_at) as latest FROM circuits"
        )
        stats['latest_update'] = result[0]['latest']
        
        return stats


class ComponentRepository(BaseRepository):
    """Repository for component data operations."""
    
    def save(self, component: ComponentModel) -> int:
        """Save component to database."""
        data = component.to_dict()
        
        # Check if component exists
        existing = self.find_by_name_type_pdk(
            component.spec.name, 
            component.spec.component_type.value,
            component.spec.pdk
        )
        
        if existing:
            # Update existing component
            query = """
                UPDATE components 
                SET component_data = ?, verilog_template = ?, parameters = ?
                WHERE name = ? AND type = ? AND pdk = ?
            """
            params = (
                data['component_data'], data['verilog_template'], data['parameters'],
                data['name'], data['type'], data['pdk']
            )
            self.db.execute_update(query, params)
            return existing['id']
        else:
            # Insert new component
            query = """
                INSERT INTO components (name, type, pdk, component_data, 
                                      verilog_template, parameters, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                data['name'], data['type'], data['pdk'], data['component_data'],
                data['verilog_template'], data['parameters'], data['created_at']
            )
            return self.db.execute_insert(query, params)
            
    def find_by_name_type_pdk(self, name: str, component_type: str, pdk: str) -> Optional[Dict[str, Any]]:
        """Find component by name, type, and PDK."""
        query = "SELECT * FROM components WHERE name = ? AND type = ? AND pdk = ?"
        results = self.db.execute_query(query, (name, component_type, pdk))
        
        if results:
            return dict(results[0])
        return None
        
    def list_by_type(self, component_type: str, pdk: Optional[str] = None) -> List[ComponentModel]:
        """List components by type, optionally filtered by PDK."""
        if pdk:
            query = "SELECT * FROM components WHERE type = ? AND pdk = ? ORDER BY name"
            results = self.db.execute_query(query, (component_type, pdk))
        else:
            query = "SELECT * FROM components WHERE type = ? ORDER BY name"
            results = self.db.execute_query(query, (component_type,))
            
        components = []
        for row in results:
            row_data = dict(row)
            components.append(ComponentModel.from_dict(row_data))
            
        return components
        
    def list_by_pdk(self, pdk: str) -> List[ComponentModel]:
        """List all components for a specific PDK."""
        query = "SELECT * FROM components WHERE pdk = ? ORDER BY type, name"
        results = self.db.execute_query(query, (pdk,))
        
        components = []
        for row in results:
            row_data = dict(row)
            components.append(ComponentModel.from_dict(row_data))
            
        return components


class SimulationRepository(BaseRepository):
    """Repository for simulation result operations."""
    
    def save(self, simulation: SimulationResult) -> int:
        """Save simulation result to database."""
        data = simulation.to_dict()
        
        query = """
            INSERT INTO simulation_results 
            (circuit_id, input_data, output_data, simulation_config, 
             execution_time, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            data['circuit_id'], data['input_data'], data['output_data'],
            data['simulation_config'], data['execution_time'], data['timestamp']
        )
        return self.db.execute_insert(query, params)
        
    def find_by_circuit_id(self, circuit_id: int, limit: int = 50) -> List[SimulationResult]:
        """Find simulation results for a circuit."""
        query = """
            SELECT * FROM simulation_results 
            WHERE circuit_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        results = self.db.execute_query(query, (circuit_id, limit))
        
        simulations = []
        for row in results:
            row_data = dict(row)
            simulations.append(SimulationResult.from_dict(row_data))
            
        return simulations
        
    def get_average_execution_time(self, circuit_id: int) -> float:
        """Get average execution time for a circuit."""
        query = """
            SELECT AVG(execution_time) as avg_time 
            FROM simulation_results 
            WHERE circuit_id = ?
        """
        result = self.db.execute_query(query, (circuit_id,))
        return result[0]['avg_time'] or 0.0
        
    def cleanup_old_results(self, days: int = 30) -> int:
        """Clean up simulation results older than specified days."""
        query = """
            DELETE FROM simulation_results 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days)
        return self.db.execute_update(query)


class PDKRepository(BaseRepository):
    """Repository for PDK data operations."""
    
    def save(self, pdk: PDKModel) -> int:
        """Save PDK to database."""
        data = pdk.to_dict()
        
        # Check if PDK exists
        existing = self.find_by_name(pdk.name)
        
        if existing:
            # Update existing PDK
            query = """
                UPDATE pdks 
                SET version = ?, description = ?, config_data = ?, component_library = ?
                WHERE name = ?
            """
            params = (
                data['version'], data['description'], data['config_data'],
                data['component_library'], pdk.name
            )
            self.db.execute_update(query, params)
            return existing['id']
        else:
            # Insert new PDK
            query = """
                INSERT INTO pdks (name, version, description, config_data, 
                                component_library, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            params = (
                data['name'], data['version'], data['description'],
                data['config_data'], data['component_library'],
                'CURRENT_TIMESTAMP'
            )
            return self.db.execute_insert(query, params)
            
    def find_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find PDK by name."""
        query = "SELECT * FROM pdks WHERE name = ?"
        results = self.db.execute_query(query, (name,))
        
        if results:
            return dict(results[0])
        return None
        
    def list_all(self) -> List[PDKModel]:
        """List all available PDKs."""
        query = "SELECT * FROM pdks ORDER BY name"
        results = self.db.execute_query(query)
        
        pdks = []
        for row in results:
            row_data = dict(row)
            pdks.append(PDKModel.from_dict(row_data))
            
        return pdks


class MetricsRepository(BaseRepository):
    """Repository for performance metrics operations."""
    
    def save(self, metric: PerformanceMetric) -> int:
        """Save performance metric to database."""
        data = metric.to_dict()
        
        query = """
            INSERT INTO performance_metrics 
            (circuit_id, metric_name, metric_value, unit, 
             measurement_config, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            data['circuit_id'], data['metric_name'], data['metric_value'],
            data['unit'], data['measurement_config'], data['timestamp']
        )
        return self.db.execute_insert(query, params)
        
    def find_by_circuit_and_metric(self, circuit_id: int, metric_name: str) -> List[PerformanceMetric]:
        """Find metrics for a circuit and metric name."""
        query = """
            SELECT * FROM performance_metrics 
            WHERE circuit_id = ? AND metric_name = ? 
            ORDER BY timestamp DESC
        """
        results = self.db.execute_query(query, (circuit_id, metric_name))
        
        metrics = []
        for row in results:
            row_data = dict(row)
            metrics.append(PerformanceMetric.from_dict(row_data))
            
        return metrics
        
    def get_latest_metrics(self, circuit_id: int) -> Dict[str, PerformanceMetric]:
        """Get latest value for each metric type for a circuit."""
        query = """
            SELECT * FROM performance_metrics p1
            WHERE circuit_id = ? 
            AND timestamp = (
                SELECT MAX(timestamp) 
                FROM performance_metrics p2 
                WHERE p2.circuit_id = p1.circuit_id 
                AND p2.metric_name = p1.metric_name
            )
        """
        results = self.db.execute_query(query, (circuit_id,))
        
        metrics = {}
        for row in results:
            row_data = dict(row)
            metric = PerformanceMetric.from_dict(row_data)
            metrics[metric.metric_name] = metric
            
        return metrics