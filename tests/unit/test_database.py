"""
Unit tests for database functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from src.photonic_foundry.database import (
    DatabaseManager, DatabaseConfig, CircuitRepository, 
    ComponentRepository, get_circuit_cache
)
from src.photonic_foundry.database.models import (
    CircuitModel, ComponentModel, ComponentSpec, ComponentType,
    CircuitMetrics, SimulationResult
)


class TestDatabaseManager:
    """Test database manager functionality."""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create temporary database configuration."""
        temp_dir = tempfile.mkdtemp()
        config = DatabaseConfig(
            db_path=os.path.join(temp_dir, "test_circuits.db"),
            cache_enabled=True,
            cache_dir=os.path.join(temp_dir, "cache"),
            max_cached_circuits=100
        )
        yield config
        # Cleanup handled by tempfile
        
    @pytest.fixture
    def db_manager(self, temp_db_config):
        """Create database manager with temporary database."""
        return DatabaseManager(temp_db_config)
        
    def test_database_initialization(self, db_manager):
        """Test database is properly initialized."""
        assert db_manager.db_path.exists()
        
        # Check tables exist
        with db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
        expected_tables = {
            'circuits', 'components', 'simulation_results', 
            'pdks', 'performance_metrics'
        }
        assert expected_tables.issubset(set(tables))
        
    def test_database_stats(self, db_manager):
        """Test database statistics retrieval."""
        stats = db_manager.get_database_stats()
        
        assert 'circuits_count' in stats
        assert 'components_count' in stats
        assert 'database_size_bytes' in stats
        assert stats['circuits_count'] == 0  # Empty database
        
    def test_database_backup_restore(self, db_manager):
        """Test database backup and restore functionality."""
        # Add some test data
        with db_manager.get_connection() as conn:
            conn.execute(
                "INSERT INTO circuits (name, model_hash, circuit_data) VALUES (?, ?, ?)",
                ("test_circuit", "hash123", '{"test": "data"}')
            )
            
        # Create backup
        backup_path = db_manager.db_path.parent / "backup.db"
        db_manager.backup_database(str(backup_path))
        assert backup_path.exists()
        
        # Verify backup contains data
        backup_manager = DatabaseManager(DatabaseConfig(db_path=str(backup_path)))
        stats = backup_manager.get_database_stats()
        assert stats['circuits_count'] == 1


class TestCircuitModel:
    """Test circuit model functionality."""
    
    def test_circuit_creation(self):
        """Test creating a circuit model."""
        circuit_data = {
            'layers': [
                {'type': 'Linear', 'input_size': 784, 'output_size': 256}
            ],
            'total_components': 200704
        }
        
        circuit = CircuitModel("test_circuit", circuit_data)
        
        assert circuit.name == "test_circuit"
        assert circuit.circuit_data == circuit_data
        assert circuit.model_hash is not None
        assert len(circuit.model_hash) == 16  # SHA256 truncated
        
    def test_circuit_hash_consistency(self):
        """Test that circuit hash is consistent for same data."""
        circuit_data = {'test': 'data'}
        
        circuit1 = CircuitModel("test1", circuit_data)
        circuit2 = CircuitModel("test2", circuit_data)
        
        assert circuit1.model_hash == circuit2.model_hash
        
    def test_circuit_update(self):
        """Test updating circuit data."""
        circuit = CircuitModel("test_circuit", {'old': 'data'})
        original_hash = circuit.model_hash
        original_version = circuit.version
        
        circuit.update_data({'new': 'data'})
        
        assert circuit.model_hash != original_hash
        assert circuit.version == original_version + 1
        assert circuit.circuit_data == {'new': 'data'}
        
    def test_circuit_serialization(self):
        """Test circuit serialization and deserialization."""
        metrics = CircuitMetrics(
            energy_per_op=0.5, latency=100, area=0.1, power=10.0,
            throughput=1000, accuracy=0.98, loss=0.5, crosstalk=-30
        )
        
        circuit = CircuitModel("test_circuit", {'test': 'data'})
        circuit.set_verilog("module test(); endmodule")
        circuit.set_metrics(metrics)
        
        # Serialize
        data = circuit.to_dict()
        
        # Deserialize
        restored_circuit = CircuitModel.from_dict(data)
        
        assert restored_circuit.name == circuit.name
        assert restored_circuit.circuit_data == circuit.circuit_data
        assert restored_circuit.verilog_code == circuit.verilog_code
        assert restored_circuit.metrics.energy_per_op == metrics.energy_per_op


class TestComponentModel:
    """Test component model functionality."""
    
    def test_component_creation(self):
        """Test creating a component model."""
        spec = ComponentSpec(
            name="test_mzi",
            component_type=ComponentType.MZI,
            pdk="skywater130",
            parameters={'phase_bits': 8, 'loss_db': 0.1}
        )
        
        component = ComponentModel(spec)
        
        assert component.spec.name == "test_mzi"
        assert component.spec.component_type == ComponentType.MZI
        assert component.spec.pdk == "skywater130"
        
    def test_component_serialization(self):
        """Test component serialization."""
        spec = ComponentSpec(
            name="test_component",
            component_type=ComponentType.RING,
            pdk="generic",
            parameters={'radius': 10, 'coupling': 0.1},
            verilog_template="module ring(); endmodule"
        )
        
        component = ComponentModel(spec)
        
        # Serialize
        data = component.to_dict()
        
        # Deserialize
        restored_component = ComponentModel.from_dict(data)
        
        assert restored_component.spec.name == spec.name
        assert restored_component.spec.component_type == spec.component_type
        assert restored_component.spec.pdk == spec.pdk
        assert restored_component.spec.parameters == spec.parameters


class TestCircuitRepository:
    """Test circuit repository functionality."""
    
    @pytest.fixture
    def circuit_repo(self, temp_db_config):
        """Create circuit repository with temporary database."""
        db_manager = DatabaseManager(temp_db_config)
        return CircuitRepository(db_manager)
        
    @pytest.fixture
    def sample_circuit(self):
        """Create sample circuit for testing."""
        circuit_data = {
            'layers': [
                {'type': 'Linear', 'input_size': 784, 'output_size': 256},
                {'type': 'Linear', 'input_size': 256, 'output_size': 10}
            ],
            'total_components': 202762
        }
        return CircuitModel("mnist_mlp", circuit_data)
        
    def test_save_and_find_circuit(self, circuit_repo, sample_circuit):
        """Test saving and finding circuits."""
        # Save circuit
        circuit_id = circuit_repo.save(sample_circuit)
        assert circuit_id > 0
        
        # Find by ID
        found_circuit = circuit_repo.find_by_id(circuit_id)
        assert found_circuit is not None
        assert found_circuit.name == sample_circuit.name
        
        # Find by name
        found_data = circuit_repo.find_by_name(sample_circuit.name)
        assert found_data is not None
        assert found_data['name'] == sample_circuit.name
        
    def test_update_circuit(self, circuit_repo, sample_circuit):
        """Test updating existing circuit."""
        # Save original
        circuit_id = circuit_repo.save(sample_circuit)
        
        # Update circuit
        sample_circuit.update_data({'updated': True})
        updated_id = circuit_repo.save(sample_circuit)
        
        # Should be same ID (update, not insert)
        assert updated_id == circuit_id
        
        # Verify update
        found_circuit = circuit_repo.find_by_id(circuit_id)
        assert found_circuit.circuit_data['updated'] is True
        
    def test_list_circuits(self, circuit_repo, sample_circuit):
        """Test listing circuits."""
        # Save multiple circuits
        circuit_repo.save(sample_circuit)
        
        circuit2 = CircuitModel("test_circuit_2", {'test': 'data2'})
        circuit_repo.save(circuit2)
        
        # List circuits
        circuits = circuit_repo.list_all(limit=10)
        assert len(circuits) == 2
        
        # Check ordering (most recent first)
        assert circuits[0].name in [sample_circuit.name, circuit2.name]
        
    def test_search_circuits(self, circuit_repo, sample_circuit):
        """Test searching circuits by name pattern."""
        circuit_repo.save(sample_circuit)
        
        # Search for circuits containing 'mnist'
        results = circuit_repo.search_by_name_pattern('mnist')
        assert len(results) == 1
        assert results[0].name == sample_circuit.name
        
        # Search for non-existent pattern
        results = circuit_repo.search_by_name_pattern('nonexistent')
        assert len(results) == 0
        
    def test_circuit_stats(self, circuit_repo, sample_circuit):
        """Test circuit statistics."""
        # Initially empty
        stats = circuit_repo.get_circuit_stats()
        assert stats['total_circuits'] == 0
        
        # Add circuit with Verilog
        sample_circuit.set_verilog("module test(); endmodule")
        circuit_repo.save(sample_circuit)
        
        # Check updated stats
        stats = circuit_repo.get_circuit_stats()
        assert stats['total_circuits'] == 1
        assert stats['circuits_with_verilog'] == 1


class TestSimulationResult:
    """Test simulation result functionality."""
    
    def test_simulation_result_creation(self):
        """Test creating simulation results."""
        input_data = np.array([1.0, 2.0, 3.0])
        output_data = np.array([0.1, 0.9, 0.0])
        
        result = SimulationResult(
            circuit_id=1,
            input_data=input_data,
            output_data=output_data,
            simulation_config={'precision': 8},
            execution_time=0.001,
            timestamp=datetime.now()
        )
        
        assert result.circuit_id == 1
        assert np.array_equal(result.input_data, input_data)
        assert np.array_equal(result.output_data, output_data)
        
    def test_accuracy_calculation(self):
        """Test accuracy calculation vs expected output."""
        output_data = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.1, 1.9, 3.1])
        
        result = SimulationResult(
            circuit_id=1,
            input_data=np.array([0]),
            output_data=output_data,
            simulation_config={},
            execution_time=0.001,
            timestamp=datetime.now()
        )
        
        accuracy = result.get_accuracy_vs_expected(expected)
        assert 0.9 < accuracy < 1.0  # Should be high accuracy
        
    def test_snr_calculation(self):
        """Test SNR calculation."""
        # Clean signal
        clean_signal = np.array([1.0, 1.0, 1.0, 1.0])
        
        result = SimulationResult(
            circuit_id=1,
            input_data=np.array([0]),
            output_data=clean_signal,
            simulation_config={},
            execution_time=0.001,
            timestamp=datetime.now()
        )
        
        snr = result.get_snr_db()
        assert snr > 0  # Should have positive SNR
        
    def test_simulation_serialization(self):
        """Test simulation result serialization."""
        input_data = np.random.randn(10)
        output_data = np.random.randn(5)
        
        result = SimulationResult(
            circuit_id=1,
            input_data=input_data,
            output_data=output_data,
            simulation_config={'test': 'config'},
            execution_time=0.123,
            timestamp=datetime.now()
        )
        
        # Serialize
        data = result.to_dict()
        
        # Deserialize
        restored_result = SimulationResult.from_dict(data)
        
        assert restored_result.circuit_id == result.circuit_id
        assert np.array_equal(restored_result.input_data, result.input_data)
        assert np.array_equal(restored_result.output_data, result.output_data)
        assert restored_result.simulation_config == result.simulation_config


class TestCircuitCache:
    """Test circuit caching functionality."""
    
    @pytest.fixture
    def temp_cache(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        cache = get_circuit_cache()
        cache.cache_dir = Path(temp_dir)
        cache.cache_dir.mkdir(exist_ok=True)
        return cache
        
    def test_cache_circuit(self, temp_cache):
        """Test caching circuit data."""
        circuit_data = {'test': 'circuit', 'layers': []}
        
        # Cache circuit
        cache_key = temp_cache.put_circuit(circuit_data)
        assert cache_key is not None
        
        # Retrieve from cache
        cached_data = temp_cache.get_circuit(circuit_data)
        assert cached_data is not None
        assert cached_data['circuit_data'] == circuit_data
        
    def test_cache_miss(self, temp_cache):
        """Test cache miss behavior."""
        circuit_data = {'test': 'circuit'}
        
        # Try to get non-existent circuit
        cached_data = temp_cache.get_circuit(circuit_data)
        assert cached_data is None
        
    def test_cache_stats(self, temp_cache):
        """Test cache statistics."""
        stats = temp_cache.get_cache_stats()
        
        assert 'total_entries' in stats
        assert 'total_size_mb' in stats
        assert 'cache_dir' in stats
        assert stats['total_entries'] == 0  # Empty cache
        
    def test_cache_cleanup(self, temp_cache):
        """Test cache cleanup functionality."""
        circuit_data = {'test': 'circuit'}
        
        # Add circuit to cache
        temp_cache.put_circuit(circuit_data)
        assert temp_cache.get_cache_stats()['total_entries'] == 1
        
        # Clear cache
        temp_cache.clear_cache()
        assert temp_cache.get_cache_stats()['total_entries'] == 0


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database functionality."""
    
    @pytest.fixture
    def temp_db_config(self):
        """Create temporary database configuration."""
        temp_dir = tempfile.mkdtemp()
        return DatabaseConfig(
            db_path=os.path.join(temp_dir, "integration_test.db"),
            cache_enabled=True,
            cache_dir=os.path.join(temp_dir, "cache"),
            max_cached_circuits=10
        )
        
    def test_full_circuit_workflow(self, temp_db_config):
        """Test complete circuit save/load workflow."""
        # Create repositories
        db_manager = DatabaseManager(temp_db_config)
        circuit_repo = CircuitRepository(db_manager)
        
        # Create circuit with metrics
        circuit_data = {
            'name': 'integration_test',
            'layers': [{'type': 'Linear', 'input_size': 10, 'output_size': 5}],
            'total_components': 50
        }
        
        circuit = CircuitModel("integration_test", circuit_data)
        circuit.set_verilog("module integration_test(); endmodule")
        
        metrics = CircuitMetrics(
            energy_per_op=1.0, latency=50, area=0.05, power=5.0,
            throughput=2000, accuracy=0.95, loss=0.2, crosstalk=-25
        )
        circuit.set_metrics(metrics)
        
        # Save circuit
        circuit_id = circuit_repo.save(circuit)
        assert circuit_id > 0
        
        # Load circuit
        loaded_circuit = circuit_repo.find_by_id(circuit_id)
        assert loaded_circuit is not None
        assert loaded_circuit.name == circuit.name
        assert loaded_circuit.verilog_code == circuit.verilog_code
        assert loaded_circuit.metrics.energy_per_op == metrics.energy_per_op
        
        # Verify database stats
        stats = db_manager.get_database_stats()
        assert stats['circuits_count'] == 1