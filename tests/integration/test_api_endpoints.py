"""
Comprehensive integration tests for API endpoints.
"""

import base64
import json
import pytest
import torch
import torch.nn as nn
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.photonic_foundry.api.server import create_app
from src.photonic_foundry.core import PhotonicAccelerator


@pytest.fixture
def api_client():
    """Create test client for API."""
    app = create_app(config={'debug': True, 'enable_rate_limiting': False})
    return TestClient(app)


@pytest.fixture
def sample_model():
    """Create a simple PyTorch model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    return model


@pytest.fixture
def encoded_model(sample_model):
    """Create base64 encoded model data."""
    import io
    buffer = io.BytesIO()
    torch.save(sample_model, buffer)
    buffer.seek(0)
    encoded_data = base64.b64encode(buffer.read()).decode('utf-8')
    return encoded_data


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, api_client):
        """Test successful health check."""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "uptime_seconds" in data
        assert isinstance(data["database_connected"], bool)
        assert isinstance(data["cache_available"], bool)
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint."""
        response = api_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Photonic Neural Network Foundry API"
        assert "version" in data
        assert data["status"] == "running"


class TestAnalysisEndpoints:
    """Test model analysis endpoints."""
    
    def test_analyze_model_success(self, api_client, encoded_model):
        """Test successful model analysis."""
        request_data = {
            "model_data": encoded_model,
            "detailed": True
        }
        
        response = api_client.post("/api/v1/analysis/model", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["total_layers"] > 0
        assert data["supported_layers"] >= 0
        assert data["total_parameters"] > 0
        assert 0 <= data["compatibility_score"] <= 1
        assert data["complexity_score"] >= 0
        assert isinstance(data["layer_details"], list)
        assert isinstance(data["recommendations"], list)
        assert data["execution_time"] > 0
    
    def test_analyze_model_invalid_data(self, api_client):
        """Test model analysis with invalid data."""
        request_data = {
            "model_data": "invalid_base64_data",
            "detailed": False
        }
        
        response = api_client.post("/api/v1/analysis/model", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_analyze_model_empty_data(self, api_client):
        """Test model analysis with empty data."""
        request_data = {
            "model_data": "",
            "detailed": False
        }
        
        response = api_client.post("/api/v1/analysis/model", json=request_data)
        assert response.status_code == 422  # Validation error


class TestTranspilationEndpoints:
    """Test model transpilation endpoints."""
    
    def test_transpile_model_success(self, api_client, encoded_model):
        """Test successful model transpilation."""
        request_data = {
            "model_data": encoded_model,
            "target": "photonic_mac",
            "precision": 8,
            "optimize": True,
            "pdk": "skywater130",
            "wavelength": 1550.0
        }
        
        with patch('src.photonic_foundry.api.endpoints.torch2verilog') as mock_transpile:
            mock_transpile.return_value = "// Mock Verilog code"
            
            response = api_client.post("/api/v1/transpile/model", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["circuit"] is not None
            assert data["verilog_code"] is not None
            assert data["metrics"] is not None
            assert data["execution_time"] > 0
    
    def test_transpile_model_invalid_precision(self, api_client, encoded_model):
        """Test transpilation with invalid precision."""
        request_data = {
            "model_data": encoded_model,
            "precision": 0  # Invalid precision
        }
        
        response = api_client.post("/api/v1/transpile/model", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_transpile_model_invalid_wavelength(self, api_client, encoded_model):
        """Test transpilation with invalid wavelength."""
        request_data = {
            "model_data": encoded_model,
            "wavelength": 500.0  # Outside valid range
        }
        
        response = api_client.post("/api/v1/transpile/model", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_get_verilog_code_success(self, api_client):
        """Test getting Verilog code for existing circuit."""
        # Mock circuit loading
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_circuit = MagicMock()
            mock_circuit.generate_verilog.return_value = "// Test Verilog"
            mock_accelerator.load_circuit.return_value = mock_circuit
            mock_accelerator.circuit_repo.find_by_name.return_value = None
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.get("/api/v1/transpile/verilog/test_circuit")
            assert response.status_code == 200
            assert "// Test Verilog" in response.text
    
    def test_get_verilog_code_not_found(self, api_client):
        """Test getting Verilog code for non-existent circuit."""
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_accelerator.load_circuit.return_value = None
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.get("/api/v1/transpile/verilog/nonexistent")
            assert response.status_code == 404


class TestBenchmarkingEndpoints:
    """Test performance benchmarking endpoints."""
    
    def test_benchmark_performance_success(self, api_client, encoded_model):
        """Test successful performance benchmarking."""
        request_data = {
            "model_data": encoded_model,
            "input_shape": [1, 10],
            "iterations": 10,
            "pdk": "skywater130",
            "include_electronic_baseline": True,
            "precision": 8
        }
        
        response = api_client.post("/api/v1/benchmark/performance", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["photonic_implementation"] is not None
        assert data["iterations_completed"] == 10
        assert data["total_execution_time"] > 0
        
        # Check metrics structure
        photonic_metrics = data["photonic_implementation"]
        assert "inference_time_ms" in photonic_metrics
        assert "throughput_ops" in photonic_metrics
        assert "energy_per_inference_pj" in photonic_metrics
        assert "power_mw" in photonic_metrics
    
    def test_benchmark_invalid_input_shape(self, api_client, encoded_model):
        """Test benchmarking with invalid input shape."""
        request_data = {
            "model_data": encoded_model,
            "input_shape": [0, -1],  # Invalid dimensions
            "iterations": 10
        }
        
        response = api_client.post("/api/v1/benchmark/performance", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_benchmark_too_many_iterations(self, api_client, encoded_model):
        """Test benchmarking with too many iterations."""
        request_data = {
            "model_data": encoded_model,
            "input_shape": [1, 10],
            "iterations": 50000  # Exceeds limit
        }
        
        response = api_client.post("/api/v1/benchmark/performance", json=request_data)
        assert response.status_code == 422  # Validation error


class TestCircuitManagementEndpoints:
    """Test circuit management endpoints."""
    
    def test_list_circuits_success(self, api_client):
        """Test successful circuit listing."""
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_accelerator.list_saved_circuits.return_value = [
                {
                    'name': 'test_circuit_1',
                    'model_hash': 'hash123',
                    'layer_count': 3,
                    'component_count': 50,
                    'created_at': '2025-08-03T00:00:00Z',
                    'updated_at': '2025-08-03T00:00:00Z',
                    'has_verilog': True,
                    'has_metrics': True,
                    'energy_per_op': 0.5,
                    'latency': 100.0,
                    'area': 0.01
                }
            ]
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.get("/api/v1/circuits/list")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert len(data["circuits"]) == 1
            assert data["total_count"] == 1
            assert data["returned_count"] == 1
            assert data["has_more"] is False
    
    def test_list_circuits_with_filters(self, api_client):
        """Test circuit listing with filters."""
        params = {
            "limit": 10,
            "offset": 0,
            "name_pattern": "test",
            "has_verilog": True,
            "has_metrics": True
        }
        
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_accelerator.list_saved_circuits.return_value = []
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.get("/api/v1/circuits/list", params=params)
            assert response.status_code == 200
    
    def test_get_circuit_success(self, api_client):
        """Test getting specific circuit."""
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_circuit = MagicMock()
            mock_circuit.name = "test_circuit"
            mock_circuit.layers = [1, 2, 3]
            mock_circuit.connections = []
            mock_circuit.total_components = 50
            mock_accelerator.load_circuit.return_value = mock_circuit
            mock_accelerator.circuit_repo.find_by_name.return_value = {
                'created_at': '2025-08-03T00:00:00Z',
                'verilog_code': '// test',
                'metrics': {'energy_per_op': 0.5}
            }
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.get("/api/v1/circuits/test_circuit")
            assert response.status_code == 200
            
            data = response.json()
            assert data["name"] == "test_circuit"
            assert data["layers"] == 3
            assert data["has_verilog"] is True
            assert data["has_metrics"] is True
    
    def test_get_circuit_not_found(self, api_client):
        """Test getting non-existent circuit."""
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_accelerator.load_circuit.return_value = None
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.get("/api/v1/circuits/nonexistent")
            assert response.status_code == 404
    
    def test_delete_circuit_success(self, api_client):
        """Test successful circuit deletion."""
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_accelerator.circuit_repo.find_by_name.return_value = {'name': 'test'}
            mock_accelerator.circuit_repo.delete_by_name.return_value = True
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.delete("/api/v1/circuits/test_circuit")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "deleted successfully" in data["message"]
    
    def test_delete_circuit_not_found(self, api_client):
        """Test deleting non-existent circuit."""
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_accelerator.circuit_repo.find_by_name.return_value = None
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.delete("/api/v1/circuits/nonexistent")
            assert response.status_code == 404


class TestSystemEndpoints:
    """Test system information endpoints."""
    
    def test_get_system_stats_success(self, api_client):
        """Test successful system stats retrieval."""
        with patch('src.photonic_foundry.api.endpoints.get_accelerator') as mock_get_acc:
            mock_accelerator = MagicMock()
            mock_accelerator.get_database_stats.return_value = {
                'database': {'total_circuits': 10},
                'cache': {'hit_rate': 0.85},
                'circuit_stats': {'avg_components': 50}
            }
            mock_get_acc.return_value = mock_accelerator
            
            response = api_client.get("/api/v1/system/stats")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "database_stats" in data
            assert "cache_stats" in data
            assert "circuit_stats" in data


class TestOptimizationEndpoints:
    """Test optimization endpoints."""
    
    def test_optimize_mzi_mesh_success(self, api_client):
        """Test successful MZI mesh optimization."""
        weight_matrix = [[1.0, 0.5], [0.3, 1.2]]
        
        params = {
            "precision": 8,
            "loss_budget_db": 3.0
        }
        
        response = api_client.post(
            "/api/v1/optimize/mzi-mesh",
            json=weight_matrix,
            params=params
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "optimization_result" in data
        assert "timestamp" in data
    
    def test_optimize_ring_resonator_success(self, api_client):
        """Test successful ring resonator optimization."""
        weights = [1.0, 0.5, 0.3, 1.2]
        
        params = {
            "q_factor": 10000,
            "fsr_ghz": 100
        }
        
        response = api_client.post(
            "/api/v1/optimize/ring-resonator",
            json=weights,
            params=params
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "optimization_result" in data
        assert "timestamp" in data
    
    def test_optimize_mzi_mesh_invalid_precision(self, api_client):
        """Test MZI mesh optimization with invalid precision."""
        weight_matrix = [[1.0, 0.5], [0.3, 1.2]]
        
        params = {
            "precision": 0,  # Invalid
            "loss_budget_db": 3.0
        }
        
        response = api_client.post(
            "/api/v1/optimize/mzi-mesh",
            json=weight_matrix,
            params=params
        )
        
        assert response.status_code == 422  # Validation error


class TestAPIValidation:
    """Test API input validation."""
    
    def test_invalid_json_payload(self, api_client):
        """Test handling of invalid JSON payload."""
        response = api_client.post(
            "/api/v1/analysis/model",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, api_client):
        """Test handling of missing required fields."""
        response = api_client.post("/api/v1/analysis/model", json={})
        assert response.status_code == 422
    
    def test_invalid_enum_values(self, api_client, encoded_model):
        """Test handling of invalid enum values."""
        request_data = {
            "model_data": encoded_model,
            "target": "invalid_target",  # Invalid enum value
            "pdk": "invalid_pdk"  # Invalid enum value
        }
        
        response = api_client.post("/api/v1/transpile/model", json=request_data)
        assert response.status_code == 422


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_internal_server_error(self, api_client, encoded_model):
        """Test handling of internal server errors."""
        with patch('src.photonic_foundry.api.endpoints.analyze_model_compatibility') as mock_analyze:
            mock_analyze.side_effect = Exception("Internal error")
            
            request_data = {
                "model_data": encoded_model,
                "detailed": False
            }
            
            response = api_client.post("/api/v1/analysis/model", json=request_data)
            assert response.status_code == 200  # Endpoint handles error gracefully
            
            data = response.json()
            assert data["success"] is False
            assert "error_message" in data
    
    def test_timeout_handling(self, api_client):
        """Test handling of request timeouts."""
        # This would require actual timeout simulation
        # For now, just verify the endpoint exists
        response = api_client.get("/api/v1/system/stats")
        assert response.status_code in [200, 500]  # Either success or error


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    def test_response_time_headers(self, api_client):
        """Test that response time headers are included."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
    
    def test_concurrent_requests(self, api_client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return api_client.get("/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
    
    def test_large_payload_handling(self, api_client):
        """Test handling of large payloads."""
        # Create a larger model
        large_model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10)
        )
        
        import io
        buffer = io.BytesIO()
        torch.save(large_model, buffer)
        buffer.seek(0)
        encoded_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        request_data = {
            "model_data": encoded_data,
            "detailed": False
        }
        
        response = api_client.post("/api/v1/analysis/model", json=request_data)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 413, 422]