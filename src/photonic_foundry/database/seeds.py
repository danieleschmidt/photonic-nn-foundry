"""
Database seeding utilities for development and testing.
"""

import logging
import json
import time
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn

from . import get_database
from .models import CircuitModel, CircuitMetrics
from ..core import PhotonicAccelerator, MZILayer, PhotonicCircuit
from ..utils.advanced_optimizers import MZIMeshOptimizer, RingResonatorOptimizer

logger = logging.getLogger(__name__)


class DataSeeder:
    """Comprehensive data seeding for development and testing."""
    
    def __init__(self):
        self.db = get_database()
        self.accelerator = PhotonicAccelerator()
        
    def seed_development_data(self):
        """Seed database with comprehensive development data."""
        logger.info("Seeding development database...")
        
        # Seed sample neural network models
        self._seed_sample_models()
        
        # Seed photonic circuits
        self._seed_photonic_circuits()
        
        # Seed optimization results
        self._seed_optimization_results()
        
        # Seed benchmark data
        self._seed_benchmark_data()
        
        logger.info("Development data seeding complete")
        
    def seed_test_data(self):
        """Seed database with test data."""
        logger.info("Seeding test database...")
        
        # Seed minimal test circuits
        self._seed_test_circuits()
        
        # Seed test metrics
        self._seed_test_metrics()
        
        logger.info("Test data seeding complete")
        
    def _seed_sample_models(self):
        """Seed database with sample neural network models."""
        logger.info("Seeding sample neural network models...")
        
        # Simple MLP model
        mlp_model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Linear(128, 10)
        )
        
        # Convert to photonic circuit
        mlp_circuit = self.accelerator.convert_pytorch_model(mlp_model)
        mlp_circuit.name = "sample_mlp_mnist"
        
        # Analyze and save
        mlp_metrics = self.accelerator.compile_and_profile(mlp_circuit)
        mlp_verilog = mlp_circuit.generate_verilog()
        
        self.accelerator.save_circuit(mlp_circuit, mlp_verilog, mlp_metrics)
        
        # CNN-style model (simplified as linear layers)
        cnn_model = nn.Sequential(
            nn.Linear(1024, 512),  # Simulated conv layer
            nn.ReLU(),
            nn.Linear(512, 256),   # Simulated conv layer
            nn.ReLU(),
            nn.Linear(256, 64),    # Simulated fc layer
            nn.ReLU(),
            nn.Linear(64, 10)      # Output layer
        )
        
        cnn_circuit = self.accelerator.convert_pytorch_model(cnn_model)
        cnn_circuit.name = "sample_cnn_cifar10"
        
        cnn_metrics = self.accelerator.compile_and_profile(cnn_circuit)
        cnn_verilog = cnn_circuit.generate_verilog()
        
        self.accelerator.save_circuit(cnn_circuit, cnn_verilog, cnn_metrics)
        
        # Transformer-style model (attention simulation)
        transformer_model = nn.Sequential(
            nn.Linear(512, 2048),  # Feed-forward layer
            nn.ReLU(),
            nn.Linear(2048, 512),  # Feed-forward layer
            nn.ReLU(),
            nn.Linear(512, 256),   # Projection layer
            nn.ReLU(),
            nn.Linear(256, 128)    # Output layer
        )
        
        transformer_circuit = self.accelerator.convert_pytorch_model(transformer_model)
        transformer_circuit.name = "sample_transformer_attention"
        
        transformer_metrics = self.accelerator.compile_and_profile(transformer_circuit)
        transformer_verilog = transformer_circuit.generate_verilog()
        
        self.accelerator.save_circuit(transformer_circuit, transformer_verilog, transformer_metrics)
        
        logger.info("Sample neural network models seeded successfully")
        
    def _seed_photonic_circuits(self):
        """Seed database with various photonic circuit configurations."""
        logger.info("Seeding photonic circuit configurations...")
        
        # Small test circuit
        small_circuit = PhotonicCircuit("test_small_2x2")
        small_layer = MZILayer(2, 2, precision=4)
        small_circuit.add_layer(small_layer)
        
        small_metrics = small_circuit.analyze_circuit()
        small_verilog = small_circuit.generate_verilog()
        
        self.accelerator.save_circuit(small_circuit, small_verilog, small_metrics)
        
        # Medium complexity circuit
        medium_circuit = PhotonicCircuit("test_medium_16x8")
        medium_layer1 = MZILayer(16, 12, precision=6)
        medium_layer2 = MZILayer(12, 8, precision=6)
        medium_circuit.add_layer(medium_layer1)
        medium_circuit.add_layer(medium_layer2)
        medium_circuit.connect_layers(0, 1)
        
        medium_metrics = medium_circuit.analyze_circuit()
        medium_verilog = medium_circuit.generate_verilog()
        
        self.accelerator.save_circuit(medium_circuit, medium_verilog, medium_metrics)
        
        # Large scale circuit
        large_circuit = PhotonicCircuit("test_large_128x64")
        large_layer1 = MZILayer(128, 96, precision=8)
        large_layer2 = MZILayer(96, 64, precision=8)
        large_layer3 = MZILayer(64, 32, precision=8)
        
        large_circuit.add_layer(large_layer1)
        large_circuit.add_layer(large_layer2)
        large_circuit.add_layer(large_layer3)
        large_circuit.connect_layers(0, 1)
        large_circuit.connect_layers(1, 2)
        
        large_metrics = large_circuit.analyze_circuit()
        large_verilog = large_circuit.generate_verilog()
        
        self.accelerator.save_circuit(large_circuit, large_verilog, large_metrics)
        
        logger.info("Photonic circuit configurations seeded successfully")
        
    def _seed_optimization_results(self):
        """Seed database with optimization results."""
        logger.info("Seeding optimization results...")
        
        # Create sample weight matrices for optimization
        weight_matrices = [
            np.random.randn(4, 4),
            np.random.randn(8, 8),
            np.random.randn(16, 12),
            np.random.randn(32, 24)
        ]
        
        mzi_optimizer = MZIMeshOptimizer(precision=8)
        ring_optimizer = RingResonatorOptimizer()
        
        optimization_results = []
        
        for i, weights in enumerate(weight_matrices):
            # MZI mesh optimization
            mzi_result = mzi_optimizer.optimize_mesh_topology(weights)
            
            # Ring resonator optimization
            ring_result = ring_optimizer.design_resonator_bank(weights)
            
            optimization_data = {
                'circuit_id': f'optimization_test_{i}',
                'weight_matrix_shape': weights.shape,
                'mzi_optimization': mzi_result,
                'ring_optimization': ring_result,
                'timestamp': time.time()
            }
            
            optimization_results.append(optimization_data)
            
        # Store optimization results in database
        for result in optimization_results:
            circuit_data = {
                'name': result['circuit_id'],
                'optimization_results': result,
                'type': 'optimization_study'
            }
            
            circuit_model = CircuitModel(result['circuit_id'], circuit_data)
            self.accelerator.circuit_repo.save(circuit_model)
            
        logger.info(f"Seeded {len(optimization_results)} optimization results")
        
    def _seed_benchmark_data(self):
        """Seed database with benchmark comparison data."""
        logger.info("Seeding benchmark comparison data...")
        
        benchmark_configs = [
            {
                'name': 'mnist_comparison',
                'model_type': 'mlp',
                'dataset': 'mnist',
                'model_size': '784x256x128x10',
                'electronic_metrics': {
                    'latency_ms': 2.1,
                    'energy_mj': 150.0,
                    'throughput_ops': 476,
                    'accuracy': 0.967
                },
                'photonic_metrics': {
                    'latency_ms': 0.3,
                    'energy_mj': 3.2,
                    'throughput_ops': 3333,
                    'accuracy': 0.961
                }
            },
            {
                'name': 'cifar10_comparison',
                'model_type': 'cnn',
                'dataset': 'cifar10',
                'model_size': '1024x512x256x64x10',
                'electronic_metrics': {
                    'latency_ms': 8.5,
                    'energy_mj': 420.0,
                    'throughput_ops': 118,
                    'accuracy': 0.892
                },
                'photonic_metrics': {
                    'latency_ms': 1.2,
                    'energy_mj': 8.1,
                    'throughput_ops': 833,
                    'accuracy': 0.886
                }
            },
            {
                'name': 'transformer_comparison',
                'model_type': 'transformer',
                'dataset': 'synthetic',
                'model_size': '512x2048x512x256x128',
                'electronic_metrics': {
                    'latency_ms': 15.3,
                    'energy_mj': 850.0,
                    'throughput_ops': 65,
                    'accuracy': 0.934
                },
                'photonic_metrics': {
                    'latency_ms': 2.1,
                    'energy_mj': 17.6,
                    'throughput_ops': 476,
                    'accuracy': 0.928
                }
            }
        ]
        
        for benchmark in benchmark_configs:
            circuit_data = {
                'name': f"benchmark_{benchmark['name']}",
                'benchmark_data': benchmark,
                'type': 'benchmark_comparison'
            }
            
            circuit_model = CircuitModel(benchmark['name'], circuit_data)
            self.accelerator.circuit_repo.save(circuit_model)
            
        logger.info(f"Seeded {len(benchmark_configs)} benchmark comparisons")
        
    def _seed_test_circuits(self):
        """Seed minimal circuits for testing."""
        logger.info("Seeding test circuits...")
        
        # Minimal test circuit
        test_circuit = PhotonicCircuit("unit_test_circuit")
        test_layer = MZILayer(2, 2, precision=4)
        test_circuit.add_layer(test_layer)
        
        test_metrics = test_circuit.analyze_circuit()
        test_verilog = test_circuit.generate_verilog()
        
        self.accelerator.save_circuit(test_circuit, test_verilog, test_metrics)
        
        logger.info("Test circuits seeded successfully")
        
    def _seed_test_metrics(self):
        """Seed test performance metrics."""
        logger.info("Seeding test performance metrics...")
        
        test_metrics_data = [
            {
                'name': 'test_metrics_1',
                'energy_per_op': 0.5,
                'latency': 100.0,
                'area': 0.01,
                'power': 5.0,
                'throughput': 1000.0,
                'accuracy': 0.95
            },
            {
                'name': 'test_metrics_2',
                'energy_per_op': 1.2,
                'latency': 250.0,
                'area': 0.05,
                'power': 12.0,
                'throughput': 400.0,
                'accuracy': 0.98
            }
        ]
        
        for metrics_data in test_metrics_data:
            circuit_data = {
                'name': metrics_data['name'],
                'test_metrics': metrics_data,
                'type': 'test_data'
            }
            
            circuit_model = CircuitModel(metrics_data['name'], circuit_data)
            
            # Add metrics
            metrics = CircuitMetrics(
                energy_per_op=metrics_data['energy_per_op'],
                latency=metrics_data['latency'],
                area=metrics_data['area'],
                power=metrics_data['power'],
                throughput=metrics_data['throughput'],
                accuracy=metrics_data['accuracy'],
                loss=0.5,
                crosstalk=-30
            )
            circuit_model.set_metrics(metrics)
            
            self.accelerator.circuit_repo.save(circuit_model)
            
        logger.info(f"Seeded {len(test_metrics_data)} test metrics")


def seed_development_data(database=None):
    """Convenience function to seed development data."""
    seeder = DataSeeder()
    seeder.seed_development_data()


def seed_test_data(database=None):
    """Convenience function to seed test data."""
    seeder = DataSeeder()
    seeder.seed_test_data()


def clear_all_data(database=None):
    """Clear all data from database."""
    if database is None:
        database = get_database()
    
    logger.info("Clearing all database data...")
    database.clear_all_data()
    logger.info("Database cleared successfully")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "development":
            seed_development_data()
        elif sys.argv[1] == "test":
            seed_test_data()
        elif sys.argv[1] == "clear":
            clear_all_data()
        else:
            print("Usage: python seeds.py [development|test|clear]")
    else:
        print("Seeding development data by default...")
        seed_development_data()