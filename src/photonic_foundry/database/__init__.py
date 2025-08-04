"""
Database layer for photonic circuit and component management.
"""

from .connection import DatabaseManager, get_database, close_database, DatabaseConfig
from .models import CircuitModel, ComponentModel, SimulationResult
from .repositories import CircuitRepository, ComponentRepository
from .cache import CircuitCache, ComponentCache, get_circuit_cache

__all__ = [
    'DatabaseManager',
    'get_database',
    'close_database',
    'DatabaseConfig',
    'CircuitModel',
    'ComponentModel', 
    'SimulationResult',
    'CircuitRepository',
    'ComponentRepository',
    'CircuitCache',
    'ComponentCache',
    'get_circuit_cache'
]