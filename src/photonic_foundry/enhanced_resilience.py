"""
Enhanced resilience and recovery system for photonic neural networks.
Implements comprehensive error handling, circuit self-healing, and predictive maintenance.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from threading import Lock, Event
from concurrent.futures import ThreadPoolExecutor
import psutil
import json

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur in photonic circuits."""
    COMPONENT_DEGRADATION = "component_degradation"
    THERMAL_DRIFT = "thermal_drift"
    OPTICAL_LOSS = "optical_loss"
    POWER_FLUCTUATION = "power_fluctuation"
    CONTROL_ERROR = "control_error"
    CALIBRATION_DRIFT = "calibration_drift"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RECALIBRATE = "recalibrate"
    THERMAL_COMPENSATION = "thermal_compensation"
    POWER_ADJUSTMENT = "power_adjustment"
    REDUNDANT_SWITCH = "redundant_switch"
    PARAMETER_RESET = "parameter_reset"
    CIRCUIT_BYPASS = "circuit_bypass"


@dataclass
class FailureEvent:
    """Represents a failure event in the system."""
    timestamp: float
    failure_type: FailureType
    component_id: str
    severity: float  # 0.0 to 1.0
    description: str
    affected_circuits: List[str] = field(default_factory=list)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    is_resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class HealthMetrics:
    """System health metrics."""
    overall_health: float  # 0.0 to 1.0
    component_health: Dict[str, float]
    thermal_status: Dict[str, float]
    power_status: Dict[str, float]
    optical_quality: Dict[str, float]
    prediction_confidence: float
    last_updated: float


class CircuitHealthMonitor:
    """Advanced health monitoring for photonic circuits."""
    
    def __init__(self, sampling_rate: float = 1.0, history_size: int = 1000):
        """
        Initialize health monitor.
        
        Args:
            sampling_rate: Health check frequency in Hz
            history_size: Number of historical readings to maintain
        """
        self.sampling_rate = sampling_rate
        self.history_size = history_size
        self.health_history = {}
        self.failure_history = []
        self.predictive_models = {}
        self._lock = Lock()
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self, circuit_ids: List[str]):
        """Start continuous health monitoring."""
        self._monitoring = True
        self._monitor_thread = ThreadPoolExecutor(max_workers=2)
        
        for circuit_id in circuit_ids:
            self.health_history[circuit_id] = []
            self._monitor_thread.submit(self._monitor_circuit, circuit_id)
            
        logger.info(f"Started health monitoring for {len(circuit_ids)} circuits")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.shutdown(wait=True)
        logger.info("Health monitoring stopped")
        
    def _monitor_circuit(self, circuit_id: str):
        """Monitor individual circuit health."""
        while self._monitoring:
            try:
                metrics = self._collect_health_metrics(circuit_id)
                
                with self._lock:
                    if circuit_id not in self.health_history:
                        self.health_history[circuit_id] = []
                    
                    self.health_history[circuit_id].append(metrics)
                    
                    # Maintain history size limit
                    if len(self.health_history[circuit_id]) > self.history_size:
                        self.health_history[circuit_id].pop(0)
                    
                # Check for anomalies
                self._detect_anomalies(circuit_id, metrics)
                
            except Exception as e:
                logger.error(f"Error monitoring circuit {circuit_id}: {e}")
                
            time.sleep(1.0 / self.sampling_rate)
            
    def _collect_health_metrics(self, circuit_id: str) -> HealthMetrics:
        """Collect comprehensive health metrics for a circuit."""
        current_time = time.time()
        
        # Simulate component health data (in real implementation, this would 
        # interface with actual hardware sensors)
        component_health = {
            f"mzi_{i}": 0.95 + np.random.normal(0, 0.02) for i in range(10)
        }
        component_health.update({
            f"detector_{i}": 0.98 + np.random.normal(0, 0.01) for i in range(5)
        })
        
        # Thermal monitoring
        thermal_status = {
            f"zone_{i}": 25.0 + np.random.normal(0, 1.0) for i in range(4)
        }
        
        # Power monitoring
        power_status = {
            "laser_power": 10.0 + np.random.normal(0, 0.5),
            "thermal_power": 2.5 + np.random.normal(0, 0.1),
            "control_power": 1.0 + np.random.normal(0, 0.05)
        }
        
        # Optical quality metrics
        optical_quality = {
            "insertion_loss": -0.5 + np.random.normal(0, 0.1),
            "crosstalk": -30.0 + np.random.normal(0, 2.0),
            "extinction_ratio": 25.0 + np.random.normal(0, 1.0)
        }
        
        # Calculate overall health score
        health_scores = list(component_health.values())
        thermal_scores = [1.0 - abs(t - 25.0) / 25.0 for t in thermal_status.values()]
        power_scores = [min(1.0, 15.0 / max(p, 0.1)) for p in power_status.values()]
        
        all_scores = health_scores + thermal_scores + power_scores
        overall_health = np.mean(all_scores)
        
        return HealthMetrics(
            overall_health=overall_health,
            component_health=component_health,
            thermal_status=thermal_status,
            power_status=power_status,
            optical_quality=optical_quality,
            prediction_confidence=0.85,  # Would be calculated from ML model
            last_updated=current_time
        )
        
    def _detect_anomalies(self, circuit_id: str, metrics: HealthMetrics):
        """Detect anomalies in health metrics."""
        anomalies = []
        
        # Component health anomalies
        for comp_id, health in metrics.component_health.items():
            if health < 0.8:
                anomalies.append(f"Component {comp_id} health degraded: {health:.3f}")
                self._create_failure_event(
                    FailureType.COMPONENT_DEGRADATION,
                    comp_id,
                    1.0 - health,
                    f"Component health below threshold: {health:.3f}"
                )
                
        # Thermal anomalies
        for zone, temp in metrics.thermal_status.items():
            if temp > 35.0:
                anomalies.append(f"High temperature in {zone}: {temp:.1f}°C")
                self._create_failure_event(
                    FailureType.THERMAL_DRIFT,
                    zone,
                    (temp - 25.0) / 25.0,
                    f"Temperature above normal: {temp:.1f}°C"
                )
                
        # Power anomalies
        if metrics.power_status["laser_power"] > 15.0:
            anomalies.append(f"High laser power: {metrics.power_status['laser_power']:.1f}mW")
            self._create_failure_event(
                FailureType.POWER_FLUCTUATION,
                "laser",
                (metrics.power_status["laser_power"] - 10.0) / 10.0,
                f"Laser power above normal: {metrics.power_status['laser_power']:.1f}mW"
            )
            
        if anomalies:
            logger.warning(f"Anomalies detected in {circuit_id}: {anomalies}")
            
    def _create_failure_event(self, failure_type: FailureType, component_id: str,
                             severity: float, description: str):
        """Create and log a failure event."""
        failure = FailureEvent(
            timestamp=time.time(),
            failure_type=failure_type,
            component_id=component_id,
            severity=min(severity, 1.0),
            description=description,
            recovery_actions=self._suggest_recovery_actions(failure_type)
        )
        
        with self._lock:
            self.failure_history.append(failure)
            
        logger.error(f"Failure detected: {failure}")
        
    def _suggest_recovery_actions(self, failure_type: FailureType) -> List[RecoveryAction]:
        """Suggest recovery actions based on failure type."""
        action_map = {
            FailureType.COMPONENT_DEGRADATION: [RecoveryAction.RECALIBRATE, RecoveryAction.REDUNDANT_SWITCH],
            FailureType.THERMAL_DRIFT: [RecoveryAction.THERMAL_COMPENSATION],
            FailureType.OPTICAL_LOSS: [RecoveryAction.POWER_ADJUSTMENT, RecoveryAction.RECALIBRATE],
            FailureType.POWER_FLUCTUATION: [RecoveryAction.POWER_ADJUSTMENT],
            FailureType.CONTROL_ERROR: [RecoveryAction.PARAMETER_RESET],
            FailureType.CALIBRATION_DRIFT: [RecoveryAction.RECALIBRATE]
        }
        
        return action_map.get(failure_type, [RecoveryAction.PARAMETER_RESET])
        
    def get_health_report(self, circuit_id: str) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        with self._lock:
            if circuit_id not in self.health_history or not self.health_history[circuit_id]:
                return {"error": f"No health data available for circuit {circuit_id}"}
            
            recent_metrics = self.health_history[circuit_id][-1]
            history = self.health_history[circuit_id]
            
        # Calculate trends
        if len(history) >= 2:
            health_trend = history[-1].overall_health - history[-2].overall_health
        else:
            health_trend = 0.0
            
        # Get recent failures
        recent_failures = [f for f in self.failure_history 
                          if time.time() - f.timestamp < 3600 and not f.is_resolved]
        
        return {
            "circuit_id": circuit_id,
            "current_health": recent_metrics.overall_health,
            "health_trend": health_trend,
            "component_count": len(recent_metrics.component_health),
            "degraded_components": [
                comp for comp, health in recent_metrics.component_health.items() 
                if health < 0.9
            ],
            "thermal_alerts": [
                zone for zone, temp in recent_metrics.thermal_status.items()
                if temp > 30.0
            ],
            "active_failures": len(recent_failures),
            "failure_details": [
                {
                    "type": f.failure_type.value,
                    "component": f.component_id,
                    "severity": f.severity,
                    "age_seconds": time.time() - f.timestamp
                }
                for f in recent_failures
            ],
            "recommended_actions": self._get_recommended_actions(recent_failures),
            "last_updated": recent_metrics.last_updated
        }
        
    def _get_recommended_actions(self, failures: List[FailureEvent]) -> List[str]:
        """Get recommended actions for current failures."""
        all_actions = []
        for failure in failures:
            all_actions.extend([action.value for action in failure.recovery_actions])
        
        # Remove duplicates and prioritize
        unique_actions = list(set(all_actions))
        priority_order = ["recalibrate", "thermal_compensation", "power_adjustment", 
                         "redundant_switch", "parameter_reset", "circuit_bypass"]
        
        return sorted(unique_actions, key=lambda x: priority_order.index(x) 
                     if x in priority_order else len(priority_order))


class SelfHealingSystem:
    """Self-healing system for photonic circuits."""
    
    def __init__(self, health_monitor: CircuitHealthMonitor):
        """
        Initialize self-healing system.
        
        Args:
            health_monitor: Health monitoring system
        """
        self.health_monitor = health_monitor
        self.recovery_strategies = {
            RecoveryAction.RECALIBRATE: self._recalibrate_component,
            RecoveryAction.THERMAL_COMPENSATION: self._thermal_compensation,
            RecoveryAction.POWER_ADJUSTMENT: self._power_adjustment,
            RecoveryAction.REDUNDANT_SWITCH: self._switch_to_redundant,
            RecoveryAction.PARAMETER_RESET: self._reset_parameters,
            RecoveryAction.CIRCUIT_BYPASS: self._bypass_circuit
        }
        self.recovery_history = []
        self._healing_active = False
        self._lock = Lock()
        
    def start_healing(self):
        """Start automatic healing process."""
        self._healing_active = True
        logger.info("Self-healing system activated")
        
    def stop_healing(self):
        """Stop automatic healing process."""
        self._healing_active = False
        logger.info("Self-healing system deactivated")
        
    async def heal_failures(self, circuit_id: str) -> Dict[str, Any]:
        """
        Automatically heal failures in a circuit.
        
        Args:
            circuit_id: Circuit to heal
            
        Returns:
            Healing report
        """
        if not self._healing_active:
            return {"error": "Self-healing system is not active"}
            
        with self._lock:
            active_failures = [f for f in self.health_monitor.failure_history 
                             if not f.is_resolved and circuit_id in f.affected_circuits]
        
        if not active_failures:
            return {"message": "No active failures found", "circuit_id": circuit_id}
            
        healing_results = []
        
        for failure in active_failures:
            for action in failure.recovery_actions:
                try:
                    result = await self._execute_recovery_action(action, failure)
                    healing_results.append({
                        "failure_id": f"{failure.component_id}_{failure.timestamp}",
                        "action": action.value,
                        "success": result["success"],
                        "details": result.get("details", "")
                    })
                    
                    if result["success"]:
                        failure.is_resolved = True
                        failure.resolution_time = time.time()
                        break  # If one action succeeds, move to next failure
                        
                except Exception as e:
                    logger.error(f"Recovery action {action.value} failed: {e}")
                    healing_results.append({
                        "failure_id": f"{failure.component_id}_{failure.timestamp}",
                        "action": action.value,
                        "success": False,
                        "details": str(e)
                    })
                    
        # Record healing session
        healing_session = {
            "timestamp": time.time(),
            "circuit_id": circuit_id,
            "failures_addressed": len(active_failures),
            "actions_taken": len(healing_results),
            "successful_recoveries": sum(1 for r in healing_results if r["success"]),
            "results": healing_results
        }
        
        self.recovery_history.append(healing_session)
        
        return healing_session
        
    async def _execute_recovery_action(self, action: RecoveryAction, 
                                     failure: FailureEvent) -> Dict[str, Any]:
        """Execute a specific recovery action."""
        strategy = self.recovery_strategies.get(action)
        if not strategy:
            return {"success": False, "details": f"Unknown recovery action: {action.value}"}
            
        return await strategy(failure)
        
    async def _recalibrate_component(self, failure: FailureEvent) -> Dict[str, Any]:
        """Recalibrate a component."""
        # Simulate recalibration process
        await asyncio.sleep(0.5)  # Simulate calibration time
        
        # Simulate successful calibration with 80% probability
        success = np.random.random() > 0.2
        
        return {
            "success": success,
            "details": f"Recalibrated component {failure.component_id}"
        }
        
    async def _thermal_compensation(self, failure: FailureEvent) -> Dict[str, Any]:
        """Apply thermal compensation."""
        await asyncio.sleep(0.2)
        
        # Thermal compensation is usually quite effective
        success = np.random.random() > 0.1
        
        return {
            "success": success,
            "details": f"Applied thermal compensation to {failure.component_id}"
        }
        
    async def _power_adjustment(self, failure: FailureEvent) -> Dict[str, Any]:
        """Adjust power levels."""
        await asyncio.sleep(0.1)
        
        success = np.random.random() > 0.15
        
        return {
            "success": success,
            "details": f"Adjusted power for {failure.component_id}"
        }
        
    async def _switch_to_redundant(self, failure: FailureEvent) -> Dict[str, Any]:
        """Switch to redundant component."""
        await asyncio.sleep(0.3)
        
        # Assume redundant switching is available 60% of the time
        has_redundancy = np.random.random() > 0.4
        success = has_redundancy and (np.random.random() > 0.05)
        
        return {
            "success": success,
            "details": f"Switched {failure.component_id} to redundant backup" if success
                      else "No redundant component available"
        }
        
    async def _reset_parameters(self, failure: FailureEvent) -> Dict[str, Any]:
        """Reset component parameters."""
        await asyncio.sleep(0.1)
        
        success = np.random.random() > 0.1
        
        return {
            "success": success,
            "details": f"Reset parameters for {failure.component_id}"
        }
        
    async def _bypass_circuit(self, failure: FailureEvent) -> Dict[str, Any]:
        """Bypass failed circuit section."""
        await asyncio.sleep(0.2)
        
        # Bypassing should usually work but may reduce performance
        success = np.random.random() > 0.05
        
        return {
            "success": success,
            "details": f"Bypassed failed section {failure.component_id} (performance may be reduced)"
        }
        
    def get_healing_stats(self) -> Dict[str, Any]:
        """Get self-healing system statistics."""
        if not self.recovery_history:
            return {"message": "No healing sessions recorded"}
            
        total_sessions = len(self.recovery_history)
        total_failures = sum(s["failures_addressed"] for s in self.recovery_history)
        total_recoveries = sum(s["successful_recoveries"] for s in self.recovery_history)
        
        success_rate = total_recoveries / max(total_failures, 1) * 100
        
        recent_sessions = [s for s in self.recovery_history 
                          if time.time() - s["timestamp"] < 3600]
        
        return {
            "total_healing_sessions": total_sessions,
            "total_failures_addressed": total_failures,
            "total_successful_recoveries": total_recoveries,
            "overall_success_rate_percent": success_rate,
            "recent_sessions_1h": len(recent_sessions),
            "healing_active": self._healing_active,
            "last_healing_session": self.recovery_history[-1]["timestamp"] if self.recovery_history else None
        }


class PredictiveMaintenance:
    """Predictive maintenance system for photonic circuits."""
    
    def __init__(self, health_monitor: CircuitHealthMonitor):
        """
        Initialize predictive maintenance system.
        
        Args:
            health_monitor: Health monitoring system
        """
        self.health_monitor = health_monitor
        self.prediction_models = {}
        self.maintenance_schedules = {}
        self.failure_predictions = {}
        
    def train_prediction_model(self, circuit_id: str) -> Dict[str, Any]:
        """Train failure prediction model for a circuit."""
        with self.health_monitor._lock:
            history = self.health_monitor.health_history.get(circuit_id, [])
            
        if len(history) < 10:
            return {"error": "Insufficient data for model training"}
            
        # Simple trend-based prediction (in production, use ML models)
        health_values = [m.overall_health for m in history[-100:]]
        
        # Calculate degradation trend
        if len(health_values) >= 2:
            trend = np.polyfit(range(len(health_values)), health_values, 1)[0]
        else:
            trend = 0.0
            
        # Estimate time to failure
        current_health = health_values[-1]
        failure_threshold = 0.7
        
        if trend < 0 and current_health > failure_threshold:
            time_to_failure = (current_health - failure_threshold) / abs(trend)
        else:
            time_to_failure = float('inf')
            
        model = {
            "circuit_id": circuit_id,
            "degradation_trend": trend,
            "current_health": current_health,
            "failure_threshold": failure_threshold,
            "predicted_time_to_failure": time_to_failure,
            "confidence": 0.75,  # Would be calculated from model validation
            "last_trained": time.time()
        }
        
        self.prediction_models[circuit_id] = model
        
        return model
        
    def predict_failures(self, circuit_id: str, horizon_hours: float = 24.0) -> Dict[str, Any]:
        """
        Predict failures within specified time horizon.
        
        Args:
            circuit_id: Circuit to analyze
            horizon_hours: Prediction horizon in hours
            
        Returns:
            Failure predictions
        """
        model = self.prediction_models.get(circuit_id)
        if not model:
            # Train model if not available
            model = self.train_prediction_model(circuit_id)
            if "error" in model:
                return model
                
        failure_risk = 0.0
        predicted_failures = []
        
        # Calculate failure risk based on trend and current health
        if model["degradation_trend"] < 0:
            degradation_rate = abs(model["degradation_trend"])
            health_in_horizon = model["current_health"] - (degradation_rate * horizon_hours)
            
            if health_in_horizon < model["failure_threshold"]:
                failure_risk = 1.0 - health_in_horizon / model["failure_threshold"]
                failure_risk = min(max(failure_risk, 0.0), 1.0)
                
                predicted_failures.append({
                    "component": "overall_circuit",
                    "predicted_failure_time": time.time() + (horizon_hours * 3600),
                    "failure_probability": failure_risk,
                    "recommended_action": "schedule_maintenance"
                })
                
        prediction = {
            "circuit_id": circuit_id,
            "prediction_horizon_hours": horizon_hours,
            "overall_failure_risk": failure_risk,
            "predicted_failures": predicted_failures,
            "confidence": model["confidence"],
            "recommendations": self._generate_maintenance_recommendations(failure_risk),
            "prediction_time": time.time()
        }
        
        self.failure_predictions[circuit_id] = prediction
        
        return prediction
        
    def _generate_maintenance_recommendations(self, failure_risk: float) -> List[str]:
        """Generate maintenance recommendations based on failure risk."""
        recommendations = []
        
        if failure_risk > 0.8:
            recommendations.extend([
                "Immediate maintenance required",
                "Consider redundant backup activation",
                "Schedule component replacement"
            ])
        elif failure_risk > 0.5:
            recommendations.extend([
                "Schedule preventive maintenance within 24 hours",
                "Increase monitoring frequency",
                "Prepare replacement components"
            ])
        elif failure_risk > 0.2:
            recommendations.extend([
                "Schedule routine maintenance within 1 week",
                "Monitor degradation trends closely"
            ])
        else:
            recommendations.append("No immediate maintenance required")
            
        return recommendations
        
    def schedule_maintenance(self, circuit_id: str, maintenance_type: str,
                           scheduled_time: float) -> Dict[str, Any]:
        """Schedule maintenance for a circuit."""
        maintenance_record = {
            "circuit_id": circuit_id,
            "maintenance_type": maintenance_type,
            "scheduled_time": scheduled_time,
            "status": "scheduled",
            "created_time": time.time()
        }
        
        if circuit_id not in self.maintenance_schedules:
            self.maintenance_schedules[circuit_id] = []
            
        self.maintenance_schedules[circuit_id].append(maintenance_record)
        
        logger.info(f"Scheduled {maintenance_type} maintenance for {circuit_id} "
                   f"at {time.ctime(scheduled_time)}")
        
        return maintenance_record
        
    def get_maintenance_status(self) -> Dict[str, Any]:
        """Get overall maintenance status."""
        total_circuits = len(self.maintenance_schedules)
        pending_maintenance = 0
        overdue_maintenance = 0
        current_time = time.time()
        
        for circuit_schedules in self.maintenance_schedules.values():
            for maintenance in circuit_schedules:
                if maintenance["status"] == "scheduled":
                    if maintenance["scheduled_time"] <= current_time:
                        overdue_maintenance += 1
                    else:
                        pending_maintenance += 1
                        
        return {
            "total_monitored_circuits": total_circuits,
            "pending_maintenance_tasks": pending_maintenance,
            "overdue_maintenance_tasks": overdue_maintenance,
            "prediction_models_trained": len(self.prediction_models),
            "active_failure_predictions": len(self.failure_predictions),
            "system_status": "normal" if overdue_maintenance == 0 else "attention_required"
        }