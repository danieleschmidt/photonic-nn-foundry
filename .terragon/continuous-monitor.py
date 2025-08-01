#!/usr/bin/env python3
"""
Continuous Value Monitoring
Runs scheduled value discovery and tracks long-term metrics.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

import sys
import importlib.util

# Import ValueDiscoveryEngine from the same directory
spec = importlib.util.spec_from_file_location("value_discovery", Path(__file__).parent / "value-discovery.py")
value_discovery_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(value_discovery_module)
ValueDiscoveryEngine = value_discovery_module.ValueDiscoveryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousMonitor:
    """Monitors repository value and triggers discovery cycles."""
    
    def __init__(self):
        """Initialize the continuous monitor."""
        self.metrics_path = Path(".terragon/value-metrics.json")
        self.trends_path = Path(".terragon/value-trends.json")
        self.discovery_engine = ValueDiscoveryEngine()
        
    def load_trends(self) -> Dict[str, Any]:
        """Load historical trend data."""
        if not self.trends_path.exists():
            return {
                "repository_info": {
                    "name": "photonic-nn-foundry",
                    "monitoring_started": datetime.now(timezone.utc).isoformat()
                },
                "historical_snapshots": [],
                "trend_metrics": {
                    "backlog_growth_rate": 0.0,
                    "average_resolution_time": 0.0,
                    "value_delivery_velocity": 0.0
                }
            }
        
        try:
            with open(self.trends_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trends: {e}")
            return {}
    
    def save_trends(self, trends: Dict[str, Any]) -> None:
        """Save updated trend data."""
        try:
            with open(self.trends_path, 'w') as f:
                json.dump(trends, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trends: {e}")
    
    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture current repository state snapshot."""
        # Run value discovery
        result = self.discovery_engine.run_discovery_cycle()
        
        # Load current metrics
        try:
            with open(self.metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}
        
        # Create snapshot
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_items": result.get("total_items_discovered", 0),
            "items_by_type": self._count_items_by_type(metrics.get("discovered_items", [])),
            "average_score": self._calculate_average_score(metrics.get("discovered_items", [])),
            "high_priority_count": self._count_high_priority(metrics.get("discovered_items", [])),
            "technical_debt_ratio": self._calculate_debt_ratio(metrics.get("discovered_items", [])),
            "next_best_score": result.get("next_best_value", {}).get("composite_score", 0)
        }
        
        return snapshot
    
    def _count_items_by_type(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count items by type."""
        counts = {}
        for item in items:
            item_type = item.get("type", "unknown")
            counts[item_type] = counts.get(item_type, 0) + 1
        return counts
    
    def _calculate_average_score(self, items: List[Dict[str, Any]]) -> float:
        """Calculate average composite score."""
        if not items:
            return 0.0
        scores = [item.get("composite_score", 0) for item in items]
        return round(sum(scores) / len(scores), 2)
    
    def _count_high_priority(self, items: List[Dict[str, Any]]) -> int:
        """Count high priority items."""
        return len([item for item in items if item.get("priority") == "high"])
    
    def _calculate_debt_ratio(self, items: List[Dict[str, Any]]) -> float:
        """Calculate technical debt ratio."""
        if not items:
            return 0.0
        debt_items = len([item for item in items if "debt" in item.get("type", "")])
        return round(debt_items / len(items), 3)
    
    def calculate_trends(self, snapshots: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend metrics from snapshots."""
        if len(snapshots) < 2:
            return {
                "backlog_growth_rate": 0.0,
                "score_improvement_rate": 0.0,
                "debt_reduction_rate": 0.0
            }
        
        # Get recent snapshots for trend calculation
        recent = snapshots[-5:]  # Last 5 snapshots
        
        # Calculate backlog growth rate
        initial_count = recent[0]["total_items"]
        final_count = recent[-1]["total_items"]
        days_diff = self._days_between_snapshots(recent[0], recent[-1])
        
        backlog_growth_rate = 0.0
        if days_diff > 0:
            backlog_growth_rate = (final_count - initial_count) / days_diff
        
        # Calculate average score trend
        initial_score = recent[0]["average_score"]
        final_score = recent[-1]["average_score"]
        score_improvement_rate = 0.0
        if days_diff > 0:
            score_improvement_rate = (final_score - initial_score) / days_diff
        
        # Calculate debt reduction rate
        initial_debt = recent[0]["technical_debt_ratio"]
        final_debt = recent[-1]["technical_debt_ratio"]
        debt_reduction_rate = 0.0
        if days_diff > 0:
            debt_reduction_rate = (initial_debt - final_debt) / days_diff
        
        return {
            "backlog_growth_rate": round(backlog_growth_rate, 3),
            "score_improvement_rate": round(score_improvement_rate, 3),
            "debt_reduction_rate": round(debt_reduction_rate, 3)
        }
    
    def _days_between_snapshots(self, snap1: Dict[str, Any], snap2: Dict[str, Any]) -> float:
        """Calculate days between two snapshots."""
        try:
            dt1 = datetime.fromisoformat(snap1["timestamp"].replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(snap2["timestamp"].replace('Z', '+00:00'))
            return (dt2 - dt1).total_seconds() / 86400  # Convert to days
        except Exception:
            return 1.0  # Default to 1 day if parsing fails
    
    def detect_anomalies(self, snapshot: Dict[str, Any], trends: Dict[str, Any]) -> List[str]:
        """Detect anomalous conditions requiring attention."""
        anomalies = []
        snapshots = trends.get("historical_snapshots", [])
        
        if not snapshots:
            return anomalies
        
        # Get baseline from recent history
        recent_snapshots = snapshots[-10:]  # Last 10 snapshots
        avg_items = sum(s["total_items"] for s in recent_snapshots) / len(recent_snapshots)
        avg_score = sum(s["average_score"] for s in recent_snapshots) / len(recent_snapshots)
        
        current_items = snapshot["total_items"]
        current_score = snapshot["average_score"]
        
        # Detect sudden backlog growth
        if current_items > avg_items * 1.5:
            anomalies.append(f"Sudden backlog growth: {current_items} items (avg: {avg_items:.1f})")
        
        # Detect score degradation
        if current_score < avg_score * 0.7:
            anomalies.append(f"Score degradation: {current_score} (avg: {avg_score:.1f})")
        
        # Detect high debt ratio
        if snapshot["technical_debt_ratio"] > 0.4:
            anomalies.append(f"High technical debt ratio: {snapshot['technical_debt_ratio']:.1%}")
        
        # Detect stagnant backlog (no high-value items)
        if snapshot["next_best_score"] < 15:
            anomalies.append("No high-value items detected - repository may be optimally maintained")
        
        return anomalies
    
    def generate_report(self, trends: Dict[str, Any]) -> str:
        """Generate monitoring report."""
        snapshots = trends.get("historical_snapshots", [])
        if not snapshots:
            return "No monitoring data available."
        
        latest = snapshots[-1]
        trend_metrics = trends.get("trend_metrics", {})
        
        report = f"""
# Continuous Value Monitoring Report

**Generated**: {datetime.now(timezone.utc).isoformat()}Z  
**Repository**: photonic-nn-foundry  
**Monitoring Duration**: {len(snapshots)} snapshots

## Current State
- **Total Items**: {latest['total_items']}
- **Average Score**: {latest['average_score']}
- **High Priority**: {latest['high_priority_count']}
- **Technical Debt Ratio**: {latest['technical_debt_ratio']:.1%}
- **Next Best Value Score**: {latest['next_best_score']}

## Trend Analysis
- **Backlog Growth Rate**: {trend_metrics.get('backlog_growth_rate', 0):.2f} items/day
- **Score Improvement Rate**: {trend_metrics.get('score_improvement_rate', 0):+.2f} points/day  
- **Debt Reduction Rate**: {trend_metrics.get('debt_reduction_rate', 0):+.3f}/day

## Item Distribution
"""
        
        for item_type, count in latest.get("items_by_type", {}).items():
            report += f"- **{item_type.replace('_', ' ').title()}**: {count} items\n"
        
        return report
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a complete monitoring cycle."""
        logger.info("Starting continuous monitoring cycle")
        
        # Load existing trends
        trends = self.load_trends()
        
        # Capture current snapshot
        snapshot = self.capture_snapshot()
        
        # Add to historical snapshots
        snapshots = trends.get("historical_snapshots", [])
        snapshots.append(snapshot)
        
        # Keep only last 100 snapshots
        snapshots = snapshots[-100:]
        trends["historical_snapshots"] = snapshots
        
        # Calculate trends
        trend_metrics = self.calculate_trends(snapshots)
        trends["trend_metrics"] = trend_metrics
        
        # Detect anomalies
        anomalies = self.detect_anomalies(snapshot, trends)
        
        # Save updated trends
        self.save_trends(trends)
        
        # Generate report
        report = self.generate_report(trends)
        
        result = {
            "status": "complete",
            "snapshot": snapshot,
            "trends": trend_metrics,
            "anomalies": anomalies,
            "report": report,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Monitoring cycle complete: {len(anomalies)} anomalies detected")
        return result


def main():
    """Main entry point for continuous monitoring."""
    monitor = ContinuousMonitor()
    result = monitor.run_monitoring_cycle()
    
    print(f"\n📊 CONTINUOUS MONITORING COMPLETE")
    print(f"📈 Status: {result['status']}")
    print(f"🔍 Total Items: {result['snapshot']['total_items']}")
    print(f"📊 Average Score: {result['snapshot']['average_score']}")
    
    if result['anomalies']:
        print(f"\n⚠️  ANOMALIES DETECTED:")
        for anomaly in result['anomalies']:
            print(f"   • {anomaly}")
    else:
        print(f"\n✅ No anomalies detected")
    
    print(f"\n📋 Full report saved to monitoring logs")


if __name__ == "__main__":
    main()