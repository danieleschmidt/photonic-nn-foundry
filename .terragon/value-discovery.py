#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers and prioritizes the highest value work.
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValueDiscoveryEngine:
    """Autonomous value discovery and prioritization system."""
    
    def __init__(self, config_path: str = ".terragon/config.json"):
        """Initialize the value discovery engine."""
        self.config_path = Path(config_path)
        self.metrics_path = Path(".terragon/value-metrics.json")
        self.config = self._load_config()
        self.discovered_items = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "scoring": {
                "repository_maturity": "maturing",
                "weights": {
                    "maturing": {
                        "wsjf": 0.6,
                        "ice": 0.1,
                        "technicalDebt": 0.2,
                        "security": 0.1
                    }
                },
                "thresholds": {
                    "minScore": 15,
                    "maxRisk": 0.7,
                    "securityBoost": 2.0
                }
            }
        }
    
    def discover_value_items(self) -> List[Dict[str, Any]]:
        """Discover all potential value items from multiple sources."""
        logger.info("Starting value discovery scan...")
        
        items = []
        items.extend(self._discover_from_code_comments())
        items.extend(self._discover_from_test_gaps())
        items.extend(self._discover_from_static_analysis())
        items.extend(self._discover_from_security_scan())
        items.extend(self._discover_from_performance_gaps())
        
        return items
    
    def _discover_from_code_comments(self) -> List[Dict[str, Any]]:
        """Discover TODO/FIXME items from code comments."""
        items = []
        
        try:
            result = subprocess.run([
                'grep', '-r', '-n', '-E', 
                'TODO|FIXME|HACK|XXX|DEPRECATED',
                'src/', 'tests/'
            ], capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    match = re.match(r'([^:]+):(\d+):(.+)', line)
                    if match:
                        file_path, line_num, comment = match.groups()
                        items.append({
                            "title": f"Address technical debt: {comment.strip()[:50]}...",
                            "type": "technical_debt",
                            "source": "code_comment",
                            "file": file_path,
                            "line": int(line_num),
                            "priority": "medium",
                            "estimatedEffort": 3,
                            "impact": "Reduces technical debt"
                        })
        except Exception as e:
            logger.warning(f"Error discovering code comments: {e}")
        
        return items
    
    def _discover_from_test_gaps(self) -> List[Dict[str, Any]]:
        """Discover missing test implementations."""
        items = []
        
        try:
            # Find test files with skeleton implementations
            test_files = Path('tests').rglob('test_*.py')
            
            for test_file in test_files:
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # Find methods with just 'pass' statements
                pass_methods = re.findall(r'def (test_\w+).*?:\s*.*?pass', content, re.DOTALL)
                
                for method in pass_methods:
                    items.append({
                        "title": f"Implement test method: {method}",
                        "type": "test_debt", 
                        "source": "static_analysis",
                        "file": str(test_file),
                        "priority": "high",
                        "estimatedEffort": 2,
                        "impact": "Improves test coverage and reliability"
                    })
        except Exception as e:
            logger.warning(f"Error discovering test gaps: {e}")
        
        return items
    
    def _discover_from_static_analysis(self) -> List[Dict[str, Any]]:
        """Discover issues from static analysis tools."""
        items = []
        
        # Check for missing CI/CD
        if not Path('.github/workflows').exists():
            items.append({
                "title": "Set up GitHub Actions CI/CD workflows",
                "type": "automation_gap",
                "source": "static_analysis",
                "priority": "high",
                "estimatedEffort": 3,
                "impact": "Enables continuous integration and deployment"
            })
        
        return items
    
    def _discover_from_security_scan(self) -> List[Dict[str, Any]]:
        """Discover security vulnerabilities."""
        items = []
        
        try:
            # Check for outdated dependencies
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                for pkg in outdated[:5]:  # Limit to top 5
                    items.append({
                        "title": f"Update dependency: {pkg['name']}",
                        "type": "dependency_update",
                        "source": "security_scan",
                        "priority": "low",
                        "estimatedEffort": 1,
                        "impact": "Improves security and compatibility"
                    })
        except Exception as e:
            logger.warning(f"Error in security scan: {e}")
        
        return items
    
    def _discover_from_performance_gaps(self) -> List[Dict[str, Any]]:
        """Discover performance improvement opportunities."""
        items = []
        
        # Check for missing performance monitoring
        if not any(Path('.').glob('**/prometheus*')) and not any(Path('.').glob('**/metrics*')):
            items.append({
                "title": "Add performance monitoring and metrics",
                "type": "performance_gap",
                "source": "performance_analysis", 
                "priority": "medium",
                "estimatedEffort": 6,
                "impact": "Enables performance tracking and optimization"
            })
        
        return items
    
    def calculate_scores(self, item: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive scores for a work item."""
        # WSJF calculation
        user_value = self._score_user_impact(item)
        time_criticality = self._score_urgency(item)
        risk_reduction = self._score_risk_mitigation(item)
        opportunity = self._score_opportunity(item)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity
        job_size = item.get('estimatedEffort', 3)
        wsjf = cost_of_delay / max(job_size, 0.5)
        
        # ICE calculation
        impact = self._score_business_impact(item)
        confidence = self._score_execution_confidence(item)
        ease = 10 - min(job_size, 10)
        ice = impact * confidence * ease
        
        # Technical debt calculation
        debt_impact = self._calculate_debt_cost(item)
        
        return {
            "wsjf": round(wsjf, 1),
            "ice": round(ice, 1),
            "technicalDebt": round(debt_impact, 1)
        }
    
    def _score_user_impact(self, item: Dict[str, Any]) -> float:
        """Score user/business impact (1-10)."""
        type_scores = {
            "security_vulnerability": 10,
            "test_debt": 8,
            "technical_debt": 6,
            "automation_gap": 7,
            "performance_gap": 5,
            "dependency_update": 3
        }
        return type_scores.get(item.get('type', ''), 5)
    
    def _score_urgency(self, item: Dict[str, Any]) -> float:
        """Score time criticality (1-10)."""
        priority_scores = {
            "high": 8,
            "medium": 5,
            "low": 2
        }
        return priority_scores.get(item.get('priority', 'medium'), 5)
    
    def _score_risk_mitigation(self, item: Dict[str, Any]) -> float:
        """Score risk reduction value (1-10)."""
        if 'security' in item.get('type', '').lower():
            return 9
        elif 'test' in item.get('type', '').lower():
            return 7
        return 4
    
    def _score_opportunity(self, item: Dict[str, Any]) -> float:
        """Score opportunity enablement (1-10)."""
        if 'automation' in item.get('type', '').lower():
            return 8
        return 3
    
    def _score_business_impact(self, item: Dict[str, Any]) -> float:
        """Score business impact for ICE (1-10)."""
        return self._score_user_impact(item)
    
    def _score_execution_confidence(self, item: Dict[str, Any]) -> float:
        """Score execution confidence for ICE (1-10)."""
        effort = item.get('estimatedEffort', 3)
        if effort <= 2:
            return 9
        elif effort <= 5:
            return 7
        else:
            return 5
    
    def _calculate_debt_cost(self, item: Dict[str, Any]) -> float:
        """Calculate technical debt cost."""
        if 'debt' in item.get('type', '').lower():
            return item.get('estimatedEffort', 3) * 10
        return item.get('estimatedEffort', 3) * 5
    
    def calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate composite score using adaptive weights."""
        maturity = self.config["scoring"]["repository_maturity"]
        weights = self.config["scoring"]["weights"][maturity]
        
        composite = (
            weights["wsjf"] * self._normalize_score(scores["wsjf"], 50) +
            weights["ice"] * self._normalize_score(scores["ice"], 500) +
            weights["technicalDebt"] * self._normalize_score(scores["technicalDebt"], 100)
        )
        
        return round(composite * 100, 1)
    
    def _normalize_score(self, score: float, max_val: float) -> float:
        """Normalize score to 0-1 range."""
        return min(score / max_val, 1.0)
    
    def select_next_best_value(self, scored_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the next highest value item to work on."""
        if not scored_items:
            return None
        
        # Sort by composite score descending
        sorted_items = sorted(scored_items, key=lambda x: x["composite_score"], reverse=True)
        
        # Apply filters
        min_score = self.config["scoring"]["thresholds"]["minScore"]
        
        for item in sorted_items:
            if item["composite_score"] >= min_score:
                return item
        
        return None
    
    def save_metrics(self, items: List[Dict[str, Any]]) -> None:
        """Save discovered items and metrics to JSON file."""
        metrics = {
            "repository_info": {
                "name": "photonic-nn-foundry", 
                "maturity_level": self.config["scoring"]["repository_maturity"],
                "assessment_date": datetime.now(timezone.utc).isoformat()
            },
            "discovered_items": items,
            "backlog_metrics": {
                "totalItems": len(items),
                "averageScore": sum(item.get("composite_score", 0) for item in items) / max(len(items), 1),
                "highPriorityItems": len([item for item in items if item.get("priority") == "high"])
            }
        }
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved {len(items)} discovered items to {self.metrics_path}")
    
    def run_discovery_cycle(self) -> Dict[str, Any]:
        """Run a complete discovery and prioritization cycle."""
        logger.info("Starting autonomous value discovery cycle")
        
        # Discover all potential work items
        raw_items = self.discover_value_items()
        
        # Score and prioritize items
        scored_items = []
        for item in raw_items:
            scores = self.calculate_scores(item)
            composite_score = self.calculate_composite_score(scores)
            
            item.update({
                "scores": scores,
                "composite_score": composite_score,
                "discovered_at": datetime.now(timezone.utc).isoformat()
            })
            scored_items.append(item)
        
        # Save metrics
        self.save_metrics(scored_items)
        
        # Select next best value item
        next_item = self.select_next_best_value(scored_items)
        
        return {
            "total_items_discovered": len(scored_items),
            "next_best_value": next_item,
            "discovery_timestamp": datetime.now(timezone.utc).isoformat()
        }


def main():
    """Main entry point for value discovery."""
    engine = ValueDiscoveryEngine()
    result = engine.run_discovery_cycle()
    
    print(f"\n🎯 AUTONOMOUS VALUE DISCOVERY COMPLETE")
    print(f"📊 Total items discovered: {result['total_items_discovered']}")
    
    if result['next_best_value']:
        item = result['next_best_value']
        print(f"\n🏆 NEXT BEST VALUE ITEM:")
        print(f"   Title: {item['title']}")
        print(f"   Score: {item['composite_score']}")
        print(f"   Type: {item['type']}")
        print(f"   Effort: {item['estimatedEffort']} hours")
        print(f"   Impact: {item['impact']}")
    else:
        print("\n✅ No high-value items discovered - repository is optimally maintained!")


if __name__ == "__main__":
    main()