#!/usr/bin/env python3
"""
Automated metrics collection script for Photonic Neural Network Foundry
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import requests
import subprocess
import sqlite3
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/metrics_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    tags: Dict[str, str] = None


class MetricsDatabase:
    """Database for storing metrics data."""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the metrics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    tags TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                ON metrics(name, timestamp)
            """)
    
    def store_metric(self, metric: MetricValue):
        """Store a metric value in the database."""
        tags_json = json.dumps(metric.tags) if metric.tags else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (name, value, unit, timestamp, source, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric.name,
                metric.value,
                metric.unit,
                metric.timestamp.isoformat(),
                metric.source,
                tags_json
            ))
    
    def get_metric_history(self, metric_name: str, days: int = 30) -> List[Dict]:
        """Get historical data for a metric."""
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name, value, unit, timestamp, source, tags
                FROM metrics
                WHERE name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (metric_name, since_date.isoformat()))
            
            return [
                {
                    "name": row[0],
                    "value": row[1],
                    "unit": row[2],
                    "timestamp": row[3],
                    "source": row[4],
                    "tags": json.loads(row[5]) if row[5] else {}
                }
                for row in cursor.fetchall()
            ]


class GitHubMetrics:
    """Collect metrics from GitHub API."""
    
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def get_repository_stats(self) -> List[MetricValue]:
        """Get repository statistics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Repository info
            repo_response = requests.get(
                f"{self.base_url}/repos/{self.repo}",
                headers=self.headers
            )
            repo_data = repo_response.json()
            
            metrics.extend([
                MetricValue("github_stars", repo_data.get("stargazers_count", 0), "count", timestamp, "github"),
                MetricValue("github_forks", repo_data.get("forks_count", 0), "count", timestamp, "github"),
                MetricValue("github_open_issues", repo_data.get("open_issues_count", 0), "count", timestamp, "github"),
                MetricValue("github_size", repo_data.get("size", 0), "kb", timestamp, "github"),
            ])
            
            # Pull requests
            pr_response = requests.get(
                f"{self.base_url}/repos/{self.repo}/pulls?state=all&per_page=100",
                headers=self.headers
            )
            pr_data = pr_response.json()
            
            open_prs = len([pr for pr in pr_data if pr["state"] == "open"])
            merged_prs = len([pr for pr in pr_data if pr["merged_at"]])
            
            metrics.extend([
                MetricValue("github_open_prs", open_prs, "count", timestamp, "github"),
                MetricValue("github_merged_prs", merged_prs, "count", timestamp, "github"),
            ])
            
            # Workflow runs (last 30 days)
            since_date = (datetime.now() - timedelta(days=30)).isoformat()
            runs_response = requests.get(
                f"{self.base_url}/repos/{self.repo}/actions/runs?created=>={since_date}",
                headers=self.headers
            )
            runs_data = runs_response.json()
            
            if "workflow_runs" in runs_data:
                successful_runs = len([r for r in runs_data["workflow_runs"] if r["conclusion"] == "success"])
                failed_runs = len([r for r in runs_data["workflow_runs"] if r["conclusion"] == "failure"])
                total_runs = len(runs_data["workflow_runs"])
                
                success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
                
                metrics.extend([
                    MetricValue("github_workflow_success_rate", success_rate, "percentage", timestamp, "github"),
                    MetricValue("github_workflow_runs", total_runs, "count", timestamp, "github"),
                ])
            
        except Exception as e:
            logger.error(f"Error collecting GitHub metrics: {e}")
        
        return metrics
    
    def get_code_frequency(self) -> List[MetricValue]:
        """Get code frequency statistics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            response = requests.get(
                f"{self.base_url}/repos/{self.repo}/stats/code_frequency",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    # Get last week's data
                    last_week = data[-1] if data else [0, 0, 0]
                    additions = last_week[1]
                    deletions = abs(last_week[2])
                    
                    metrics.extend([
                        MetricValue("github_weekly_additions", additions, "lines", timestamp, "github"),
                        MetricValue("github_weekly_deletions", deletions, "lines", timestamp, "github"),
                        MetricValue("github_weekly_net_changes", additions - deletions, "lines", timestamp, "github"),
                    ])
        
        except Exception as e:
            logger.error(f"Error collecting code frequency metrics: {e}")
        
        return metrics


class CodeQualityMetrics:
    """Collect code quality metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def get_test_coverage(self) -> List[MetricValue]:
        """Get test coverage metrics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Run pytest with coverage
            result = subprocess.run(
                ["pytest", "--cov=src/photonic_foundry", "--cov-report=json", "--cov-report=term"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Read coverage report
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data["totals"]["percent_covered"]
                
                metrics.append(
                    MetricValue("test_coverage", total_coverage, "percentage", timestamp, "pytest")
                )
                
                # Per-file coverage
                for filename, file_data in coverage_data["files"].items():
                    if "src/photonic_foundry" in filename:
                        file_coverage = file_data["summary"]["percent_covered"]
                        metrics.append(
                            MetricValue(
                                "file_coverage",
                                file_coverage,
                                "percentage",
                                timestamp,
                                "pytest",
                                {"file": filename}
                            )
                        )
        
        except Exception as e:
            logger.error(f"Error collecting test coverage metrics: {e}")
        
        return metrics
    
    def get_code_complexity(self) -> List[MetricValue]:
        """Get code complexity metrics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Run radon for complexity analysis
            result = subprocess.run(
                ["radon", "cc", "src/", "-j"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                
                total_complexity = 0
                function_count = 0
                
                for file_path, functions in complexity_data.items():
                    for func in functions:
                        if isinstance(func, dict) and "complexity" in func:
                            total_complexity += func["complexity"]
                            function_count += 1
                
                avg_complexity = total_complexity / function_count if function_count > 0 else 0
                
                metrics.append(
                    MetricValue("average_complexity", avg_complexity, "cyclomatic_complexity", timestamp, "radon")
                )
        
        except Exception as e:
            logger.error(f"Error collecting complexity metrics: {e}")
        
        return metrics
    
    def get_code_style_metrics(self) -> List[MetricValue]:
        """Get code style and quality metrics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Run pylint
            result = subprocess.run(
                ["pylint", "src/photonic_foundry/", "--output-format=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                pylint_data = json.loads(result.stdout)
                
                # Count issues by type
                error_count = len([msg for msg in pylint_data if msg["type"] == "error"])
                warning_count = len([msg for msg in pylint_data if msg["type"] == "warning"])
                convention_count = len([msg for msg in pylint_data if msg["type"] == "convention"])
                
                metrics.extend([
                    MetricValue("pylint_errors", error_count, "count", timestamp, "pylint"),
                    MetricValue("pylint_warnings", warning_count, "count", timestamp, "pylint"),
                    MetricValue("pylint_conventions", convention_count, "count", timestamp, "pylint"),
                ])
            
            # Calculate pylint score
            score_result = subprocess.run(
                ["pylint", "src/photonic_foundry/", "--score=y"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            # Extract score from output
            for line in score_result.stdout.split('\n'):
                if "Your code has been rated at" in line:
                    score_str = line.split("at")[1].split("/")[0].strip()
                    try:
                        score = float(score_str)
                        metrics.append(
                            MetricValue("pylint_score", score, "score", timestamp, "pylint")
                        )
                    except ValueError:
                        pass
        
        except Exception as e:
            logger.error(f"Error collecting code style metrics: {e}")
        
        return metrics


class SecurityMetrics:
    """Collect security-related metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def get_vulnerability_metrics(self) -> List[MetricValue]:
        """Get vulnerability scan results."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Run bandit security scan
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                
                high_severity = len([issue for issue in bandit_data["results"] if issue["issue_severity"] == "HIGH"])
                medium_severity = len([issue for issue in bandit_data["results"] if issue["issue_severity"] == "MEDIUM"])
                low_severity = len([issue for issue in bandit_data["results"] if issue["issue_severity"] == "LOW"])
                
                metrics.extend([
                    MetricValue("security_high_issues", high_severity, "count", timestamp, "bandit"),
                    MetricValue("security_medium_issues", medium_severity, "count", timestamp, "bandit"),
                    MetricValue("security_low_issues", low_severity, "count", timestamp, "bandit"),
                ])
            
            # Run safety check
            safety_result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if safety_result.stdout and safety_result.stdout.strip():
                safety_data = json.loads(safety_result.stdout)
                vulnerability_count = len(safety_data) if isinstance(safety_data, list) else 0
                
                metrics.append(
                    MetricValue("dependency_vulnerabilities", vulnerability_count, "count", timestamp, "safety")
                )
        
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
        
        return metrics


class PerformanceMetrics:
    """Collect performance metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def get_benchmark_metrics(self) -> List[MetricValue]:
        """Get performance benchmark results."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Run performance tests
            result = subprocess.run(
                ["pytest", "tests/performance/", "--benchmark-only", "--benchmark-json=benchmark.json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            benchmark_file = self.project_root / "benchmark.json"
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                
                for benchmark in benchmark_data["benchmarks"]:
                    name = benchmark["name"]
                    mean_time = benchmark["stats"]["mean"]
                    
                    metrics.append(
                        MetricValue(
                            f"benchmark_{name}",
                            mean_time,
                            "seconds",
                            timestamp,
                            "pytest-benchmark"
                        )
                    )
        
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics


class MetricsCollector:
    """Main metrics collection orchestrator."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.project_root = Path.cwd()
        self.db = MetricsDatabase()
        self.load_config()
    
    def load_config(self):
        """Load metrics configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = json.load(f)
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            self.config = {}
    
    def collect_all_metrics(self) -> List[MetricValue]:
        """Collect all configured metrics."""
        all_metrics = []
        
        # GitHub metrics
        github_token = os.getenv("GITHUB_TOKEN")
        github_repo = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/photonic-nn-foundry")
        
        if github_token:
            github_metrics = GitHubMetrics(github_token, github_repo)
            all_metrics.extend(github_metrics.get_repository_stats())
            all_metrics.extend(github_metrics.get_code_frequency())
        
        # Code quality metrics
        quality_metrics = CodeQualityMetrics(self.project_root)
        all_metrics.extend(quality_metrics.get_test_coverage())
        all_metrics.extend(quality_metrics.get_code_complexity())
        all_metrics.extend(quality_metrics.get_code_style_metrics())
        
        # Security metrics
        security_metrics = SecurityMetrics(self.project_root)
        all_metrics.extend(security_metrics.get_vulnerability_metrics())
        
        # Performance metrics
        performance_metrics = PerformanceMetrics(self.project_root)
        all_metrics.extend(performance_metrics.get_benchmark_metrics())
        
        return all_metrics
    
    def store_metrics(self, metrics: List[MetricValue]):
        """Store metrics in database."""
        for metric in metrics:
            self.db.store_metric(metric)
            logger.info(f"Stored metric: {metric.name} = {metric.value} {metric.unit}")
    
    def generate_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate metrics report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "metrics": {}
        }
        
        # Get recent metrics for key indicators
        key_metrics = [
            "test_coverage",
            "pylint_score",
            "security_high_issues",
            "github_workflow_success_rate"
        ]
        
        for metric_name in key_metrics:
            history = self.db.get_metric_history(metric_name, days)
            if history:
                latest = history[0]
                report["metrics"][metric_name] = {
                    "current_value": latest["value"],
                    "unit": latest["unit"],
                    "trend": self.calculate_trend(history),
                    "data_points": len(history)
                }
        
        return report
    
    def calculate_trend(self, history: List[Dict]) -> str:
        """Calculate trend direction from historical data."""
        if len(history) < 2:
            return "insufficient_data"
        
        recent_avg = sum(point["value"] for point in history[:3]) / min(3, len(history))
        older_avg = sum(point["value"] for point in history[-3:]) / min(3, len(history))
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"
    
    def run_collection(self):
        """Run the complete metrics collection process."""
        logger.info("Starting metrics collection...")
        
        try:
            metrics = self.collect_all_metrics()
            self.store_metrics(metrics)
            
            report = self.generate_report()
            
            # Save report
            report_path = Path("reports") / f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Metrics collection completed. Report saved to {report_path}")
            logger.info(f"Collected {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error during metrics collection: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    collector = MetricsCollector()
    collector.run_collection()


if __name__ == "__main__":
    main()