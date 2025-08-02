#!/usr/bin/env python3
"""
Automated Repository Management Script for Photonic Neural Network Foundry

This script provides automated management capabilities for maintaining repository health,
compliance, and performance metrics collection.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AutomationTask:
    """Represents an automation task."""
    name: str
    description: str
    frequency: str  # daily, weekly, monthly
    command: str
    enabled: bool = True
    last_run: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


class RepositoryAutomation:
    """Main automation orchestrator for repository management."""
    
    def __init__(self, config_path: str = "automation_config.json"):
        self.config_path = Path(config_path)
        self.project_root = Path.cwd()
        self.tasks = {}
        self.load_configuration()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging for automation."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create specific log files for different automation areas
        for area in ['security', 'quality', 'deployment', 'metrics']:
            area_logger = logging.getLogger(f'automation.{area}')
            handler = logging.FileHandler(log_dir / f'{area}.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            area_logger.addHandler(handler)
    
    def load_configuration(self):
        """Load automation configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
        else:
            # Create default configuration
            config = self.create_default_config()
            self.save_configuration(config)
        
        # Convert config to tasks
        for task_name, task_config in config.get('tasks', {}).items():
            self.tasks[task_name] = AutomationTask(
                name=task_name,
                description=task_config['description'],
                frequency=task_config['frequency'],
                command=task_config['command'],
                enabled=task_config.get('enabled', True)
            )
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default automation configuration."""
        return {
            "version": "1.0.0",
            "project": "photonic-nn-foundry",
            "automation": {
                "enabled": True,
                "dry_run": False,
                "notification_channels": ["log", "console"]
            },
            "tasks": {
                "dependency_update": {
                    "description": "Update Python dependencies and check for security vulnerabilities",
                    "frequency": "weekly",
                    "command": "python scripts/dependency_health_check.py --update",
                    "enabled": True
                },
                "security_scan": {
                    "description": "Run comprehensive security scanning",
                    "frequency": "daily",
                    "command": "make security-scan",
                    "enabled": True
                },
                "code_quality_check": {
                    "description": "Run code quality analysis and generate reports",
                    "frequency": "daily",
                    "command": "make lint && make type-check",
                    "enabled": True
                },
                "test_execution": {
                    "description": "Execute full test suite with coverage",
                    "frequency": "daily",
                    "command": "make test-cov",
                    "enabled": True
                },
                "metrics_collection": {
                    "description": "Collect and store project metrics",
                    "frequency": "hourly",
                    "command": "python scripts/collect_metrics.py",
                    "enabled": True
                },
                "documentation_build": {
                    "description": "Build and validate documentation",
                    "frequency": "daily",
                    "command": "make docs-build",
                    "enabled": True
                },
                "container_security_scan": {
                    "description": "Scan container images for vulnerabilities",
                    "frequency": "daily",
                    "command": "docker run --rm -v $(pwd):/workspace aquasec/trivy fs /workspace",
                    "enabled": True
                },
                "backup_generation": {
                    "description": "Generate automated backups of critical data",
                    "frequency": "daily",
                    "command": "python scripts/backup_manager.py --create",
                    "enabled": True
                },
                "performance_benchmarks": {
                    "description": "Run performance benchmarks and track regressions",
                    "frequency": "weekly",
                    "command": "make benchmark",
                    "enabled": True
                },
                "license_compliance_check": {
                    "description": "Verify license compliance for all dependencies",
                    "frequency": "weekly",
                    "command": "python scripts/license_checker.py",
                    "enabled": True
                },
                "sbom_generation": {
                    "description": "Generate Software Bill of Materials",
                    "frequency": "weekly",
                    "command": "python scripts/generate_sbom.py",
                    "enabled": True
                },
                "health_check": {
                    "description": "Comprehensive system health check",
                    "frequency": "hourly",
                    "command": "./scripts/health-check.sh",
                    "enabled": True
                }
            }
        }
    
    def save_configuration(self, config: Dict[str, Any]):
        """Save automation configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def should_run_task(self, task: AutomationTask) -> bool:
        """Determine if a task should run based on its frequency and last run time."""
        if not task.enabled:
            return False
        
        if task.last_run is None:
            return True
        
        now = datetime.now()
        time_since_last_run = now - task.last_run
        
        if task.frequency == "hourly":
            return time_since_last_run >= timedelta(hours=1)
        elif task.frequency == "daily":
            return time_since_last_run >= timedelta(days=1)
        elif task.frequency == "weekly":
            return time_since_last_run >= timedelta(weeks=1)
        elif task.frequency == "monthly":
            return time_since_last_run >= timedelta(days=30)
        
        return False
    
    def execute_task(self, task: AutomationTask) -> bool:
        """Execute a single automation task."""
        logger.info(f"Executing task: {task.name}")
        logger.info(f"Command: {task.command}")
        
        try:
            start_time = time.time()
            
            # Execute command
            result = subprocess.run(
                task.command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                task.success_count += 1
                task.last_run = datetime.now()
                
                logger.info(f"Task {task.name} completed successfully in {execution_time:.2f}s")
                
                # Log output if verbose
                if result.stdout:
                    logger.debug(f"Task output: {result.stdout}")
                
                return True
            else:
                task.failure_count += 1
                logger.error(f"Task {task.name} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            task.failure_count += 1
            logger.error(f"Task {task.name} timed out after 30 minutes")
            return False
        except Exception as e:
            task.failure_count += 1
            logger.error(f"Task {task.name} failed with exception: {e}")
            return False
    
    def run_due_tasks(self):
        """Run all tasks that are due for execution."""
        logger.info("Starting automated task execution cycle")
        
        executed_tasks = 0
        successful_tasks = 0
        
        for task_name, task in self.tasks.items():
            if self.should_run_task(task):
                logger.info(f"Task {task_name} is due for execution")
                executed_tasks += 1
                
                if self.execute_task(task):
                    successful_tasks += 1
                
                # Small delay between tasks to avoid overwhelming the system
                time.sleep(5)
        
        logger.info(f"Automation cycle completed: {successful_tasks}/{executed_tasks} tasks successful")
        
        # Generate summary report
        self.generate_execution_report(executed_tasks, successful_tasks)
    
    def generate_execution_report(self, executed: int, successful: int):
        """Generate execution summary report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "executed_tasks": executed,
            "successful_tasks": successful,
            "success_rate": (successful / executed * 100) if executed > 0 else 0,
            "task_status": {}
        }
        
        for task_name, task in self.tasks.items():
            report["task_status"][task_name] = {
                "enabled": task.enabled,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "success_count": task.success_count,
                "failure_count": task.failure_count,
                "success_rate": (
                    task.success_count / (task.success_count + task.failure_count) * 100
                    if (task.success_count + task.failure_count) > 0 else 0
                )
            }
        
        # Save report
        reports_dir = Path("reports/automation")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Execution report saved to {report_file}")
    
    def run_specific_task(self, task_name: str):
        """Run a specific task by name."""
        if task_name not in self.tasks:
            logger.error(f"Task '{task_name}' not found")
            return False
        
        task = self.tasks[task_name]
        logger.info(f"Running specific task: {task_name}")
        
        return self.execute_task(task)
    
    def list_tasks(self):
        """List all available tasks."""
        print("\nAvailable Automation Tasks:")
        print("=" * 50)
        
        for task_name, task in self.tasks.items():
            status = "✅ Enabled" if task.enabled else "❌ Disabled"
            last_run = task.last_run.strftime("%Y-%m-%d %H:%M:%S") if task.last_run else "Never"
            
            print(f"\n{task_name}")
            print(f"  Description: {task.description}")
            print(f"  Frequency: {task.frequency}")
            print(f"  Status: {status}")
            print(f"  Last Run: {last_run}")
            print(f"  Success Rate: {task.success_count}/{task.success_count + task.failure_count}")
    
    def enable_task(self, task_name: str):
        """Enable a specific task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            logger.info(f"Task '{task_name}' enabled")
        else:
            logger.error(f"Task '{task_name}' not found")
    
    def disable_task(self, task_name: str):
        """Disable a specific task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            logger.info(f"Task '{task_name}' disabled")
        else:
            logger.error(f"Task '{task_name}' not found")
    
    def get_task_status(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of tasks."""
        if task_name:
            if task_name in self.tasks:
                task = self.tasks[task_name]
                return {
                    "name": task.name,
                    "enabled": task.enabled,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "success_count": task.success_count,
                    "failure_count": task.failure_count
                }
            else:
                return {"error": f"Task '{task_name}' not found"}
        else:
            return {
                task_name: {
                    "enabled": task.enabled,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "success_count": task.success_count,
                    "failure_count": task.failure_count
                }
                for task_name, task in self.tasks.items()
            }


def main():
    """Main entry point for the automation script."""
    parser = argparse.ArgumentParser(
        description="Automated Repository Management for Photonic Neural Network Foundry"
    )
    
    parser.add_argument(
        '--mode',
        choices=['run', 'list', 'status', 'task'],
        default='run',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--task',
        help='Specific task to run (when mode=task)'
    )
    
    parser.add_argument(
        '--enable',
        help='Enable specific task'
    )
    
    parser.add_argument(
        '--disable',
        help='Disable specific task'
    )
    
    parser.add_argument(
        '--config',
        default='automation_config.json',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize automation
    automation = RepositoryAutomation(args.config)
    
    try:
        if args.mode == 'run':
            automation.run_due_tasks()
        elif args.mode == 'list':
            automation.list_tasks()
        elif args.mode == 'status':
            status = automation.get_task_status()
            print(json.dumps(status, indent=2))
        elif args.mode == 'task':
            if not args.task:
                logger.error("Task name required when mode=task")
                sys.exit(1)
            success = automation.run_specific_task(args.task)
            sys.exit(0 if success else 1)
        
        if args.enable:
            automation.enable_task(args.enable)
        
        if args.disable:
            automation.disable_task(args.disable)
    
    except KeyboardInterrupt:
        logger.info("Automation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Automation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()