#!/usr/bin/env python3
"""
Repository automation script for Photonic Neural Network Foundry
Handles automated maintenance, cleanup, and optimization tasks
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GitHubAutomation:
    """Automate GitHub repository management tasks."""
    
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def update_repository_settings(self) -> bool:
        """Update repository settings for optimal configuration."""
        try:
            settings = {
                "name": "photonic-nn-foundry",
                "description": "Turn the latest silicon-photonic AI accelerators into a reproducible software stack",
                "homepage": "https://danieleschmidt.github.io/photonic-nn-foundry",
                "topics": [
                    "photonic-computing",
                    "neural-networks",
                    "pytorch",
                    "verilog",
                    "silicon-photonics",
                    "ai-accelerator",
                    "machine-learning",
                    "quantum-computing",
                    "optical-computing"
                ],
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_downloads": True,
                "has_discussions": True,
                "allow_squash_merge": True,
                "allow_merge_commit": False,
                "allow_rebase_merge": True,
                "allow_auto_merge": True,
                "delete_branch_on_merge": True,
                "security_and_analysis": {
                    "secret_scanning": {"status": "enabled"},
                    "secret_scanning_push_protection": {"status": "enabled"},
                    "dependency_graph": {"status": "enabled"}
                }
            }
            
            response = requests.patch(
                f"{self.base_url}/repos/{self.repo}",
                headers=self.headers,
                json=settings
            )
            
            if response.status_code == 200:
                logger.info("Repository settings updated successfully")
                return True
            else:
                logger.error(f"Failed to update repository settings: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating repository settings: {e}")
            return False
    
    def setup_branch_protection(self) -> bool:
        """Set up branch protection rules for main branch."""
        try:
            protection_rules = {
                "required_status_checks": {
                    "strict": True,
                    "checks": [
                        {"context": "Code Quality & Security (lint)"},
                        {"context": "Code Quality & Security (security)"},
                        {"context": "Code Quality & Security (type-check)"},
                        {"context": "Test Suite (ubuntu-latest, 3.11)"},
                        {"context": "Container Security Scan"}
                    ]
                },
                "enforce_admins": False,
                "required_pull_request_reviews": {
                    "required_approving_review_count": 1,
                    "dismiss_stale_reviews": True,
                    "require_code_owner_reviews": True,
                    "require_last_push_approval": True
                },
                "restrictions": None,
                "required_linear_history": False,
                "allow_force_pushes": False,
                "allow_deletions": False,
                "required_conversation_resolution": True
            }
            
            response = requests.put(
                f"{self.base_url}/repos/{self.repo}/branches/main/protection",
                headers=self.headers,
                json=protection_rules
            )
            
            if response.status_code in [200, 201]:
                logger.info("Branch protection rules updated successfully")
                return True
            else:
                logger.error(f"Failed to set branch protection: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up branch protection: {e}")
            return False
    
    def manage_labels(self) -> bool:
        """Create and manage repository labels."""
        try:
            labels = [
                {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
                {"name": "documentation", "color": "0075ca", "description": "Improvements or additions to documentation"},
                {"name": "duplicate", "color": "cfd3d7", "description": "This issue or pull request already exists"},
                {"name": "enhancement", "color": "a2eeef", "description": "New feature or request"},
                {"name": "good first issue", "color": "7057ff", "description": "Good for newcomers"},
                {"name": "help wanted", "color": "008672", "description": "Extra attention is needed"},
                {"name": "invalid", "color": "e4e669", "description": "This doesn't seem right"},
                {"name": "question", "color": "d876e3", "description": "Further information is requested"},
                {"name": "wontfix", "color": "ffffff", "description": "This will not be worked on"},
                
                # Project-specific labels
                {"name": "photonic-simulation", "color": "ff6b6b", "description": "Related to photonic simulation"},
                {"name": "transpiler", "color": "4ecdc4", "description": "PyTorch to Verilog transpiler"},
                {"name": "performance", "color": "45b7d1", "description": "Performance optimization"},
                {"name": "security", "color": "f39c12", "description": "Security-related issue"},
                {"name": "testing", "color": "9b59b6", "description": "Testing infrastructure"},
                {"name": "ci/cd", "color": "1abc9c", "description": "Continuous integration/deployment"},
                {"name": "dependencies", "color": "e67e22", "description": "Dependency updates"},
                {"name": "automation", "color": "34495e", "description": "Automation and tooling"},
                {"name": "breaking-change", "color": "e74c3c", "description": "Breaking change"},
                {"name": "optimization", "color": "2ecc71", "description": "Code or performance optimization"}
            ]
            
            for label in labels:
                response = requests.post(
                    f"{self.base_url}/repos/{self.repo}/labels",
                    headers=self.headers,
                    json=label
                )
                
                if response.status_code == 201:
                    logger.info(f"Created label: {label['name']}")
                elif response.status_code == 422:
                    # Label already exists, update it
                    requests.patch(
                        f"{self.base_url}/repos/{self.repo}/labels/{label['name']}",
                        headers=self.headers,
                        json=label
                    )
                    logger.info(f"Updated label: {label['name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error managing labels: {e}")
            return False
    
    def cleanup_old_workflows(self) -> bool:
        """Clean up old workflow runs to save storage."""
        try:
            # Get workflow runs older than 90 days
            cutoff_date = datetime.now() - timedelta(days=90)
            
            response = requests.get(
                f"{self.base_url}/repos/{self.repo}/actions/runs?per_page=100",
                headers=self.headers
            )
            
            if response.status_code == 200:
                runs = response.json()["workflow_runs"]
                deleted_count = 0
                
                for run in runs:
                    run_date = datetime.fromisoformat(run["created_at"].replace('Z', '+00:00'))
                    if run_date < cutoff_date and run["status"] == "completed":
                        delete_response = requests.delete(
                            f"{self.base_url}/repos/{self.repo}/actions/runs/{run['id']}",
                            headers=self.headers
                        )
                        if delete_response.status_code == 204:
                            deleted_count += 1
                
                logger.info(f"Deleted {deleted_count} old workflow runs")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cleaning up workflow runs: {e}")
            return False


class CodeMaintenance:
    """Automated code maintenance tasks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def update_copyright_years(self) -> bool:
        """Update copyright years in source files."""
        try:
            current_year = datetime.now().year
            files_updated = 0
            
            for file_path in self.project_root.rglob("*.py"):
                if "venv" in str(file_path) or "__pycache__" in str(file_path):
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update copyright patterns
                import re
                patterns = [
                    r'Copyright \(c\) (\d{4})',
                    r'Copyright (\d{4})',
                    r'© (\d{4})'
                ]
                
                updated_content = content
                for pattern in patterns:
                    updated_content = re.sub(
                        pattern,
                        lambda m: m.group(0).replace(m.group(1), str(current_year)),
                        updated_content
                    )
                
                if updated_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    files_updated += 1
            
            logger.info(f"Updated copyright years in {files_updated} files")
            return True
            
        except Exception as e:
            logger.error(f"Error updating copyright years: {e}")
            return False
    
    def optimize_imports(self) -> bool:
        """Optimize Python imports using isort."""
        try:
            result = subprocess.run(
                ["isort", "src/", "tests/", "--profile", "black"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Optimized Python imports")
                return True
            else:
                logger.error(f"Failed to optimize imports: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error optimizing imports: {e}")
            return False
    
    def remove_unused_dependencies(self) -> bool:
        """Identify and suggest removal of unused dependencies."""
        try:
            # Use pip-autoremove to find unused dependencies
            result = subprocess.run(
                ["pip-autoremove", "--list"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                unused_deps = result.stdout.strip().split('\n')
                logger.info(f"Found unused dependencies: {unused_deps}")
                
                # Create a report file
                report_path = self.project_root / "reports" / "unused_dependencies.txt"
                report_path.parent.mkdir(exist_ok=True)
                
                with open(report_path, 'w') as f:
                    f.write("Unused Dependencies Report\n")
                    f.write("=" * 30 + "\n\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                    f.write("Potentially unused dependencies:\n")
                    for dep in unused_deps:
                        f.write(f"- {dep}\n")
                
                return True
            else:
                logger.info("No unused dependencies found")
                return True
                
        except Exception as e:
            logger.error(f"Error checking unused dependencies: {e}")
            return False
    
    def update_docstrings(self) -> bool:
        """Check and suggest docstring improvements."""
        try:
            result = subprocess.run(
                ["pydocstyle", "src/photonic_foundry/", "--explain"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                report_path = self.project_root / "reports" / "docstring_issues.txt"
                report_path.parent.mkdir(exist_ok=True)
                
                with open(report_path, 'w') as f:
                    f.write("Docstring Issues Report\n")
                    f.write("=" * 25 + "\n\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                    f.write(result.stdout)
                
                logger.info(f"Docstring issues report saved to {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking docstrings: {e}")
            return False


class DependencyManagement:
    """Automated dependency management."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def check_outdated_dependencies(self) -> bool:
        """Check for outdated dependencies."""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                
                if outdated:
                    report_path = self.project_root / "reports" / "outdated_dependencies.json"
                    report_path.parent.mkdir(exist_ok=True)
                    
                    report = {
                        "generated": datetime.now().isoformat(),
                        "outdated_packages": outdated,
                        "total_count": len(outdated)
                    }
                    
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    logger.info(f"Found {len(outdated)} outdated dependencies")
                else:
                    logger.info("All dependencies are up to date")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking outdated dependencies: {e}")
            return False
    
    def audit_security_vulnerabilities(self) -> bool:
        """Audit dependencies for security vulnerabilities."""
        try:
            # Use safety to check for vulnerabilities
            result = subprocess.run(
                ["safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                
                report_path = self.project_root / "reports" / "security_vulnerabilities.json"
                report_path.parent.mkdir(exist_ok=True)
                
                report = {
                    "generated": datetime.now().isoformat(),
                    "vulnerabilities": vulnerabilities,
                    "total_count": len(vulnerabilities) if isinstance(vulnerabilities, list) else 0
                }
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                if isinstance(vulnerabilities, list) and vulnerabilities:
                    logger.warning(f"Found {len(vulnerabilities)} security vulnerabilities")
                else:
                    logger.info("No security vulnerabilities found")
            
            return True
            
        except Exception as e:
            logger.error(f"Error auditing security vulnerabilities: {e}")
            return False


class RepositoryAutomation:
    """Main repository automation orchestrator."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_repo = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/photonic-nn-foundry")
        
        # Create logs directory
        (self.project_root / "logs").mkdir(exist_ok=True)
        (self.project_root / "reports").mkdir(exist_ok=True)
    
    def run_github_automation(self) -> bool:
        """Run GitHub-specific automation tasks."""
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not found, skipping GitHub automation")
            return True
        
        github = GitHubAutomation(self.github_token, self.github_repo)
        
        success = True
        success &= github.update_repository_settings()
        success &= github.manage_labels()
        success &= github.cleanup_old_workflows()
        
        # Only attempt branch protection in CI environment
        if os.getenv("CI"):
            success &= github.setup_branch_protection()
        
        return success
    
    def run_code_maintenance(self) -> bool:
        """Run code maintenance tasks."""
        maintenance = CodeMaintenance(self.project_root)
        
        success = True
        success &= maintenance.update_copyright_years()
        success &= maintenance.optimize_imports()
        success &= maintenance.remove_unused_dependencies()
        success &= maintenance.update_docstrings()
        
        return success
    
    def run_dependency_management(self) -> bool:
        """Run dependency management tasks."""
        deps = DependencyManagement(self.project_root)
        
        success = True
        success &= deps.check_outdated_dependencies()
        success &= deps.audit_security_vulnerabilities()
        
        return success
    
    def generate_automation_report(self) -> Dict[str, Any]:
        """Generate automation execution report."""
        report = {
            "execution_time": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "github_repo": self.github_repo,
            "tasks_executed": [
                "github_automation",
                "code_maintenance", 
                "dependency_management"
            ],
            "reports_generated": []
        }
        
        # List generated reports
        reports_dir = self.project_root / "reports"
        if reports_dir.exists():
            report["reports_generated"] = [
                str(f.relative_to(self.project_root))
                for f in reports_dir.glob("*")
                if f.is_file()
            ]
        
        return report
    
    def run_full_automation(self):
        """Run complete repository automation."""
        logger.info("Starting repository automation...")
        
        try:
            # Run automation tasks
            github_success = self.run_github_automation()
            code_success = self.run_code_maintenance()
            deps_success = self.run_dependency_management()
            
            # Generate report
            report = self.generate_automation_report()
            report["success"] = github_success and code_success and deps_success
            
            # Save automation report
            report_path = self.project_root / "reports" / f"automation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Repository automation completed. Report: {report_path}")
            
            if not report["success"]:
                logger.warning("Some automation tasks failed. Check logs for details.")
                sys.exit(1)
            
        except Exception as e:
            logger.error(f"Error during repository automation: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    automation = RepositoryAutomation()
    automation.run_full_automation()


if __name__ == "__main__":
    main()