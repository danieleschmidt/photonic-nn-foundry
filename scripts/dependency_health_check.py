#!/usr/bin/env python3
"""
Dependency health check script for Photonic Neural Network Foundry
Analyzes dependency freshness, security, and compatibility
"""

import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests
import toml
from packaging import version as pkg_version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """Analyze project dependencies for health and security."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.requirements_path = project_root / "requirements.txt"
        self.requirements_dev_path = project_root / "requirements-dev.txt"
    
    def load_project_dependencies(self) -> Dict[str, List[str]]:
        """Load dependencies from project files."""
        dependencies = {
            "production": [],
            "development": [],
            "optional": []
        }
        
        # Load from pyproject.toml if available
        if self.pyproject_path.exists():
            try:
                with open(self.pyproject_path, 'r') as f:
                    pyproject_data = toml.load(f)
                
                project = pyproject_data.get("project", {})
                dependencies["production"] = project.get("dependencies", [])
                dependencies["optional"] = list(project.get("optional-dependencies", {}).values())
                
            except Exception as e:
                logger.warning(f"Error loading pyproject.toml: {e}")
        
        # Load from requirements.txt
        if self.requirements_path.exists():
            try:
                with open(self.requirements_path, 'r') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    dependencies["production"].extend(deps)
            except Exception as e:
                logger.warning(f"Error loading requirements.txt: {e}")
        
        # Load from requirements-dev.txt
        if self.requirements_dev_path.exists():
            try:
                with open(self.requirements_dev_path, 'r') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    dependencies["development"] = deps
            except Exception as e:
                logger.warning(f"Error loading requirements-dev.txt: {e}")
        
        return dependencies
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages and their versions."""
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            packages = json.loads(result.stdout)
            return {pkg["name"]: pkg["version"] for pkg in packages}
            
        except Exception as e:
            logger.error(f"Error getting installed packages: {e}")
            return {}
    
    def check_package_freshness(self, package_name: str, current_version: str) -> Dict[str, Any]:
        """Check if a package has newer versions available."""
        try:
            # Query PyPI API
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            if response.status_code != 200:
                return {"error": "Package not found on PyPI"}
            
            data = response.json()
            latest_version = data["info"]["version"]
            
            # Get release dates
            releases = data.get("releases", {})
            current_release_date = None
            latest_release_date = None
            
            if current_version in releases and releases[current_version]:
                current_release_date = releases[current_version][0].get("upload_time")
            
            if latest_version in releases and releases[latest_version]:
                latest_release_date = releases[latest_version][0].get("upload_time")
            
            # Calculate age
            age_days = None
            if current_release_date:
                try:
                    current_date = datetime.fromisoformat(current_release_date.replace('Z', '+00:00'))
                    age_days = (datetime.now().replace(tzinfo=current_date.tzinfo) - current_date).days
                except Exception:
                    pass
            
            # Compare versions
            is_latest = current_version == latest_version
            needs_update = False
            update_type = "none"
            
            if not is_latest:
                try:
                    current_ver = pkg_version.parse(current_version)
                    latest_ver = pkg_version.parse(latest_version)
                    
                    if latest_ver > current_ver:
                        needs_update = True
                        
                        # Determine update type
                        if latest_ver.major > current_ver.major:
                            update_type = "major"
                        elif latest_ver.minor > current_ver.minor:
                            update_type = "minor"
                        else:
                            update_type = "patch"
                            
                except Exception:
                    needs_update = True
                    update_type = "unknown"
            
            return {
                "package": package_name,
                "current_version": current_version,
                "latest_version": latest_version,
                "is_latest": is_latest,
                "needs_update": needs_update,
                "update_type": update_type,
                "age_days": age_days,
                "current_release_date": current_release_date,
                "latest_release_date": latest_release_date
            }
            
        except Exception as e:
            logger.warning(f"Error checking {package_name}: {e}")
            return {"error": str(e)}
    
    def check_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Check for known security vulnerabilities using safety."""
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                return vulnerabilities if isinstance(vulnerabilities, list) else []
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Error checking security vulnerabilities: {e}")
            return []
    
    def check_license_compatibility(self, package_name: str) -> Dict[str, Any]:
        """Check package license compatibility."""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
            if response.status_code != 200:
                return {"error": "Package not found"}
            
            data = response.json()
            license_info = data["info"].get("license", "Unknown")
            classifiers = data["info"].get("classifiers", [])
            
            # Extract license from classifiers
            license_classifiers = [c for c in classifiers if c.startswith("License ::")]
            
            # Common incompatible licenses for commercial use
            incompatible_licenses = ["GPL", "AGPL", "LGPL"]
            potentially_incompatible = any(lic in license_info.upper() for lic in incompatible_licenses)
            
            return {
                "package": package_name,
                "license": license_info,
                "license_classifiers": license_classifiers,
                "potentially_incompatible": potentially_incompatible
            }
            
        except Exception as e:
            logger.warning(f"Error checking license for {package_name}: {e}")
            return {"error": str(e)}
    
    def analyze_dependency_tree(self) -> Dict[str, Any]:
        """Analyze dependency tree for conflicts and redundancies."""
        try:
            result = subprocess.run(
                ["pipdeptree", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                dependency_tree = json.loads(result.stdout)
                
                # Count total dependencies
                total_packages = len(dependency_tree)
                
                # Find packages with many dependencies
                heavy_packages = []
                for pkg in dependency_tree:
                    dep_count = len(pkg.get("dependencies", []))
                    if dep_count > 10:
                        heavy_packages.append({
                            "package": pkg["package"]["package_name"],
                            "version": pkg["package"]["installed_version"],
                            "dependency_count": dep_count
                        })
                
                return {
                    "total_packages": total_packages,
                    "heavy_packages": heavy_packages,
                    "dependency_tree": dependency_tree
                }
            else:
                return {"error": "Failed to generate dependency tree"}
                
        except Exception as e:
            logger.warning(f"Error analyzing dependency tree: {e}")
            return {"error": str(e)}


class DependencyHealthChecker:
    """Main dependency health checker."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.analyzer = DependencyAnalyzer(self.project_root)
        self.report_data = {}
    
    def extract_package_name(self, dependency_spec: str) -> str:
        """Extract package name from dependency specification."""
        # Handle various formats: package>=1.0, package==1.0.0, package[extra]>=1.0
        import re
        match = re.match(r'^([a-zA-Z0-9_-]+)', dependency_spec)
        return match.group(1) if match else dependency_spec
    
    def run_freshness_analysis(self) -> Dict[str, Any]:
        """Run dependency freshness analysis."""
        logger.info("Analyzing dependency freshness...")
        
        dependencies = self.analyzer.load_project_dependencies()
        installed_packages = self.analyzer.get_installed_packages()
        
        freshness_report = {
            "timestamp": datetime.now().isoformat(),
            "production_dependencies": [],
            "development_dependencies": [],
            "summary": {}
        }
        
        # Analyze production dependencies
        for dep_spec in dependencies["production"]:
            package_name = self.extract_package_name(dep_spec)
            if package_name in installed_packages:
                current_version = installed_packages[package_name]
                freshness_info = self.analyzer.check_package_freshness(package_name, current_version)
                freshness_report["production_dependencies"].append(freshness_info)
        
        # Analyze development dependencies
        for dep_spec in dependencies["development"]:
            package_name = self.extract_package_name(dep_spec)
            if package_name in installed_packages:
                current_version = installed_packages[package_name]
                freshness_info = self.analyzer.check_package_freshness(package_name, current_version)
                freshness_report["development_dependencies"].append(freshness_info)
        
        # Generate summary
        all_deps = freshness_report["production_dependencies"] + freshness_report["development_dependencies"]
        total_deps = len(all_deps)
        outdated_deps = len([d for d in all_deps if d.get("needs_update", False)])
        very_old_deps = len([d for d in all_deps if d.get("age_days", 0) > 365])
        
        freshness_report["summary"] = {
            "total_dependencies": total_deps,
            "outdated_dependencies": outdated_deps,
            "very_old_dependencies": very_old_deps,
            "freshness_score": round((total_deps - outdated_deps) / total_deps * 100, 1) if total_deps > 0 else 100
        }
        
        return freshness_report
    
    def run_security_analysis(self) -> Dict[str, Any]:
        """Run security vulnerability analysis."""
        logger.info("Analyzing security vulnerabilities...")
        
        vulnerabilities = self.analyzer.check_security_vulnerabilities()
        
        security_report = {
            "timestamp": datetime.now().isoformat(),
            "vulnerabilities": vulnerabilities,
            "summary": {
                "total_vulnerabilities": len(vulnerabilities),
                "critical_vulnerabilities": len([v for v in vulnerabilities if v.get("severity") == "critical"]),
                "high_vulnerabilities": len([v for v in vulnerabilities if v.get("severity") == "high"]),
                "medium_vulnerabilities": len([v for v in vulnerabilities if v.get("severity") == "medium"]),
                "low_vulnerabilities": len([v for v in vulnerabilities if v.get("severity") == "low"])
            }
        }
        
        return security_report
    
    def run_license_analysis(self) -> Dict[str, Any]:
        """Run license compatibility analysis."""
        logger.info("Analyzing license compatibility...")
        
        installed_packages = self.analyzer.get_installed_packages()
        license_report = {
            "timestamp": datetime.now().isoformat(),
            "packages": [],
            "summary": {}
        }
        
        # Check licenses for main packages (not all dependencies)
        main_packages = list(installed_packages.keys())[:20]  # Limit to avoid rate limiting
        
        for package_name in main_packages:
            license_info = self.analyzer.check_license_compatibility(package_name)
            if "error" not in license_info:
                license_report["packages"].append(license_info)
        
        # Generate summary
        packages_with_licenses = [p for p in license_report["packages"] if "error" not in p]
        incompatible_packages = [p for p in packages_with_licenses if p.get("potentially_incompatible", False)]
        
        license_report["summary"] = {
            "total_packages_checked": len(packages_with_licenses),
            "potentially_incompatible": len(incompatible_packages),
            "compatibility_score": round((len(packages_with_licenses) - len(incompatible_packages)) / len(packages_with_licenses) * 100, 1) if packages_with_licenses else 100
        }
        
        return license_report
    
    def run_dependency_tree_analysis(self) -> Dict[str, Any]:
        """Run dependency tree analysis."""
        logger.info("Analyzing dependency tree...")
        
        tree_analysis = self.analyzer.analyze_dependency_tree()
        
        if "error" not in tree_analysis:
            tree_report = {
                "timestamp": datetime.now().isoformat(),
                "total_packages": tree_analysis["total_packages"],
                "heavy_packages": tree_analysis["heavy_packages"],
                "summary": {
                    "complexity_score": min(100, max(0, 100 - len(tree_analysis["heavy_packages"]) * 10)),
                    "packages_with_many_deps": len(tree_analysis["heavy_packages"])
                }
            }
        else:
            tree_report = {
                "timestamp": datetime.now().isoformat(),
                "error": tree_analysis["error"],
                "summary": {"complexity_score": 0}
            }
        
        return tree_report
    
    def calculate_overall_health_score(self) -> Dict[str, Any]:
        """Calculate overall dependency health score."""
        scores = {}
        
        if "freshness" in self.report_data:
            scores["freshness"] = self.report_data["freshness"]["summary"]["freshness_score"]
        
        if "security" in self.report_data:
            # Security score: 100 - (critical*20 + high*10 + medium*5 + low*1)
            vuln_summary = self.report_data["security"]["summary"]
            security_penalty = (
                vuln_summary["critical_vulnerabilities"] * 20 +
                vuln_summary["high_vulnerabilities"] * 10 +
                vuln_summary["medium_vulnerabilities"] * 5 +
                vuln_summary["low_vulnerabilities"] * 1
            )
            scores["security"] = max(0, 100 - security_penalty)
        
        if "licenses" in self.report_data:
            scores["license_compatibility"] = self.report_data["licenses"]["summary"]["compatibility_score"]
        
        if "dependency_tree" in self.report_data:
            scores["complexity"] = self.report_data["dependency_tree"]["summary"]["complexity_score"]
        
        # Calculate weighted overall score
        weights = {
            "freshness": 0.25,
            "security": 0.40,
            "license_compatibility": 0.20,
            "complexity": 0.15
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores if key in weights)
        
        return {
            "overall_score": round(overall_score, 1),
            "component_scores": scores,
            "grade": self.get_health_grade(overall_score)
        }
    
    def get_health_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if "freshness" in self.report_data:
            freshness = self.report_data["freshness"]["summary"]
            if freshness["freshness_score"] < 80:
                recommendations.append(f"Consider updating {freshness['outdated_dependencies']} outdated dependencies")
            if freshness["very_old_dependencies"] > 0:
                recommendations.append(f"Review {freshness['very_old_dependencies']} very old dependencies (>1 year)")
        
        if "security" in self.report_data:
            security = self.report_data["security"]["summary"]
            if security["critical_vulnerabilities"] > 0:
                recommendations.append(f"URGENT: Fix {security['critical_vulnerabilities']} critical security vulnerabilities")
            if security["high_vulnerabilities"] > 0:
                recommendations.append(f"Fix {security['high_vulnerabilities']} high-severity security vulnerabilities")
        
        if "licenses" in self.report_data:
            licenses = self.report_data["licenses"]["summary"]
            if licenses["potentially_incompatible"] > 0:
                recommendations.append(f"Review {licenses['potentially_incompatible']} packages with potentially incompatible licenses")
        
        if "dependency_tree" in self.report_data:
            tree = self.report_data["dependency_tree"]
            if "heavy_packages" in tree and tree["heavy_packages"]:
                recommendations.append(f"Consider alternatives to heavy dependencies: {', '.join([p['package'] for p in tree['heavy_packages'][:3]])}")
        
        if not recommendations:
            recommendations.append("Dependencies are in good health! Continue regular monitoring.")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete dependency health check."""
        logger.info("Starting dependency health check...")
        
        try:
            # Run all analyses
            self.report_data["freshness"] = self.run_freshness_analysis()
            self.report_data["security"] = self.run_security_analysis()
            self.report_data["licenses"] = self.run_license_analysis()
            self.report_data["dependency_tree"] = self.run_dependency_tree_analysis()
            
            # Calculate overall health
            self.report_data["health_score"] = self.calculate_overall_health_score()
            self.report_data["recommendations"] = self.generate_recommendations()
            
            # Add metadata
            self.report_data["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "analysis_version": "1.0.0"
            }
            
            # Save report
            report_path = self.project_root / "reports" / f"dependency_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(self.report_data, f, indent=2)
            
            # Print summary
            health_score = self.report_data["health_score"]
            print(f"\n{'='*60}")
            print(f"DEPENDENCY HEALTH CHECK REPORT")
            print(f"{'='*60}")
            print(f"Overall Health Score: {health_score['overall_score']}/100 (Grade: {health_score['grade']})")
            print(f"")
            print(f"Component Scores:")
            for component, score in health_score['component_scores'].items():
                print(f"  {component.replace('_', ' ').title()}: {score}/100")
            print(f"")
            print(f"Recommendations:")
            for i, rec in enumerate(self.report_data["recommendations"], 1):
                print(f"  {i}. {rec}")
            print(f"")
            print(f"Detailed report saved: {report_path}")
            print(f"{'='*60}")
            
            logger.info("Dependency health check completed successfully")
            
        except Exception as e:
            logger.error(f"Error during dependency health check: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    checker = DependencyHealthChecker()
    checker.run_complete_analysis()


if __name__ == "__main__":
    main()