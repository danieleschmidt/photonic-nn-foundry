#!/usr/bin/env python3
"""
Generate Software Bill of Materials (SBOM) for Photonic Neural Network Foundry
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pkg_resources
import platform


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        commit_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
        
        return {
            "commit_hash": commit_hash,
            "commit_date": commit_date,
            "branch": branch,
            "remote_url": remote_url
        }
    except subprocess.CalledProcessError:
        return {
            "commit_hash": "unknown",
            "commit_date": "unknown",
            "branch": "unknown",
            "remote_url": "unknown"
        }


def get_python_dependencies() -> List[Dict[str, Any]]:
    """Get list of Python dependencies with versions."""
    dependencies = []
    
    try:
        # Get installed packages
        installed_packages = [d for d in pkg_resources.working_set]
        
        for package in installed_packages:
            dep_info = {
                "name": package.project_name,
                "version": package.version,
                "type": "python-package",
                "location": package.location,
                "requires": [str(req) for req in package.requires()]
            }
            
            # Try to get additional metadata
            try:
                metadata = package.get_metadata("METADATA")
                lines = metadata.split("\n")
                for line in lines:
                    if line.startswith("Home-page:"):
                        dep_info["homepage"] = line.split(":", 1)[1].strip()
                    elif line.startswith("License:"):
                        dep_info["license"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Summary:"):
                        dep_info["description"] = line.split(":", 1)[1].strip()
            except Exception:
                pass
            
            dependencies.append(dep_info)
    
    except Exception as e:
        print(f"Warning: Could not get Python dependencies: {e}", file=sys.stderr)
    
    return sorted(dependencies, key=lambda x: x["name"])


def get_system_dependencies() -> List[Dict[str, Any]]:
    """Get system-level dependencies."""
    system_deps = []
    
    # Python runtime
    system_deps.append({
        "name": "python",
        "version": platform.python_version(),
        "type": "runtime",
        "description": "Python programming language runtime"
    })
    
    # Operating system
    system_deps.append({
        "name": platform.system().lower(),
        "version": platform.release(),
        "type": "operating-system",
        "description": f"{platform.system()} operating system"
    })
    
    # Architecture
    system_deps.append({
        "name": "architecture",
        "version": platform.machine(),
        "type": "hardware",
        "description": "System architecture"
    })
    
    return system_deps


def get_docker_dependencies() -> List[Dict[str, Any]]:
    """Get Docker image dependencies."""
    docker_deps = []
    
    # Check if Dockerfile exists and extract base images
    dockerfile_path = Path("Dockerfile")
    if dockerfile_path.exists():
        try:
            with open(dockerfile_path, "r") as f:
                content = f.read()
            
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("FROM "):
                    image = line.split("FROM ", 1)[1].split(" as ")[0].strip()
                    if image != "scratch":
                        parts = image.split(":")
                        name = parts[0]
                        version = parts[1] if len(parts) > 1 else "latest"
                        
                        docker_deps.append({
                            "name": name,
                            "version": version,
                            "type": "docker-image",
                            "description": f"Docker base image: {image}"
                        })
        except Exception as e:
            print(f"Warning: Could not parse Dockerfile: {e}", file=sys.stderr)
    
    return docker_deps


def get_project_info() -> Dict[str, Any]:
    """Get project information from pyproject.toml."""
    project_info = {
        "name": "photonic-nn-foundry",
        "version": "unknown",
        "description": "Turn the latest silicon-photonic AI accelerators into a reproducible software stack",
        "license": "MIT",
        "authors": [],
        "homepage": "https://github.com/danieleschmidt/photonic-nn-foundry"
    }
    
    # Try to read from pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        try:
            import tomli
            with open(pyproject_path, "rb") as f:
                data = tomli.load(f)
            
            project = data.get("project", {})
            project_info.update({
                "name": project.get("name", project_info["name"]),
                "version": project.get("version", project_info["version"]),
                "description": project.get("description", project_info["description"]),
                "license": project.get("license", {}).get("text", project_info["license"]),
                "authors": [author.get("name", "") for author in project.get("authors", [])],
                "homepage": project.get("urls", {}).get("Homepage", project_info["homepage"])
            })
        except ImportError:
            print("Warning: tomli not available, using default project info", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not parse pyproject.toml: {e}", file=sys.stderr)
    
    return project_info


def generate_sbom() -> Dict[str, Any]:
    """Generate complete SBOM."""
    git_info = get_git_info()
    project_info = get_project_info()
    
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:photonic-foundry-{datetime.now().isoformat()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "tools": [
                {
                    "vendor": "terragon-labs",
                    "name": "sbom-generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": project_info["name"],
                "name": project_info["name"],
                "version": project_info["version"],
                "description": project_info["description"],
                "licenses": [
                    {
                        "license": {
                            "name": project_info["license"]
                        }
                    }
                ],
                "externalReferences": [
                    {
                        "type": "website",
                        "url": project_info["homepage"]
                    },
                    {
                        "type": "vcs",
                        "url": git_info["remote_url"]
                    }
                ]
            },
            "properties": [
                {
                    "name": "git:commit",
                    "value": git_info["commit_hash"]
                },
                {
                    "name": "git:branch",
                    "value": git_info["branch"]
                },
                {
                    "name": "git:commit-date",
                    "value": git_info["commit_date"]
                },
                {
                    "name": "build:platform",
                    "value": platform.platform()
                },
                {
                    "name": "build:python-version",
                    "value": platform.python_version()
                }
            ]
        },
        "components": []
    }
    
    # Add Python dependencies
    python_deps = get_python_dependencies()
    for dep in python_deps:
        component = {
            "type": "library",
            "bom-ref": f"python:{dep['name']}@{dep['version']}",
            "name": dep["name"],
            "version": dep["version"],
            "scope": "required",
            "purl": f"pkg:pypi/{dep['name']}@{dep['version']}"
        }
        
        if "description" in dep:
            component["description"] = dep["description"]
        
        if "license" in dep:
            component["licenses"] = [{"license": {"name": dep["license"]}}]
        
        if "homepage" in dep:
            component["externalReferences"] = [
                {"type": "website", "url": dep["homepage"]}
            ]
        
        sbom["components"].append(component)
    
    # Add system dependencies
    system_deps = get_system_dependencies()
    for dep in system_deps:
        component = {
            "type": "operating-system" if dep["type"] == "operating-system" else "library",
            "bom-ref": f"system:{dep['name']}@{dep['version']}",
            "name": dep["name"],
            "version": dep["version"],
            "scope": "required",
            "description": dep["description"]
        }
        sbom["components"].append(component)
    
    # Add Docker dependencies
    docker_deps = get_docker_dependencies()
    for dep in docker_deps:
        component = {
            "type": "container",
            "bom-ref": f"docker:{dep['name']}@{dep['version']}",
            "name": dep["name"],
            "version": dep["version"],
            "scope": "required",
            "description": dep["description"],
            "purl": f"pkg:docker/{dep['name']}@{dep['version']}"
        }
        sbom["components"].append(component)
    
    return sbom


def main():
    """Main function."""
    try:
        sbom = generate_sbom()
        
        # Write SBOM to file
        output_path = Path("sbom.json")
        with open(output_path, "w") as f:
            json.dump(sbom, f, indent=2, sort_keys=True)
        
        print(f"SBOM generated successfully: {output_path}")
        
        # Generate summary
        num_components = len(sbom["components"])
        python_deps = len([c for c in sbom["components"] if c["bom-ref"].startswith("python:")])
        system_deps = len([c for c in sbom["components"] if c["bom-ref"].startswith("system:")])
        docker_deps = len([c for c in sbom["components"] if c["bom-ref"].startswith("docker:")])
        
        print(f"Components found:")
        print(f"  - Python packages: {python_deps}")
        print(f"  - System dependencies: {system_deps}")
        print(f"  - Docker images: {docker_deps}")
        print(f"  - Total: {num_components}")
        
    except Exception as e:
        print(f"Error generating SBOM: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()