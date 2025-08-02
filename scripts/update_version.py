#!/usr/bin/env python3
"""
Update version information across the project
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def update_pyproject_toml(version: str) -> bool:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print(f"Warning: {pyproject_path} not found")
        return False
    
    try:
        with open(pyproject_path, "r") as f:
            content = f.read()
        
        # Update version line
        updated_content = re.sub(
            r'^version\s*=\s*["\'][^"\']*["\']',
            f'version = "{version}"',
            content,
            flags=re.MULTILINE
        )
        
        with open(pyproject_path, "w") as f:
            f.write(updated_content)
        
        print(f"Updated version in {pyproject_path}")
        return True
    
    except Exception as e:
        print(f"Error updating {pyproject_path}: {e}")
        return False


def update_version_file(version: str) -> bool:
    """Update or create version file."""
    version_dir = Path("src/photonic_foundry")
    version_dir.mkdir(parents=True, exist_ok=True)
    version_path = version_dir / "_version.py"
    
    try:
        content = f'''"""Version information for Photonic Neural Network Foundry."""

__version__ = "{version}"
__version_info__ = tuple(int(x) for x in __version__.split(".") if x.isdigit())
'''
        
        with open(version_path, "w") as f:
            f.write(content)
        
        print(f"Updated version in {version_path}")
        return True
    
    except Exception as e:
        print(f"Error updating {version_path}: {e}")
        return False


def update_init_file(version: str) -> bool:
    """Update __init__.py to import version."""
    init_path = Path("src/photonic_foundry/__init__.py")
    if not init_path.exists():
        return True  # Skip if doesn't exist
    
    try:
        with open(init_path, "r") as f:
            content = f.read()
        
        # Check if version import already exists
        if "__version__" not in content:
            # Add version import
            version_import = "\nfrom ._version import __version__, __version_info__\n"
            content += version_import
        else:
            # Update existing import
            content = re.sub(
                r'from\s+\._version\s+import\s+__version__.*',
                'from ._version import __version__, __version_info__',
                content
            )
        
        with open(init_path, "w") as f:
            f.write(content)
        
        print(f"Updated version import in {init_path}")
        return True
    
    except Exception as e:
        print(f"Error updating {init_path}: {e}")
        return False


def update_dockerfile(version: str) -> bool:
    """Update version label in Dockerfile."""
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        return True  # Skip if doesn't exist
    
    try:
        with open(dockerfile_path, "r") as f:
            lines = f.readlines()
        
        updated_lines = []
        version_label_added = False
        
        for line in lines:
            if line.strip().startswith("LABEL version="):
                updated_lines.append(f'LABEL version="{version}"\n')
                version_label_added = True
            elif line.strip().startswith("LABEL") and "version" in line:
                # Update existing version label
                updated_lines.append(re.sub(
                    r'version="[^"]*"',
                    f'version="{version}"',
                    line
                ))
                version_label_added = True
            else:
                updated_lines.append(line)
        
        # Add version label if not present
        if not version_label_added:
            # Find a good place to insert the label (after FROM statements)
            insert_index = 0
            for i, line in enumerate(updated_lines):
                if line.strip().startswith("FROM "):
                    insert_index = i + 1
            
            updated_lines.insert(insert_index, f'LABEL version="{version}"\n')
        
        with open(dockerfile_path, "w") as f:
            f.writelines(updated_lines)
        
        print(f"Updated version in {dockerfile_path}")
        return True
    
    except Exception as e:
        print(f"Error updating {dockerfile_path}: {e}")
        return False


def validate_version(version: str) -> bool:
    """Validate semantic version format."""
    pattern = r'^\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?$'
    if not re.match(pattern, version):
        print(f"Error: Invalid version format: {version}")
        print("Expected format: MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-alpha.N")
        return False
    return True


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        print("Example: python update_version.py 1.2.3")
        sys.exit(1)
    
    version = sys.argv[1]
    
    if not validate_version(version):
        sys.exit(1)
    
    print(f"Updating version to: {version}")
    
    success_count = 0
    total_updates = 0
    
    updates: List[Tuple[str, callable]] = [
        ("pyproject.toml", lambda: update_pyproject_toml(version)),
        ("version file", lambda: update_version_file(version)),
        ("__init__.py", lambda: update_init_file(version)),
        ("Dockerfile", lambda: update_dockerfile(version)),
    ]
    
    for name, update_func in updates:
        total_updates += 1
        if update_func():
            success_count += 1
        else:
            print(f"Failed to update {name}")
    
    print(f"\nVersion update complete: {success_count}/{total_updates} successful")
    
    if success_count == total_updates:
        print(f"✅ All version references updated to {version}")
    else:
        print(f"⚠️  Some updates failed. Please check manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()