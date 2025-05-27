#!/usr/bin/env python3
"""
Set up local configuration files for the project.

This script creates a local paths.yaml configuration file based on the template,
with paths adjusted for the current environment.
"""

import os
import sys
from pathlib import Path


def main():
    # Get project root directory
    project_root = Path(__file__).parent.parent.resolve()

    # Define template and output paths
    template_path = project_root / "configs" / "paths.yaml.template"
    output_path = project_root / "configs" / "paths.yaml"

    # Check if the file already exists
    if output_path.exists():
        print(
            f"{output_path} already exists. Please remove it first if you want to regenerate it."
        )
        return 1

    # Read the template
    try:
        with open(template_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}")
        return 1

    # Replace variables
    content = content.replace(
        '"${PROJECT_DIR}"', f'"{str(project_root).replace(os.sep, "/")}"'
    )

    # Write the output file
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Created local configuration file at: {output_path}")
        print("Please review and customize the paths as needed.")
        return 0
    except PermissionError:
        print(f"Error: Permission denied when writing to {output_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
