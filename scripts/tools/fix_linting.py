#!/usr/bin/env python3
"""
Script to automatically fix common linting issues in the codebase.

This script fixes:
1. Trailing whitespace
2. Blank lines with whitespace
3. Removes unused imports
4. Fixes spacing between functions and classes
"""

import re
import sys
from pathlib import Path
import subprocess
from typing import List, Tuple

# Root directory
ROOT_DIR = Path(__file__).parents[2]  # Go up two levels from scripts/tools


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in the given directory and subdirectories."""
    return list(directory.glob("**/*.py"))


def fix_trailing_whitespace(file_path: Path) -> int:
    """Fix trailing whitespace in a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace trailing whitespace on lines
    new_content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)

    # Count fixes
    fixes = content.count("\n") - new_content.count("\n")

    # Only write if changes were made
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

    return fixes


def fix_blank_lines_with_whitespace(file_path: Path) -> int:
    """Fix blank lines containing whitespace."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace lines with only whitespace with empty lines
    new_content = re.sub(r"^[ \t]+$", "", content, flags=re.MULTILINE)

    # Count fixes
    fixes = content.count("\n") - new_content.count("\n")

    # Only write if changes were made
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

    return fixes


def fix_double_blank_lines(file_path: Path) -> int:
    """Ensure there are exactly two blank lines between top-level definitions."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Track the current state of the file
    in_class_or_function = False
    new_lines = []
    fixes = 0

    for i, line in enumerate(lines):
        # Detect if we're starting a new class or function
        if re.match(r"^(class|def)\s+", line) and not in_class_or_function:
            # Check if we have exactly two blank lines before
            blank_count = 0
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                blank_count += 1
                j -= 1

            # If we don't have exactly two blank lines and we're not at the start of the file
            if blank_count != 2 and i > 0 and j >= 0:
                # Remove any existing blank lines before this
                while new_lines and new_lines[-1].strip() == "":
                    new_lines.pop()
                    fixes += 1

                # Add exactly two blank lines
                new_lines.append("\n")
                new_lines.append("\n")
                fixes += 2

            in_class_or_function = True

        # Add the current line
        new_lines.append(line)

        # If we have a line that's not a continuation and not blank, we're out of the definition
        if (
            line.strip()
            and not line.strip().endswith("\\")
            and not line.strip().endswith("(")
        ):
            in_class_or_function = False

    # Only write if changes were made
    if fixes > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return fixes


def fix_unused_imports(file_path: Path) -> int:
    """Remove unused imports using autoflake."""
    try:
        result = subprocess.run(
            ["autoflake", "--remove-all-unused-imports", "--in-place", str(file_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: autoflake failed on {file_path}: {result.stderr}")
            return 0

        # Just check if the command succeeded
        return 1
    except Exception as e:
        print(f"Error running autoflake on {file_path}: {e}")
        return 0


def fix_file(file_path: Path) -> Tuple[int, int, int, int]:
    """Apply all fixes to a single file and return the number of fixes by type."""
    print(f"Processing {file_path}...")

    whitespace_fixes = fix_trailing_whitespace(file_path)
    blank_line_fixes = fix_blank_lines_with_whitespace(file_path)
    double_blank_fixes = fix_double_blank_lines(file_path)
    import_fixes = fix_unused_imports(file_path)

    return (whitespace_fixes, blank_line_fixes, double_blank_fixes, import_fixes)


def main():
    """Main function to fix linting issues in the codebase."""
    # Check if autoflake is installed
    try:
        subprocess.run(["autoflake", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: autoflake is not installed. Please install it with:")
        print("pip install autoflake")
        sys.exit(1)

    # Find all Python files
    python_files = find_python_files(ROOT_DIR)

    # Initialize counters
    total_files = len(python_files)
    total_whitespace_fixes = 0
    total_blank_line_fixes = 0
    total_double_blank_fixes = 0
    total_import_fixes = 0

    # Process each file
    for file_path in python_files:
        try:
            fixes = fix_file(file_path)
            total_whitespace_fixes += fixes[0]
            total_blank_line_fixes += fixes[1]
            total_double_blank_fixes += fixes[2]
            total_import_fixes += fixes[3]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Print summary
    print("\nLinting fixes complete!")
    print(f"Processed {total_files} Python files")
    print(f"Fixed {total_whitespace_fixes} instances of trailing whitespace")
    print(f"Fixed {total_blank_line_fixes} blank lines with whitespace")
    print(
        f"Fixed {total_double_blank_fixes} instances of incorrect blank lines between defs"
    )
    print(f"Removed {total_import_fixes} unused imports")
    print("\nRemaining issues may require manual fixes.")


if __name__ == "__main__":
    main()
