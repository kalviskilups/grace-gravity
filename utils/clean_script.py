#!/usr/bin/env python3
"""
Utility script for cleaning GRACE GFC files.

This script removes unnecessary reference blocks from GFC files to create
clean versions more suitable for processing.
"""

import argparse
import os
import re
from typing import List, Tuple


def clean_gfc_file(filepath: str) -> Tuple[int, str]:
    """
    Remove reference blocks from a GFC file.

    Args:
        filepath: Path to the GFC file to clean

    Returns:
        Tuple of (number of substitutions made, encoding used)

    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If the file encoding cannot be determined
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Define the pattern to match and remove reference blocks
    pattern = re.compile(
        r"^\s*Reference:\s*\r?\n"  # Match 'Reference:' with optional whitespace and newline
        r"(?:.*\r?\n)+?"  # Match one or more lines, non-greedily
        r"(?=^\s*(?:time_period_of_data|begin_of_head))",  # Lookahead for the next header
        re.MULTILINE,
    )

    # Try different encodings
    encodings = ["utf-8", "latin-1", "cp1252"]
    content = None
    encoding_used = None

    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                content = f.read()
            encoding_used = enc
            break
        except UnicodeDecodeError:
            continue

    if content is None or encoding_used is None:
        raise UnicodeDecodeError("Failed to decode file with any of the attempted encodings")

    # Apply the substitution
    new_content, num_subs = re.subn(pattern, "", content)

    # Write back the cleaned content
    with open(filepath, "w", encoding=encoding_used) as f:
        f.write(new_content)

    return num_subs, encoding_used


def clean_directory(directory: str, pattern: str = "*.gfc") -> List[Tuple[str, int, str]]:
    """
    Clean all GFC files in a directory matching a pattern.

    Args:
        directory: Directory containing GFC files
        pattern: File pattern to match (default: "*.gfc")

    Returns:
        List of tuples (filename, substitutions, encoding) for each processed file

    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    import glob

    results = []

    # Find all files matching the pattern
    for filepath in glob.glob(os.path.join(directory, pattern)):
        try:
            num_subs, encoding = clean_gfc_file(filepath)
            filename = os.path.basename(filepath)
            results.append((filename, num_subs, encoding))
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Clean GRACE GFC files by removing reference blocks.")
    parser.add_argument("--directory", "-d", required=True, help="Directory containing GFC files")
    parser.add_argument("--pattern", "-p", default="*.gfc", help="File pattern to match (default: *.gfc)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    try:
        results = clean_directory(args.directory, args.pattern)

        # Print summary
        total_files = len(results)
        total_subs = sum(r[1] for r in results)
        print(f"Processed {total_files} files with {total_subs} total substitutions.")

        # Print details if verbose
        if args.verbose:
            print("\nDetailed results:")
            for filename, num_subs, encoding in results:
                print(f"{filename}: {num_subs} substitution(s) made using encoding {encoding}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
