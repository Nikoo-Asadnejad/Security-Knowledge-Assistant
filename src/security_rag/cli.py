"""Command-line entry point for the project."""

import argparse
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Security Knowledge Assistant")
    parser.add_argument("--version", action="version", version=version("security-rag"))
    parser.parse_args()
    parser.print_help()
