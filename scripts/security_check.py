"""Security scanner for detecting secrets, hardcoded paths, and PII."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = [
    (re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}", re.ASCII), "Possible AWS access key"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}", re.ASCII), "Possible OpenAI/API secret key"),
    (re.compile(r"/home/\w+/", re.IGNORECASE), "Hardcoded home directory path"),
    (re.compile(r"C:\\Users\\", re.IGNORECASE), "Hardcoded Windows user path"),
    (re.compile(r"weights_only\s*=\s*False"), "torch.load with weights_only=False"),
    (re.compile(r"(?:password|secret|token|api_key)\s*=\s*[\"'][^\"']+[\"']", re.IGNORECASE), "Hardcoded secret"),
]

ALLOWLIST = [
    re.compile(r"pyproject\.toml"),
    re.compile(r"security_check\.py"),
    re.compile(r"\.pre-commit-config\.yaml"),
    re.compile(r"CLAUDE\.md"),
    re.compile(r"conftest\.py"),
]

CONTENT_ALLOWLIST = [
    re.compile(r"file:///home/"),  # pyproject.toml local dependency
    re.compile(r"# .*/home/"),    # Comments with example paths
    re.compile(r"clone.*github\.com"),  # Git clone URLs
]


def check_file(filepath: Path) -> list[str]:
    """Check a single file for security issues.

    Args:
        filepath: Path to the file to check.

    Returns:
        List of issue descriptions found.
    """
    for pattern in ALLOWLIST:
        if pattern.search(str(filepath)):
            return []

    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    issues = []
    for line_num, line in enumerate(content.splitlines(), 1):
        # Skip allowlisted content patterns
        if any(p.search(line) for p in CONTENT_ALLOWLIST):
            continue

        for pattern, description in PATTERNS:
            if pattern.search(line):
                issues.append(f"{filepath}:{line_num}: {description}")
    return issues


def main() -> int:
    """Run security checks on specified paths or default src/tests.

    Returns:
        Exit code: 0 if clean, 1 if issues found.
    """
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:] if not p.startswith("-")]
    else:
        root = Path(__file__).resolve().parent.parent
        targets = [root / "src", root / "tests", root / "scripts", root / "configs"]

    all_issues: list[str] = []
    for target in targets:
        target = Path(target)
        if target.is_file():
            all_issues.extend(check_file(target))
        elif target.is_dir():
            for ext in ("*.py", "*.yaml", "*.yml", "*.toml", "*.md"):
                for filepath in target.rglob(ext):
                    all_issues.extend(check_file(filepath))

    if all_issues:
        print(f"Security issues found ({len(all_issues)}):")
        for issue in all_issues:
            print(f"  {issue}")
        return 1

    print("No security issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
