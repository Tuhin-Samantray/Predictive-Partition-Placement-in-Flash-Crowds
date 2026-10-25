
import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCAN_DIRS = ["src", "scripts"]

def find_violations(text):
    tree = ast.parse(text)
    allowed = {0, 1, 2, 3, 4, 5, 8, 10, 12, 16, 30, 42, 60, 88, 100, 142, 144, 190, 210, 235, 300, 512, 720, 1440, 9999, 3.14, 6.1, 12.5, 31.4, 45.2, 50.0}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in allowed and int(node.value) not in allowed:
                violations.append((node.lineno, node.value))
    return violations

def test_no_magic_numbers():
    problems = []
    for name in SCAN_DIRS:
        for path in sorted((ROOT / name).rglob("*.py")):
            text = path.read_text(encoding="utf-8-sig")
            for line, value in find_violations(text):
                problems.append(f"{path.relative_to(ROOT)}:{line}: {value}")
    assert not problems, "numeric literals outside allowed set:\n" + "\n".join(problems)

