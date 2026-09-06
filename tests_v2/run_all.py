"""v2 test runner: runs all tests_v2 suites."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
suites = ["test_ir_adapters"]
failed = []
for s in suites:
    print(f"=== {s} ===")
    try:
        runpy.run_path(os.path.join(os.path.dirname(__file__), s + ".py"), run_name="__main__")
    except SystemExit as e:
        if e.code:
            failed.append(s)
    print()
if failed:
    print(f"FAILED suites: {', '.join(failed)}")
    sys.exit(1)
print("all suites passed")
