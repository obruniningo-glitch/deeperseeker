"""v2 test runner: runs all tests_v2 suites."""
import os, sys, runpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
suites = ["test_ir_adapters"]
failed = 0
for s in suites:
    print(f"=== {s} ===")
    rc = runpy.run_path(os.path.join(os.path.dirname(__file__), s + ".py"), run_name="__main__")
