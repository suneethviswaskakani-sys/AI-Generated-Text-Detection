"""
Standalone test runner for the AI-Generated Text Detection System.
Executes all automated tests without requiring external test dependencies.
Usage:
    python run_tests.py
"""
import sys
import unittest

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AI-GENERATED TEXT DETECTION VERSION 2.0 TEST SUITE")
    print("=" * 70)
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n[SUCCESS] ALL AUTOMATED TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n[FAILURE] {len(result.failures)} failures, {len(result.errors)} errors.")
        sys.exit(1)
