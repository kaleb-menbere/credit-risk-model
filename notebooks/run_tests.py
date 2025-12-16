"""
Simple test runner for the project
"""

import subprocess
import sys
import os

def run_tests():
    """Run all tests in the tests directory"""
    print("=" * 60)
    print("CREDIT RISK MODEL - TEST RUNNER")
    print("=" * 60)
    
    # Check if tests directory exists
    if not os.path.exists("tests"):
        print("[ERROR] tests directory not found!")
        print("Please create tests directory with test files.")
        return 1
    
    # Check if there are any test files
    test_files = [f for f in os.listdir("tests") if f.startswith("test_") and f.endswith(".py")]
    if not test_files:
        print("[WARNING] No test files found in tests/ directory")
        print("Test files should start with 'test_' and end with '.py'")
    
    print(f"\nFound {len(test_files)} test files: {', '.join(test_files)}")
    print("\nRunning unit tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("\n[WARNINGS/ERRORS]:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ SUCCESS: All tests passed!")
        else:
            print(f"\n❌ FAILURE: Tests failed with exit code {result.returncode}")
        
        return result.returncode
        
    except FileNotFoundError:
        print("\n[ERROR] pytest not found!")
        print("Please install pytest: pip install pytest")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    print("\n" + "=" * 60)
    exit_code = run_tests()
    print("=" * 60)
    sys.exit(exit_code)