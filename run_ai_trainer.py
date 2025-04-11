# Simple utility script to start the AI Trainer
import os
import sys

# Add the package directory to the path if needed
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

try:
    from ai_platform_trainer.main import main
    main()
except ImportError as e:
    print(f"Error: {e}")
    # Try alternative import
    try:
        import ai_platform_trainer
        print(f"AI Platform Trainer package found at: {ai_platform_trainer.__file__}")
        # Call directly through module
        import runpy
        runpy.run_module("ai_platform_trainer", run_name="__main__")
    except ImportError as e2:
        print(f"Failed to import the AI Platform Trainer package: {e2}")
        print("\nTroubleshooting suggestions:")
        print("1. Make sure the package is installed: pip install -e .")
        print("2. Check that the package directory is in your PYTHONPATH")
        print("3. Verify that the package structure is correct")
        sys.exit(1)
