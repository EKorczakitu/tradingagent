import sys
import os

print("=== DIAGNOSTIC REPORT ===")
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Working directory:", os.getcwd())

libs = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "torch",
    "gymnasium",
    "stable_baselines3",
    "sb3_contrib",
    "optuna",
    "polars",
    "pyarrow"
]

for lib in libs:
    try:
        print(f"Importing {lib}...", end=" ", flush=True)
        mod = __import__(lib)
        print(f"SUCCESS (version: {getattr(mod, '__version__', 'unknown')}, path: {getattr(mod, '__file__', 'unknown')})")
    except Exception as e:
        print(f"FAILED with error: {e}")
    except BaseException as e:
        # Catch system exit, keyboard interrupt, or generator exit, and critical crashes (though core dump won't be caught here)
        print(f"CRITICAL FAILED: {type(e)}")

print("=== END OF REPORT ===")
