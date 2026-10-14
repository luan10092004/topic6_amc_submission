from pathlib import Path
import importlib.util
import os
import subprocess
import sys


# =========================================================
# AMC TOPIC 6 - QUICK DEMO LAUNCHER
# =========================================================

ROOT = Path(__file__).resolve().parent


# =========================================================
# 1. REQUIRED PYTHON PACKAGES
# =========================================================

REQUIRED_MODULES = [
    "streamlit",
    "torch",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
]


missing_modules = [
    module
    for module in REQUIRED_MODULES
    if importlib.util.find_spec(module) is None
]


if missing_modules:

    print()
    print("=" * 60)
    print("MISSING PYTHON PACKAGES")
    print("=" * 60)

    print(
        "Missing:",
        ", ".join(missing_modules),
    )

    print()
    print("Install dependencies with:")
    print()

    print(
        f'"{sys.executable}" '
        "-m pip install -r requirements.txt"
    )

    print()

    sys.exit(1)


# =========================================================
# 2. REQUIRED QUICK-DEMO FILES
# =========================================================

required_files = [
    ROOT
    / "data"
    / "demo"
    / "demo_samples.npz",

    ROOT
    / "models"
    / "iq_cnn1d.pt",

    ROOT
    / "models"
    / "spectrogram_cnn2d.pt",

    ROOT
    / "models"
    / "density_cnn2d.pt",

    ROOT
    / "models"
    / "final_results_complete.csv",

    ROOT
    / "demo"
    / "app.py",
]


missing_files = [
    path
    for path in required_files
    if not path.exists()
]


if missing_files:

    print()
    print("=" * 60)
    print("MISSING DEMO FILES")
    print("=" * 60)

    for path in missing_files:

        try:
            shown_path = path.relative_to(
                ROOT
            )
        except ValueError:
            shown_path = path

        print(
            "-",
            shown_path,
        )

    print()
    print(
        "The quick demo cannot start "
        "until these files are available."
    )

    sys.exit(1)


# =========================================================
# 3. DEMO INFORMATION
# =========================================================

print()
print("=" * 60)
print("AMC TOPIC 6 - MULTI-DOMAIN DEMO")
print("=" * 60)

print(
    "Python executable:",
    sys.executable,
)

print(
    "Repository root:",
    ROOT,
)

print()
print(
    "Mode       : Quick GitHub Demo"
)

print(
    "Samples    : 240 balanced unseen test bursts"
)

print(
    "Classes    : 8 modulation classes"
)

print(
    "SNR range  : -10 dB to +18 dB"
)

print(
    "Input      : Raw I/Q [2 x 128]"
)

print()
print(
    "Launching Streamlit..."
)

print(
    "Press Ctrl+C to stop."
)

print()


# =========================================================
# 4. FORCE QUICK-DEMO MODE
# =========================================================

environment = os.environ.copy()

environment[
    "AMC_DEMO_MODE"
] = "quick"


# =========================================================
# 5. LAUNCH STREAMLIT
# =========================================================

app_path = (
    ROOT
    / "demo"
    / "app.py"
)


command = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(app_path),
]


try:

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
    )

    if result.returncode != 0:

        print()
        print(
            "Streamlit exited with code:",
            result.returncode,
        )

        sys.exit(
            result.returncode
        )


except KeyboardInterrupt:

    print()
    print(
        "Demo stopped by user."
    )