"""
Writes requirements-api.txt with the EXACT package versions installed in your venv.
The saved model must be loaded with the same scikit-learn version it was trained with,
so the Docker image needs the same versions.

Run once (with your venv active) from the fraud-detector folder:
    python make_requirements.py
"""

import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "joblib", "fastapi", "uvicorn", "sqlalchemy", "pydantic"]

lines, missing = [], []
for package in PACKAGES:
    try:
        lines.append(f"{package}=={version(package)}")
    except PackageNotFoundError:
        missing.append(package)

if missing:
    print("These packages are not installed in this environment:", ", ".join(missing))
    print("Install them with:  python -m pip install " + " ".join(missing))
    sys.exit(1)

lines.append("psycopg2-binary")  # the PostgreSQL driver
with open("requirements-api.txt", "w") as file:
    file.write("\n".join(lines) + "\n")

print("Saved requirements-api.txt:")
print("\n".join(lines))
