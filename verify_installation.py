"""
Installation Verification Script
Checks all required files and imports are working
"""

import os
import sys

print("=" * 70)
print("🔍 ML Platform Installation Verification")
print("=" * 70)

# Check files exist
required_files = [
    "main.py",
    "app/__init__.py",
    "app/core/database.py",
    "app/core/auth.py",
    "app/models/models.py",
    "app/models/__init__.py",
    "app/api/auth.py",
    "app/api/workspaces.py",
    "app/api/projects.py",
    "app/api/models.py",
    "app/api/activities.py",
    "app/api/datasources.py",
    "app/api/datasets.py",
]

print("\n📁 Checking files...")
all_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - MISSING!")
        all_exist = False

if not all_exist:
    print("\n❌ Some files are missing!")
    sys.exit(1)

print("\n✅ All files present!")

# Try importing models
print("\n🔄 Checking imports...")
try:
    from app.core.database import engine, Base
    print("  ✅ Database imports")
    
    from app.models.models import User, Workspace, Project, Datasource, Dataset, Model, Activity
    print("  ✅ All models imported")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ Installation verification PASSED!")
print("=" * 70)
print("\n🚀 Ready to run: python main.py")

