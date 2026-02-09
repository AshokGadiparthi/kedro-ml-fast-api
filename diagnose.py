"""
Diagnostic script to check database initialization
"""
import sys
sys.path.insert(0, '/home/claude/ml_platform_phase1')

print("=" * 70)
print("🔍 DIAGNOSTIC: Checking Database Initialization")
print("=" * 70)

# Step 1: Check imports
print("\n1️⃣  Testing imports...")
try:
    from app.core.database import engine, Base
    print("   ✅ Database engine imported")
except Exception as e:
    print(f"   ❌ Database import failed: {e}")
    sys.exit(1)

try:
    from app.models.models import User, Workspace, Project, Datasource, Dataset, Model, Activity
    print("   ✅ All models imported")
except Exception as e:
    print(f"   ❌ Models import failed: {e}")
    sys.exit(1)

# Step 2: Check Base.metadata
print("\n2️⃣  Checking Base.metadata...")
print(f"   Tables registered: {len(Base.metadata.tables)}")
if Base.metadata.tables:
    print("   Registered tables:")
    for table_name in Base.metadata.tables:
        print(f"      - {table_name}")
else:
    print("   ⚠️  NO TABLES REGISTERED! Models not properly imported.")
    sys.exit(1)

# Step 3: Try to create tables
print("\n3️⃣  Creating tables...")
try:
    Base.metadata.create_all(bind=engine)
    print("   ✅ Tables created successfully")
except Exception as e:
    print(f"   ❌ Table creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Verify tables exist
print("\n4️⃣  Verifying tables in database...")
from sqlalchemy import inspect
inspector = inspect(engine)
existing_tables = inspector.get_table_names()
print(f"   Tables in database: {len(existing_tables)}")
if existing_tables:
    print("   Existing tables:")
    for table in sorted(existing_tables):
        print(f"      - {table}")
else:
    print("   ❌ NO TABLES FOUND IN DATABASE!")

print("\n" + "=" * 70)
if existing_tables:
    print("✅ SUCCESS: All tables created!")
else:
    print("❌ FAILURE: No tables were created!")
print("=" * 70)

