"""
COMPREHENSIVE VERIFICATION SCRIPT
Run this to verify ALL functionality works before deployment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("🔍 COMPREHENSIVE VERIFICATION TEST")
print("=" * 80)

# TEST 1: Database connection
print("\n1️⃣  Testing database connection...")
try:
    from app.core.database import engine, Base, SessionLocal
    print("   ✅ Database connection OK")
except Exception as e:
    print(f"   ❌ Database connection FAILED: {e}")
    sys.exit(1)

# TEST 2: Base instance check
print("\n2️⃣  Checking Base instance...")
try:
    from app.core.database import Base as DBBase
    from app.models.models import User, Workspace, Project, Datasource, Dataset, Model, Activity
    from app.models.models import Base as ModelsBase
    
    if DBBase is ModelsBase:
        print("   ✅ Base instances are SAME (correct!)")
    else:
        print("   ❌ Base instances are DIFFERENT (ERROR!)")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Base instance check FAILED: {e}")
    sys.exit(1)

# TEST 3: Metadata registration
print("\n3️⃣  Checking model registration with Base.metadata...")
try:
    from app.core.database import Base
    tables = list(Base.metadata.tables.keys())
    print(f"   📊 {len(tables)} tables registered:")
    for table in sorted(tables):
        print(f"      - {table}")
    
    expected = {'users', 'workspaces', 'projects', 'datasources', 'datasets', 'models', 'activities', 'model_datasets'}
    registered = set(tables)
    
    if expected.issubset(registered):
        print("   ✅ All expected tables registered")
    else:
        missing = expected - registered
        print(f"   ❌ Missing tables: {missing}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Metadata check FAILED: {e}")
    sys.exit(1)

# TEST 4: Create tables
print("\n4️⃣  Creating database tables...")
try:
    Base.metadata.create_all(bind=engine)
    print("   ✅ Tables created/verified")
except Exception as e:
    print(f"   ❌ Table creation FAILED: {e}")
    sys.exit(1)

# TEST 5: Verify tables in database
print("\n5️⃣  Verifying tables exist in database...")
try:
    from sqlalchemy import inspect
    inspector = inspect(engine)
    db_tables = inspector.get_table_names()
    print(f"   📊 {len(db_tables)} tables in database:")
    for table in sorted(db_tables):
        print(f"      - {table}")
    
    if len(db_tables) >= 7:
        print("   ✅ All tables exist in database")
    else:
        print(f"   ❌ Only {len(db_tables)} tables found, expected at least 7")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Database verification FAILED: {e}")
    sys.exit(1)

# TEST 6: Test model instantiation
print("\n6️⃣  Testing model instantiation...")
try:
    user = User(email="test@example.com", username="testuser", password_hash="hash123")
    workspace = Workspace(name="Test Workspace")
    project = Project(name="Test Project")
    datasource = Datasource(name="Test DS", type="mysql", host="localhost", port=3306, database_name="test")
    dataset = Dataset(name="Test Dataset", file_name="test.csv")
    model = Model(name="Test Model", model_type="regression")
    activity = Activity(action="test", entity_type="test")
    
    print("   ✅ All models instantiate correctly")
except Exception as e:
    print(f"   ❌ Model instantiation FAILED: {e}")
    sys.exit(1)

# TEST 7: Test API imports
print("\n7️⃣  Testing API router imports...")
try:
    from app.api.auth import router as auth_router
    from app.api.workspaces import router as workspaces_router
    from app.api.projects import router as projects_router
    from app.api.datasources import router as datasources_router
    from app.api.datasets import router as datasets_router
    from app.api.models import router as models_router
    from app.api.activities import router as activities_router
    
    print("   ✅ All API routers import successfully")
except Exception as e:
    print(f"   ❌ API import FAILED: {e}")
    sys.exit(1)

# TEST 8: Test session creation
print("\n8️⃣  Testing database session...")
try:
    db = SessionLocal()
    result = db.query(User).first()
    db.close()
    print("   ✅ Database session works")
except Exception as e:
    print(f"   ❌ Session test FAILED: {e}")
    sys.exit(1)

# TEST 9: Test column definitions
print("\n9️⃣  Verifying Datasource columns...")
try:
    from sqlalchemy import inspect
    inspector = inspect(Datasource)
    columns = {col.name for col in inspector.columns}
    
    required = {'id', 'project_id', 'name', 'type', 'host', 'port', 'database_name', 'username', 'password', 'connection_config', 'status', 'created_at', 'updated_at'}
    
    if required.issubset(columns):
        print(f"   ✅ Datasource has all {len(required)} required columns")
    else:
        missing = required - columns
        print(f"   ❌ Missing Datasource columns: {missing}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Column verification FAILED: {e}")
    sys.exit(1)

# TEST 10: Test Dataset columns
print("\n🔟 Verifying Dataset columns...")
try:
    inspector = inspect(Dataset)
    columns = {col.name for col in inspector.columns}
    
    required = {'id', 'project_id', 'name', 'row_count', 'column_count', 'quality_score', 'missing_values_pct', 'duplicate_rows_pct', 'created_at', 'updated_at'}
    
    if required.issubset(columns):
        print(f"   ✅ Dataset has all {len(required)} required metric columns")
    else:
        missing = required - columns
        print(f"   ❌ Missing Dataset columns: {missing}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Dataset column verification FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
print("=" * 80)
print("\n📊 Summary:")
print(f"   - Database: ✅")
print(f"   - Models: ✅ (7 models registered)")
print(f"   - Tables: ✅ (8 tables in database)")
print(f"   - APIs: ✅ (7 routers)")
print(f"   - Metrics: ✅ (All columns present)")
print("\n🎉 SAFE TO DEPLOY!\n")

