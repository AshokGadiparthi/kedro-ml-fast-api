"""
Complete Database Setup
Run: python setup_database.py
"""

import os
import sys

def setup_database():
    print("\n" + "="*70)
    print("🔧 COMPLETE DATABASE SETUP")
    print("="*70)

    # Step 1: Delete old database
    db_file = "ml_platform.db"
    if os.path.exists(db_file):
        print(f"\n🗑️  Deleting old database: {db_file}")
        os.remove(db_file)
        print(f"✅ Deleted!")
    else:
        print(f"\n📝 No old database found")

    # Step 2: Initialize database
    print(f"\n🔨 Creating new database...")
    try:
        # Import and initialize
        from app.core.database import init_db, Base, engine
        from app.models.models import User, Workspace, Project, Datasource, Dataset

        # Create tables
        print("   Creating tables...")
        Base.metadata.create_all(bind=engine)

        print("✅ Database created!")

        # Step 3: Verify
        print(f"\n✅ Verifying tables...")
        import sqlite3
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()

        if tables:
            print(f"✅ Found {len(tables)} tables:")
            for table in tables:
                print(f"   ✅ {table[0]}")
        else:
            print(f"❌ No tables found!")
            return False

        print("\n" + "="*70)
        print("✅ DATABASE SETUP COMPLETE!")
        print("="*70)
        print("\n🚀 Next steps:")
        print("   1. Run: python main.py")
        print("   2. Visit: http://192.168.1.147:8000/docs")
        print("   3. Register a user!")
        print("\n" + "="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70 + "\n")
        return False

if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)