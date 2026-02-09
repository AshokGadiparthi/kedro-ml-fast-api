#!/usr/bin/env python3
"""
Database Verification Script
Checks that all 20 tables are created correctly
"""

import sqlite3
from pathlib import Path

DB_PATH = Path.cwd() / "ml_platform.db"

def verify_database():
    """Verify all tables exist"""
    
    if not DB_PATH.exists():
        print("❌ Database file not found!")
        print(f"   Expected at: {DB_PATH}")
        return False
    
    print(f"✅ Database file found: {DB_PATH}")
    print()
    
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'activities', 'correlations', 'datasources', 'datasets',
                'distributions', 'eda_jobs', 'eda_results', 'feature_clustering',
                'jobs', 'models', 'multicollinearity_warnings', 'normality_tests',
                'outliers', 'pipelines', 'projects', 'quality_metrics',
                'session_cache', 'statistics', 'users', 'vif_analysis'
            ]
            
            print(f"Found {len(tables)} tables:")
            print("=" * 60)
            
            for i, table in enumerate(sorted(tables), 1):
                status = "✓" if table in expected_tables else "⚠"
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                cursor.execute(f"PRAGMA table_info({table})")
                columns = len(cursor.fetchall())
                print(f"{status} {i:2d}. {table:30s} ({columns:2d} cols, {count:4d} rows)")
            
            print("=" * 60)
            print()
            
            # Check for missing tables
            missing = set(expected_tables) - set(tables)
            if missing:
                print(f"❌ Missing tables: {missing}")
                return False
            
            extra = set(tables) - set(expected_tables)
            if extra:
                print(f"⚠️  Extra tables: {extra}")
            
            print(f"✅ All {len(expected_tables)} expected tables found!")
            print()
            
            # Get database stats
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM projects")
            project_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM datasets")
            dataset_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM jobs")
            job_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM eda_jobs")
            eda_count = cursor.fetchone()[0]
            
            print("Database Statistics:")
            print("-" * 60)
            print(f"  Users:       {user_count}")
            print(f"  Projects:    {project_count}")
            print(f"  Datasets:    {dataset_count}")
            print(f"  Jobs:        {job_count}")
            print(f"  EDA Jobs:    {eda_count}")
            print("-" * 60)
            print()
            
            return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Database Verification")
    print("=" * 60)
    print()
    
    if verify_database():
        print("✅ Database verification PASSED!")
        print()
        print("Next steps:")
        print("1. Start FastAPI: python main.py")
        print("2. Test endpoints: curl http://localhost:8000/health")
        print("3. Create data: Use API endpoints")
        print("4. Verify data: sqlite3 ml_platform.db 'SELECT * FROM ...'")
    else:
        print("❌ Database verification FAILED!")
        print()
        print("Solutions:")
        print("1. Make sure FastAPI has started (python main.py)")
        print("2. Wait a moment for database initialization")
        print("3. Check application logs for errors")
        print("4. Delete ml_platform.db and restart")
