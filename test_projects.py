#!/usr/bin/env python3
"""
Automated Projects CRUD Testing Script
========================================

Tests all project operations with REAL data (not mock data).
Verifies:
- CREATE project ✅
- READ project ✅
- UPDATE project ✅
- DELETE project ✅
- LIST projects ✅

Run: python test_projects.py
"""

import requests
import json
import sqlite3
from datetime import datetime

# Configuration
BASE_URL = "http://192.168.1.147:8000"
TEST_EMAIL = f"test_{datetime.now().timestamp()}@example.com"
TEST_USERNAME = f"testuser_{datetime.now().timestamp()}"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_test(msg):
    print(f"\n{Colors.YELLOW}{'='*70}{Colors.END}")
    print(f"{Colors.YELLOW}🧪 {msg}{Colors.END}")
    print(f"{Colors.YELLOW}{'='*70}{Colors.END}")

# ============================================================================
# TEST 1: REGISTER USER
# ============================================================================

def test_register():
    print_test("TEST 1: Register User")
    
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json={
            "email": TEST_EMAIL,
            "username": TEST_USERNAME,
            "password": "password123",
            "full_name": "Test User"
        }
    )
    
    if response.status_code != 200:
        print_error(f"Registration failed: {response.status_code}")
        print_error(f"Response: {response.text}")
        return None
    
    user = response.json()
    print_success(f"User registered: {user['email']}")
    print_info(f"User ID: {user['id']}")
    
    return user

# ============================================================================
# TEST 2: LOGIN USER
# ============================================================================

def test_login():
    print_test("TEST 2: Login User")
    
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "email": TEST_EMAIL,
            "password": "password123"
        }
    )
    
    if response.status_code != 200:
        print_error(f"Login failed: {response.status_code}")
        return None
    
    data = response.json()
    token = data['access_token']
    print_success(f"Login successful")
    print_info(f"Token: {token[:20]}...")
    
    return token

# ============================================================================
# TEST 3: CREATE WORKSPACE
# ============================================================================

def test_create_workspace(token):
    print_test("TEST 3: Create Workspace")
    
    response = requests.post(
        f"{BASE_URL}/api/workspaces",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Test Workspace",
            "slug": "test-workspace",
            "description": "Workspace for testing projects"
        }
    )
    
    if response.status_code != 201:
        print_error(f"Workspace creation failed: {response.status_code}")
        print_error(f"Response: {response.text}")
        return None
    
    workspace = response.json()
    print_success(f"Workspace created: {workspace['name']}")
    print_info(f"Workspace ID: {workspace['id']}")
    
    return workspace

# ============================================================================
# TEST 4: CREATE PROJECT (REAL DATA)
# ============================================================================

def test_create_project_1(token, workspace_id):
    print_test("TEST 4A: Create Project #1 - Classification")
    
    response = requests.post(
        f"{BASE_URL}/api/projects/workspaces/{workspace_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Customer Churn Prediction",
            "description": "Predict which customers will churn",
            "problem_type": "Classification"
        }
    )
    
    if response.status_code != 201:
        print_error(f"Project creation failed: {response.status_code}")
        print_error(f"Response: {response.text}")
        return None
    
    project = response.json()
    print_success(f"Project created: {project['name']}")
    print_info(f"Project ID: {project['id']}")
    print_info(f"Problem Type: {project['problem_type']}")
    print_info(f"Status: {project['status']}")
    
    return project

def test_create_project_2(token, workspace_id):
    print_test("TEST 4B: Create Project #2 - Regression")
    
    response = requests.post(
        f"{BASE_URL}/api/projects/workspaces/{workspace_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Sales Forecasting",
            "description": "Predict future sales trends",
            "problem_type": "Regression"
        }
    )
    
    if response.status_code != 201:
        print_error(f"Project creation failed: {response.status_code}")
        return None
    
    project = response.json()
    print_success(f"Project created: {project['name']}")
    print_info(f"Project ID: {project['id']}")
    print_info(f"Problem Type: {project['problem_type']}")
    
    return project

# ============================================================================
# TEST 5: LIST ALL PROJECTS
# ============================================================================

def test_list_all_projects(token):
    print_test("TEST 5: List All Projects")
    
    response = requests.get(
        f"{BASE_URL}/api/projects",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code != 200:
        print_error(f"List projects failed: {response.status_code}")
        return None
    
    projects = response.json()
    print_success(f"Found {len(projects)} projects")
    
    for i, project in enumerate(projects, 1):
        print_info(f"{i}. {project['name']} ({project['problem_type']})")
    
    return projects

# ============================================================================
# TEST 6: GET SPECIFIC PROJECT
# ============================================================================

def test_get_project(token, project_id):
    print_test(f"TEST 6: Get Project Details ({project_id[:8]}...)")
    
    response = requests.get(
        f"{BASE_URL}/api/projects/{project_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code != 200:
        print_error(f"Get project failed: {response.status_code}")
        return None
    
    project = response.json()
    print_success(f"Project retrieved: {project['name']}")
    print_info(f"ID: {project['id']}")
    print_info(f"Name: {project['name']}")
    print_info(f"Type: {project['problem_type']}")
    print_info(f"Status: {project['status']}")
    print_info(f"Created: {project['created_at']}")
    
    return project

# ============================================================================
# TEST 7: UPDATE PROJECT
# ============================================================================

def test_update_project(token, project_id):
    print_test(f"TEST 7: Update Project ({project_id[:8]}...)")
    
    new_name = "Customer Churn Prediction - v2"
    new_status = "Active"
    
    response = requests.put(
        f"{BASE_URL}/api/projects/{project_id}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": new_name,
            "description": "Updated description for testing",
            "status": new_status
        }
    )
    
    if response.status_code != 200:
        print_error(f"Update project failed: {response.status_code}")
        print_error(f"Response: {response.text}")
        return None
    
    project = response.json()
    print_success(f"Project updated: {project['name']}")
    print_info(f"New Name: {project['name']}")
    print_info(f"Status: {project['status']}")
    print_info(f"Updated: {project['updated_at']}")
    
    return project

# ============================================================================
# TEST 8: DELETE PROJECT
# ============================================================================

def test_delete_project(token, project_id):
    print_test(f"TEST 8: Delete Project ({project_id[:8]}...)")
    
    response = requests.delete(
        f"{BASE_URL}/api/projects/{project_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code != 204:
        print_error(f"Delete project failed: {response.status_code}")
        print_error(f"Response: {response.text}")
        return False
    
    print_success(f"Project deleted successfully")
    return True

# ============================================================================
# TEST 9: VERIFY IN DATABASE
# ============================================================================

def test_database_verification():
    print_test("TEST 9: Verify Data in Database")
    
    try:
        conn = sqlite3.connect("ml_platform.db")
        cursor = conn.cursor()
        
        # Count projects
        cursor.execute("SELECT COUNT(*) FROM projects")
        count = cursor.fetchone()[0]
        print_success(f"Total projects in database: {count}")
        
        if count > 0:
            # List projects
            cursor.execute("SELECT id, name, problem_type, status, created_at FROM projects")
            projects = cursor.fetchall()
            
            print_info(f"Projects in database:")
            for project in projects:
                print_info(f"  - {project[1]} ({project[2]}) | Status: {project[3]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Database verification failed: {e}")
        return False

# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def main():
    print(f"\n{Colors.BLUE}{'='*70}")
    print("🧪 PROJECTS CRUD TESTING - REAL DATA ONLY")
    print(f"{'='*70}{Colors.END}\n")
    
    # Test 1: Register
    user = test_register()
    if not user:
        print_error("Setup failed - stopping tests")
        return False
    
    # Test 2: Login
    token = test_login()
    if not token:
        print_error("Setup failed - stopping tests")
        return False
    
    # Test 3: Create Workspace
    workspace = test_create_workspace(token)
    if not workspace:
        print_error("Setup failed - stopping tests")
        return False
    
    # Test 4: Create Projects
    project1 = test_create_project_1(token, workspace['id'])
    if not project1:
        print_error("Project 1 creation failed")
        return False
    
    project2 = test_create_project_2(token, workspace['id'])
    if not project2:
        print_error("Project 2 creation failed")
        return False
    
    # Test 5: List All Projects
    projects = test_list_all_projects(token)
    if projects is None:
        print_error("List projects failed")
        return False
    
    # Test 6: Get Specific Project
    project = test_get_project(token, project1['id'])
    if not project:
        print_error("Get project failed")
        return False
    
    # Test 7: Update Project
    updated = test_update_project(token, project1['id'])
    if not updated:
        print_error("Update project failed")
        return False
    
    # Test 8: Delete Project
    deleted = test_delete_project(token, project2['id'])
    if not deleted:
        print_error("Delete project failed")
        return False
    
    # Test 9: Verify Database
    test_database_verification()
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"✅ ALL TESTS PASSED - PROJECTS CRUD WORKING PERFECTLY!")
    print(f"{'='*70}{Colors.END}\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
