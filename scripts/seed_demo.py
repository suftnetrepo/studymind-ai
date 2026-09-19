"""
Demo seed script — creates a sample admin, lecturer, student,
institution, module, and semester so you can test the API immediately.

Usage:
    python scripts/seed_demo.py

Then use the printed credentials to log in via /docs.
"""
import asyncio, sys, os, uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    from app.db.engine import AsyncSessionLocal, create_all_tables
    from app.db.models import Institution, User, Semester, Module, Department, InstitutionCode
    from app.auth.security import hash_password

    await create_all_tables()

    async with AsyncSessionLocal() as db:
        # Institution
        inst = Institution(
            id=uuid.uuid4(), name="Demo University",
            domain="demo.ac.uk", tier="institution",
        )
        db.add(inst)
        await db.flush()

        # Department
        dept = Department(
            id=uuid.uuid4(), institution_id=inst.id,
            name="Computer Science", code="CSC",
        )
        db.add(dept)
        await db.flush()

        # Admin
        admin = User(
            id=uuid.uuid4(), email="admin@demo.ac.uk",
            password_hash=hash_password("Admin1234"),
            full_name="System Admin", role="admin",
            institution_id=inst.id, is_active=True, is_verified=True,
        )
        db.add(admin)

        # Lecturer
        lecturer = User(
            id=uuid.uuid4(), email="lecturer@demo.ac.uk",
            password_hash=hash_password("Lecturer1234"),
            full_name="Dr Jane Smith", role="lecturer",
            institution_id=inst.id, department_id=dept.id,
            is_active=True, is_verified=True,
        )
        db.add(lecturer)

        # Student
        student = User(
            id=uuid.uuid4(), email="student@demo.ac.uk",
            password_hash=hash_password("Student1234"),
            full_name="John Doe", role="student",
            institution_id=inst.id, is_active=True, is_verified=True,
        )
        db.add(student)

        # Self-learner
        selflearner = User(
            id=uuid.uuid4(), email="learner@example.com",
            password_hash=hash_password("Learner1234"),
            full_name="Alice Johnson", role="self_learner",
            is_active=True, is_verified=True,
        )
        db.add(selflearner)
        await db.flush()

        # Semester
        semester = Semester(
            id=uuid.uuid4(), institution_id=inst.id,
            label="2025 Semester 1", is_current=True,
        )
        db.add(semester)
        await db.flush()

        # Module
        module = Module(
            id=uuid.uuid4(), course_code="CSC109",
            title="Introduction to Programming",
            owner_id=lecturer.id, institution_id=inst.id,
            department_id=dept.id, semester_id=semester.id,
            access_type="class", status="active",
        )
        db.add(module)
        await db.flush()

        # Institution join code — students use this to join the institution
        join_code = InstitutionCode(
            id=uuid.uuid4(), institution_id=inst.id,
            code="DEMO2025", code_type="institution_join",
            target_role="student", max_uses=100, use_count=0,
            created_by=admin.id, is_active=True,
        )
        db.add(join_code)

        # Module enrolment code — students use this to enrol in CSC109
        enrol_code = InstitutionCode(
            id=uuid.uuid4(), institution_id=inst.id,
            code="CSC109", code_type="module_enrolment",
            target_role="student", max_uses=100, use_count=0,
            created_by=lecturer.id, is_active=True,
            metadata_={"module_id": str(module.id)},
        )
        db.add(enrol_code)

        await db.commit()

        print("\n✅ Demo data seeded!\n")
        print("─" * 50)
        print("CREDENTIALS")
        print("─" * 50)
        print(f"Admin:        admin@demo.ac.uk       / Admin1234")
        print(f"Lecturer:     lecturer@demo.ac.uk    / Lecturer1234")
        print(f"Student:      student@demo.ac.uk     / Student1234")
        print(f"Self-learner: learner@example.com    / Learner1234")
        print("─" * 50)
        print(f"\nInstitution ID: {inst.id}")
        print(f"Module ID:      {module.id}  (CSC109)")
        print(f"Semester ID:    {semester.id}")
        print(f"\nInstitution join code: DEMO2025")
        print(f"Module enrolment code: CSC109")
        print(f"\nOpen Swagger UI: http://localhost:8000/docs")
        print("1. POST /api/auth/login with any credentials above")
        print("2. Copy access_token → click Authorize → paste token")
        print("3. Explore all endpoints\n")

if __name__ == "__main__":
    asyncio.run(main())
