"""
seed_data.py — generates realistic fake employee profiles for development.
No real resumes or GitHub accounts needed. Run once to populate the DB.
"""
from faker import Faker
from datetime import datetime, timezone, timedelta
import random
import uuid

fake = Faker()

PERSONA_TEMPLATES = [
    {
        "title": "Senior Backend Engineer",
        "seniority": "senior",
        "department": "Engineering",
        "skills": [
            {"name": "Python", "type": "technical", "proficiency": "expert", "is_hands_on": True, "last_used_year": 2024},
            {"name": "FastAPI", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "PostgreSQL", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "Docker", "type": "tool", "proficiency": "intermediate", "is_hands_on": True, "last_used_year": 2023},
        ],
        "github_stats": {"total_commits": 520, "top_languages": ["Python", "SQL"], "active_repos": 8},
        "resume_text": "Senior backend engineer with 6 years experience building REST APIs with FastAPI and Python. Led migration of monolith to microservices. Deep expertise in PostgreSQL query optimization and async programming.",
    },
    {
        "title": "ML Engineer",
        "seniority": "senior",
        "department": "AI",
        "skills": [
            {"name": "Python", "type": "technical", "proficiency": "expert", "is_hands_on": True, "last_used_year": 2024},
            {"name": "LangChain", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "RAG", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "PyTorch", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "FAISS", "type": "technical", "proficiency": "intermediate", "is_hands_on": True, "last_used_year": 2024},
        ],
        "github_stats": {"total_commits": 380, "top_languages": ["Python", "Jupyter"], "active_repos": 6},
        "resume_text": "ML Engineer specializing in LLM applications and RAG pipelines. Built production retrieval systems using FAISS and Pinecone. Hands-on experience fine-tuning and evaluating LLMs. LangChain contributor.",
    },
    {
        "title": "Audio DSP Engineer",
        "seniority": "senior",
        "department": "Hardware",
        "is_sme": True,
        "sme_domains": ["audio_dsp", "embedded_systems"],
        "skills": [
            {"name": "DSP", "type": "domain", "proficiency": "expert", "is_hands_on": True, "last_used_year": 2024},
            {"name": "IIR Filters", "type": "technical", "proficiency": "expert", "is_hands_on": True, "last_used_year": 2024},
            {"name": "C++", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "MATLAB", "type": "tool", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2023},
            {"name": "Embedded Systems", "type": "domain", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
        ],
        "github_stats": {"total_commits": 210, "top_languages": ["C++", "Python", "MATLAB"], "active_repos": 4},
        "resume_text": "Audio DSP Engineer with 8 years experience designing digital filters for speaker systems. Expert in IIR/FIR filter design, frequency response analysis, and embedded audio processing on ARM Cortex-M platforms.",
    },
    {
        "title": "UX Designer",
        "seniority": "mid",
        "department": "Design",
        "is_sme": True,
        "sme_domains": ["ux_design"],
        "skills": [
            {"name": "Figma", "type": "tool", "proficiency": "expert", "is_hands_on": True, "last_used_year": 2024},
            {"name": "User Research", "type": "domain", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "Prototyping", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "Design Systems", "type": "technical", "proficiency": "intermediate", "is_hands_on": True, "last_used_year": 2023},
        ],
        "github_stats": {"total_commits": 45, "top_languages": ["CSS", "JavaScript"], "active_repos": 2},
        "resume_text": "UX Designer with 4 years experience designing B2B SaaS products. Conducted 50+ user interviews. Built and maintained design systems in Figma. Strong focus on accessibility and developer handoff.",
    },
    {
        "title": "Frontend Engineer",
        "seniority": "mid",
        "department": "Engineering",
        "skills": [
            {"name": "React", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "TypeScript", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "Tailwind CSS", "type": "tool", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "REST APIs", "type": "technical", "proficiency": "intermediate", "is_hands_on": True, "last_used_year": 2024},
        ],
        "github_stats": {"total_commits": 290, "top_languages": ["TypeScript", "JavaScript"], "active_repos": 5},
        "resume_text": "Frontend engineer building React + TypeScript applications. Shipped 3 production SaaS products. Strong focus on performance optimization and component architecture.",
    },
    {
        "title": "DevOps Engineer",
        "seniority": "senior",
        "department": "Infrastructure",
        "skills": [
            {"name": "Kubernetes", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "Docker", "type": "technical", "proficiency": "expert", "is_hands_on": True, "last_used_year": 2024},
            {"name": "CI/CD", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "AWS", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
        ],
        "github_stats": {"total_commits": 180, "top_languages": ["YAML", "Bash", "Python"], "active_repos": 6},
        "resume_text": "DevOps engineer with 5 years experience running production Kubernetes clusters on AWS. Built zero-downtime deployment pipelines. Reduced infrastructure costs 30% through autoscaling.",
    },
    {
        "title": "Data Engineer",
        "seniority": "mid",
        "department": "Data",
        "skills": [
            {"name": "Python", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "Apache Spark", "type": "technical", "proficiency": "intermediate", "is_hands_on": True, "last_used_year": 2023},
            {"name": "dbt", "type": "tool", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
            {"name": "Airflow", "type": "tool", "proficiency": "intermediate", "is_hands_on": True, "last_used_year": 2024},
        ],
        "github_stats": {"total_commits": 220, "top_languages": ["Python", "SQL"], "active_repos": 4},
        "resume_text": "Data engineer building ELT pipelines with dbt and Airflow. Experience processing 10TB+ datasets with Spark. Strong SQL and data modeling skills.",
    },
    {
        "title": "Security Engineer",
        "seniority": "senior",
        "department": "Security",
        "is_sme": True,
        "sme_domains": ["security", "compliance"],
        "skills": [
            {"name": "Penetration Testing", "type": "domain", "proficiency": "expert", "is_hands_on": True, "last_used_year": 2024},
            {"name": "OWASP", "type": "domain", "proficiency": "expert", "is_hands_on": True, "last_used_year": 2024},
            {"name": "Python", "type": "technical", "proficiency": "advanced", "is_hands_on": True, "last_used_year": 2024},
        ],
        "github_stats": {"total_commits": 95, "top_languages": ["Python", "Bash"], "active_repos": 3},
        "resume_text": "Security engineer specializing in application security and penetration testing. OSCP certified. Conducted 20+ security audits for fintech and healthtech clients.",
    },
]


def generate_employees(count_per_persona: int = 3) -> list[dict]:
    """Generate employee dicts ready for DB insertion."""
    employees = []
    for persona in PERSONA_TEMPLATES:
        for _ in range(count_per_persona):
            emp_id = str(uuid.uuid4())
            employees.append({
                "id": emp_id,
                "email": fake.unique.email(),
                "full_name": fake.name(),
                "title": persona["title"],
                "department": persona.get("department"),
                "seniority_level": persona["seniority"],
                "employment_type": "full_time",
                "resume_text": persona["resume_text"],
                "github_username": fake.user_name(),
                "github_stats": persona["github_stats"],
                "is_active": True,
                "is_sme": persona.get("is_sme", False),
                "sme_domains": persona.get("sme_domains", []),
                "skills": persona["skills"],
                "availability": {
                    "available_percentage": random.choice([1.0, 0.8, 0.6, 0.4, 0.0]),
                    "status": random.choice(["available", "partially_available", "busy"]),
                    "free_from_date": (
                        datetime.now(timezone.utc) + timedelta(days=random.randint(7, 60))
                        if random.random() > 0.5 else None
                    ),
                },
            })
    return employees
