import sqlite3

# Connect database
conn = sqlite3.connect("database/llm_logs.db")

# Cursor
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    prompt_version TEXT,
    prompt TEXT,
    latency REAL,
    quality_score INTEGER,
    estimated_cost REAL,
    hallucination_score INTEGER,
    consistency_score REAL,
    approval_status TEXT,
    response TEXT
)
""")

conn.commit()

print("Database & table created successfully!")

conn.close()