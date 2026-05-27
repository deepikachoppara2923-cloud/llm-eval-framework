import sqlite3

# Connect database
conn = sqlite3.connect("database/llm_logs.db")

cursor = conn.cursor()

# Fetch all records
cursor.execute("SELECT * FROM evaluations")

rows = cursor.fetchall()

# Print records
for row in rows:
    print(row)
    print("\n" + "="*80 + "\n")

conn.close()