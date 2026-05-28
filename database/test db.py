from sqlalchemy import text
from database.db import engine


def test_connection():
    try:
        # Create database connection
        with engine.connect() as connection:

            # Test query
            result = connection.execute(text("SELECT version();"))

            # Fetch PostgreSQL version
            db_version = result.fetchone()

            print("✅ PostgreSQL connection successful!")
            print(f"📦 Database Version: {db_version[0]}")

    except Exception as e:
        print("❌ Connection failed!")
        print("Error:", e)


if __name__ == "__main__":
    test_connection()