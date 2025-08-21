#!/usr/bin/env python3
"""
Database Clear Script
Clears all contents from the memories database
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clear_database():
    """Clear all contents from the database"""
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector
        
        # Connect to database
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        register_vector(conn)
        cursor = conn.cursor()
        
        # Get count before deletion
        cursor.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]
        
        if memory_count == 0:
            print("Database is already empty")
            cursor.close()
            conn.close()
            return True
        
        # Delete all memories
        cursor.execute("DELETE FROM memories")
        deleted_count = cursor.rowcount
        
        # Commit the transaction
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Cleared {deleted_count} memories from database")
        return True
        
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False

if __name__ == "__main__":
    success = clear_database()
    if success:
        print("Database clear operation completed successfully")
    else:
        print("Database clear operation failed")
