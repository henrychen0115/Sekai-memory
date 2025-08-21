#!/usr/bin/env python3
"""
Database Statistics Script for Sekai Memory System

This script provides comprehensive statistics about the memory database including:
- Number of distinct characters
- Memory counts by character
- Memory type distribution
- Chapter distribution
- Access count statistics
- Salience distribution
"""

import os
import sys
from dotenv import load_dotenv
from memory_manager import MemoryManager

def get_database_statistics():
    """
    Get comprehensive database statistics
    
    Returns:
        dict: Dictionary containing all database statistics
    """
    try:
        memory_manager = MemoryManager()
        
        # Get basic stats from memory manager
        basic_stats = memory_manager.get_database_stats()
        
        # Get additional detailed statistics
        detailed_stats = get_detailed_statistics(memory_manager)
        
        # Combine all statistics
        all_stats = {
            "basic_stats": basic_stats,
            "detailed_stats": detailed_stats
        }
        
        return all_stats
        
    except Exception as e:
        print(f"Error getting database statistics: {e}")
        return None

def get_detailed_statistics(memory_manager):
    """
    Get detailed statistics beyond the basic memory manager stats
    
    Args:
        memory_manager: MemoryManager instance
        
    Returns:
        dict: Detailed statistics
    """
    try:
        conn = memory_manager._get_db_connection()
        cursor = conn.cursor()
        
        # Get distinct characters count
        cursor.execute("""
            SELECT COUNT(DISTINCT source_char) as distinct_characters
            FROM memories 
            WHERE source_char IS NOT NULL
        """)
        distinct_characters = cursor.fetchone()[0]
        
        # Get character list
        cursor.execute("""
            SELECT DISTINCT source_char 
            FROM memories 
            WHERE source_char IS NOT NULL 
            ORDER BY source_char
        """)
        character_list = [row[0] for row in cursor.fetchall()]
        
        # Get memory counts by character
        cursor.execute("""
            SELECT source_char, COUNT(*) as memory_count
            FROM memories 
            WHERE source_char IS NOT NULL
            GROUP BY source_char 
            ORDER BY memory_count DESC
        """)
        character_counts = dict(cursor.fetchall())
        
        # Get memory type distribution
        cursor.execute("""
            SELECT memory_type, COUNT(*) as type_count
            FROM memories 
            WHERE memory_type IS NOT NULL
            GROUP BY memory_type 
            ORDER BY type_count DESC
        """)
        type_distribution = dict(cursor.fetchall())
        

        
        cursor.close()
        conn.close()
        
        return {
            "distinct_characters": distinct_characters,
            "character_list": character_list,
            "character_counts": character_counts,
            "type_distribution": type_distribution
        }
        
    except Exception as e:
        print(f"Error getting detailed statistics: {e}")
        return None

def print_statistics(stats):
    """
    Print formatted database statistics
    
    Args:
        stats: Dictionary containing database statistics
    """
    if not stats:
        print("No statistics available")
        return
    
    basic_stats = stats.get("basic_stats", {})
    detailed_stats = stats.get("detailed_stats", {})
    
    if not detailed_stats:
        print("Detailed statistics not available, showing basic stats only")
        detailed_stats = {}
    
    print("SEKAI MEMORY SYSTEM - DATABASE STATISTICS")
    print("=" * 60)
    
    # Basic Statistics
    print("\nBASIC STATISTICS")
    print("-" * 30)
    print(f"Total Memories: {basic_stats.get('total_memories', 0)}")
    print(f"Distinct Characters: {detailed_stats.get('distinct_characters', 0)}")
    
    # Character Information
    print(f"\nCHARACTERS ({detailed_stats.get('distinct_characters', 0)} total)")
    print("-" * 30)
    character_list = detailed_stats.get('character_list', [])
    if character_list:
        print(f"Character List: {', '.join(character_list)}")
    
    # Character Memory Counts
    character_counts = detailed_stats.get('character_counts', {})
    if character_counts:
        print("\nMEMORIES PER CHARACTER")
        print("-" * 30)
        for character, count in character_counts.items():
            print(f"{character}: {count} memories")
    
    # Memory Type Distribution
    type_distribution = detailed_stats.get('type_distribution', {})
    if type_distribution:
        print(f"\nMEMORY TYPE DISTRIBUTION")
        print("-" * 30)
        for memory_type, count in type_distribution.items():
            print(f"{memory_type}: {count} memories")

def main():
    """Main function to run database statistics"""
    # Load environment variables
    load_dotenv()
    
    print("Analyzing Sekai Memory System Database...")
    print("=" * 60)
    
    # Get statistics
    stats = get_database_statistics()
    
    if stats:
        print_statistics(stats)
        print("Database statistics completed successfully!")
    else:
        print("Failed to retrieve database statistics")
        sys.exit(1)

if __name__ == "__main__":
    main()
