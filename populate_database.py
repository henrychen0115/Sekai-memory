#!/usr/bin/env python3
"""
LangGraph Database Population Script
Uses LangGraph workflow for memory insertion with multiple nodes:
1. Extract atomic memories from synopsis
2. Validate memories against synopsis
3. Check for conflicts/redundancy with existing memories
4. Insert into database

Input Formats:
1. Chapter range: "1-20" (processes chapters 1 through 20)
2. Single chapter: "5" (processes only chapter 5)
3. Custom chapter: "chapter_number,synopsis" (processes custom chapter)
"""

import os
import json
import re
import time
from dotenv import load_dotenv

from memory_manager.langgraph_memory_processor import LangGraphMemoryProcessor

# Load environment variables
load_dotenv()

def load_story_data():
    """Load story data from memory_data.json"""
    try:
        with open('memory_data.json', 'r') as f:
            story_data = json.load(f)
        return story_data
    except Exception as e:
        print(f"Failed to load story data: {e}")
        return None

def parse_input(input_text):
    """
    Parse user input to determine processing mode
    
    Args:
        input_text: User input string
        
    Returns:
        dict: Processing configuration
    """
    input_text = input_text.strip()
    
    # Check for chapter range format (e.g., "1-20")
    range_match = re.match(r'^(\d+)-(\d+)$', input_text)
    if range_match:
        start_chapter = int(range_match.group(1))
        end_chapter = int(range_match.group(2))
        if start_chapter > end_chapter:
            print("Error: Start chapter must be less than or equal to end chapter")
            return None
        return {
            "mode": "range",
            "start_chapter": start_chapter,
            "end_chapter": end_chapter
        }
    
    # Check for single chapter format (e.g., "5")
    if input_text.isdigit():
        chapter_num = int(input_text)
        return {
            "mode": "single",
            "chapter_number": chapter_num
        }
    
    # Check for custom chapter format (e.g., "25,This is a custom synopsis")
    custom_match = re.match(r'^(\d+),(.+)$', input_text)
    if custom_match:
        chapter_num = int(custom_match.group(1))
        synopsis = custom_match.group(2).strip()
        return {
            "mode": "custom",
            "chapter_number": chapter_num,
            "synopsis": synopsis
        }
    
    print("Invalid input format. Please use one of these formats:")
    print("1. Chapter range: '1-20' (processes chapters 1 through 20)")
    print("2. Single chapter: '5' (processes only chapter 5)")
    print("3. Custom chapter: '25,This is a custom synopsis'")
    return None

def get_chapters_to_process(config, story_data):
    """
    Get list of chapters to process based on configuration
    
    Args:
        config: Processing configuration from parse_input()
        story_data: Loaded story data
        
    Returns:
        list: Chapters to process
    """
    if config["mode"] == "range":
        start_chapter = config["start_chapter"]
        end_chapter = config["end_chapter"]
        
        # Filter chapters within range
        chapters = [ch for ch in story_data if start_chapter <= ch['chapter_number'] <= end_chapter]
        
        if not chapters:
            print(f"No chapters found in range {start_chapter}-{end_chapter}")
            return []
        
        print(f"Found {len(chapters)} chapters in range {start_chapter}-{end_chapter}")
        return chapters
    
    elif config["mode"] == "single":
        chapter_num = config["chapter_number"]
        
        # Find specific chapter
        chapter = next((ch for ch in story_data if ch['chapter_number'] == chapter_num), None)
        
        if not chapter:
            print(f"Chapter {chapter_num} not found in story data")
            return []
        
        print(f"Found Chapter {chapter_num}")
        return [chapter]
    
    elif config["mode"] == "custom":
        chapter_num = config["chapter_number"]
        synopsis = config["synopsis"]
        
        # Create custom chapter data
        custom_chapter = {
            "chapter_number": chapter_num,
            "synopsis": synopsis
        }
        
        print(f"Processing custom Chapter {chapter_num}")
        return [custom_chapter]
    
    return []

def populate_database_langgraph(config):
    """
    Populate database using LangGraph memory processor
    
    Args:
        config: Processing configuration from parse_input()
        
    Returns:
        bool: Success status
    """
    # Initialize processing setup
    setup_result = _setup_langgraph_processing(config)
    if not setup_result:
        return False
    
    processor, chapters_to_process = setup_result
    
    # Display processing header
    _display_processing_header(chapters_to_process)
    
    # Process all chapters
    processing_stats = _process_all_chapters(processor, chapters_to_process)
    
    # Display final summary
    _display_final_summary(processing_stats)
    
    return _determine_success_status(processing_stats)


def _setup_langgraph_processing(config):
    """Setup LangGraph processing with data loading and validation"""
    # Load story data (for range and single modes)
    story_data = None
    if config["mode"] in ["range", "single"]:
        story_data = load_story_data()
        if not story_data:
            return False
    
    # Get chapters to process
    chapters_to_process = get_chapters_to_process(config, story_data)
    if not chapters_to_process:
        return False
    
    # Create LangGraph memory processor
    processor = LangGraphMemoryProcessor()
    
    return processor, chapters_to_process


def _display_processing_header(chapters_to_process):
    """Display the processing header with chapter count"""
    print(f"\n🚀 LangGraph Memory Processing")
    print(f"Processing {len(chapters_to_process)} chapters...")
    print("=" * 60)


def _process_all_chapters(processor, chapters_to_process):
    """Process all chapters and collect statistics"""
    total_start_time = time.time()
    
    processing_stats = {
        'total_extracted': 0,
        'total_validated': 0,
        'total_conflict_checked': 0,
        'total_inserted': 0,
        'total_errors': 0
    }
    
    for i, chapter_data in enumerate(chapters_to_process, 1):
        chapter_stats = _process_single_chapter(processor, chapter_data, i, len(chapters_to_process))
        
        # Accumulate statistics
        for key in processing_stats:
            processing_stats[key] += chapter_stats.get(key, 0)
    
    processing_stats['total_time'] = (time.time() - total_start_time) * 1000
    return processing_stats


def _process_single_chapter(processor, chapter_data, chapter_index, total_chapters):
    """Process a single chapter and return its statistics"""
    chapter_num = chapter_data['chapter_number']
    synopsis = chapter_data['synopsis']
    
    _display_chapter_header(chapter_num, chapter_index, total_chapters, synopsis)
    
    # Start timing individual chapter processing
    chapter_start_time = time.time()
    
    # Process chapter through LangGraph workflow
    results = processor.process_chapter(chapter_data)
    
    # Calculate chapter processing time
    chapter_time = (time.time() - chapter_start_time) * 1000
    
    # Display chapter results
    _display_chapter_results(chapter_num, results, chapter_time)
    
    # Return chapter statistics
    return {
        'total_extracted': results['extracted_count'],
        'total_validated': results['validated_count'],
        'total_conflict_checked': results['conflict_checked_count'],
        'total_inserted': results['inserted_count'],
        'total_errors': len(results['errors'])
    }


def _display_chapter_header(chapter_num, chapter_index, total_chapters, synopsis):
    """Display header information for a single chapter"""
    print(f"\n📖 Chapter {chapter_num} ({chapter_index}/{total_chapters})")
    print(f"📝 Synopsis: {synopsis[:100]}{'...' if len(synopsis) > 100 else ''}")
    print("-" * 40)


def _display_chapter_results(chapter_num, results, chapter_time):
    """Display detailed results for a single chapter"""
    print(f"\n📊 Chapter {chapter_num} Results:")
    print(f"   🔍 Extracted: {results['extracted_count']}")
    print(f"   ✅ Validated: {results['validated_count']}")
    print(f"   🔍 Conflict Checked: {results['conflict_checked_count']}")
    print(f"   💾 Inserted: {results['inserted_count']}")
    print(f"   ⏱️ Processing Time: {chapter_time:.2f}ms")
    
    if results['errors']:
        print(f"   ❌ Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"      - {error}")
    
    # Show processing log
    if results['processing_log']:
        print(f"   📋 Processing Log:")
        for log_entry in results['processing_log']:
            print(f"      - {log_entry}")


def _display_final_summary(processing_stats):
    """Display the final summary of all processing"""
    print(f"\n" + "=" * 60)
    print(f"🎉 LangGraph Database Population Completed!")
    print(f"📊 Final Statistics:")
    print(f"   🔍 Total Extracted: {processing_stats['total_extracted']}")
    print(f"   ✅ Total Validated: {processing_stats['total_validated']}")
    print(f"   🔍 Total Conflict Checked: {processing_stats['total_conflict_checked']}")
    print(f"   💾 Total Inserted: {processing_stats['total_inserted']}")
    print(f"   ❌ Total Errors: {processing_stats['total_errors']}")
    print(f"   ⏱️ Total Insertion Time: {processing_stats['total_time']:.2f}ms")


def _determine_success_status(processing_stats):
    """Determine if the processing was successful"""
    if processing_stats['total_inserted'] > 0:
        print(f"\n✅ Successfully inserted {processing_stats['total_inserted']} new memories into database")
        return True
    else:
        print(f"\n⚠️  No new memories were inserted (likely due to conflict checking)")
        return processing_stats['total_errors'] == 0  # Consider successful if no errors, even if no insertions

def main():
    """Main function to handle user interaction"""
    print("🗄️  LangGraph Database Population Script")
    print("=" * 50)
    print("Uses LangGraph workflow with 4 processing nodes:")
    print("1. 🔍 Extract atomic memories from synopsis")
    print("2. ✅ Validate memories against synopsis")
    print("3. 🔍 Check for conflicts/redundancy")
    print("4. 💾 Insert into database")
    print("=" * 50)
    print("Input Formats:")
    print("1. Chapter range: '1-20' (processes chapters 1 through 20)")
    print("2. Single chapter: '5' (processes only chapter 5)")
    print("3. Custom chapter: '25,This is a custom synopsis'")
    print("=" * 50)
    
    # Get input from user
    user_input = input("\nEnter your choice: ").strip()
    
    # Parse input
    config = parse_input(user_input)
    if not config:
        return False
    
    # Start population process
    success = populate_database_langgraph(config)
    
    if success:
        print("\n✅ LangGraph database population completed successfully")
    else:
        print("\n❌ LangGraph database population failed")
    
    return success

if __name__ == "__main__":
    main()
