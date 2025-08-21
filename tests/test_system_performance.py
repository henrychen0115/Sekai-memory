#!/usr/bin/env python3
"""
System Performance & Efficiency Evaluation (C2)
Tests: Retrieval Speed and Insertion Time with detailed step logging
"""

import os
import time
import statistics
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from memory_manager.memory_manager import MemoryManager

# Load environment variables
load_dotenv()

# Initialize models
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

def test_retrieval_speed():
    """Test 1: Retrieval Speed - Response time for memory queries"""
    print_section_header("RETRIEVAL SPEED EVALUATION")
    
    # Test queries for different complexity levels
    test_queries = [
        {"query": "first day", "complexity": "simple"},
        {"query": "office strategy meeting", "complexity": "medium"},
        {"query": "emotional conflict resolution between characters", "complexity": "complex"},
        {"query": "detailed analysis of character motivations and relationships", "complexity": "very_complex"}
    ]
    
    characters = ["Byleth", "Dimitri", "Sylvain", "Felix"]
    
    results = {
        "simple": [],
        "medium": [],
        "complex": [],
        "very_complex": []
    }
    
    print("Testing retrieval speed with different query complexities...")
    
    for test_case in test_queries:
        query = test_case["query"]
        complexity = test_case["complexity"]
        
        print(f"\nTesting {complexity} query: '{query}'")
        
        for character in characters:
            # Test only once per character per complexity
            start_time = time.time()
            
            try:
                # Use MemoryManager for retrieval (includes LLM agent scoring)
                memory_manager = MemoryManager()
                memories = memory_manager.retrieve_memories(character, query, limit=5)
                
                end_time = time.time()
                query_time = (end_time - start_time) * 1000  # Convert to milliseconds
                results[complexity].append(query_time)
                
                # Log memory details for analysis
                if memories:
                    llm_scores = [f"{m.get('llm_score', 0):.2f}" for m in memories]
                    print(f"   {character}: {query_time:.2f}ms - Retrieved {len(memories)} memories with LLM scores: {llm_scores}")
                else:
                    print(f"   {character}: {query_time:.2f}ms - No memories found")
                
            except Exception as e:
                print(f"   Error testing {character}: {e}")
                continue
    
    # Calculate and display results
    print(f"\nRetrieval Speed Results:")
    for complexity, times in results.items():
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            print(f"   {complexity.capitalize()} queries: {avg_time:.2f}ms avg ({min_time:.2f}ms - {max_time:.2f}ms)")
    
    return results



def test_insertion_time():
    """Test 2: Insertion Time - Detailed step-by-step timing of LangGraph workflow"""
    print_section_header("INSERTION TIME EVALUATION")
    
    # Test chapters for insertion
    test_chapters = [
        {
            "chapter_number": 1001,
            "synopsis": "Byleth and Dimitri have a brief conversation in the hallway about upcoming projects."
        },
        {
            "chapter_number": 1002,
            "synopsis": "Sylvain discovers a new coffee shop near the office and shares the information with Annette."
        },
        {
            "chapter_number": 1003,
            "synopsis": "Felix observes the team dynamics during a morning meeting and takes mental notes."
        }
    ]
    
    print("Testing insertion time with detailed step logging...")
    
    total_results = []
    
    for i, chapter_data in enumerate(test_chapters, 1):
        print(f"\nTest Insertion {i}: Chapter {chapter_data['chapter_number']}")
        print(f"   Synopsis: {chapter_data['synopsis'][:80]}...")
        
        # Create memory manager
        memory_manager = MemoryManager()
        
        # Start timing the entire insertion process
        total_start_time = time.time()
        
        print(f"   Starting LangGraph workflow...")
        
        # Use the write_new_memories method which internally uses LangGraph
        result = memory_manager.write_new_memories(chapter_data)
        
        total_end_time = time.time()
        total_time = (total_end_time - total_start_time) * 1000  # Convert to milliseconds
        
        # Extract timing information from the result
        step_times = result.get('step_times', {})
        
        print(f"   Insertion Results:")
        print(f"      Total Time: {total_time:.2f}ms")
        print(f"      Extracted: {result.get('extracted_count', 0)}")
        print(f"      Validated: {result.get('validated_count', 0)}")
        print(f"      Inserted: {result.get('inserted_count', 0)}")
        print(f"      Success: {'Yes' if result.get('success', False) else 'No'}")
        
        # Log step times if available
        if step_times:
            print(f"   Step-by-step timing:")
            for step, step_time in step_times.items():
                print(f"      {step}: {step_time:.2f}ms")
        
        # Log processing details
        if result.get('processing_log'):
            print(f"   Processing Log:")
            for log_entry in result['processing_log']:
                print(f"      - {log_entry}")
        
        total_results.append({
            "chapter": chapter_data['chapter_number'],
            "total_time": total_time,
            "extracted": result.get('extracted_count', 0),
            "validated": result.get('validated_count', 0),
            "inserted": result.get('inserted_count', 0),
            "success": result.get('success', False),
            "step_times": step_times
        })
    
    # Calculate and display summary statistics
    print(f"\nInsertion Time Summary:")
    
    successful_insertions = [r for r in total_results if r['success']]
    if successful_insertions:
        avg_total_time = statistics.mean([r['total_time'] for r in successful_insertions])
        min_total_time = min([r['total_time'] for r in successful_insertions])
        max_total_time = max([r['total_time'] for r in successful_insertions])
        
        print(f"   Successful Insertions: {len(successful_insertions)}/{len(total_results)}")
        print(f"   Average Total Time: {avg_total_time:.2f}ms")
        print(f"   Time Range: {min_total_time:.2f}ms - {max_total_time:.2f}ms")
        
        # Calculate average step times if available
        all_step_times = {}
        for result in successful_insertions:
            for step, step_time in result['step_times'].items():
                if step not in all_step_times:
                    all_step_times[step] = []
                all_step_times[step].append(step_time)
        
        if all_step_times:
            print(f"   Average Step Times:")
            for step, times in all_step_times.items():
                avg_step_time = statistics.mean(times)
                print(f"      {step}: {avg_step_time:.2f}ms")
    
    return total_results

def generate_performance_report():
    """Generate comprehensive performance report"""
    print("SYSTEM PERFORMANCE & EFFICIENCY EVALUATION REPORT")
    print("=" * 80)
    print("Running comprehensive performance evaluation...")
    
    # Test retrieval speed
    retrieval_results = test_retrieval_speed()
    
    # Test insertion time
    insertion_results = test_insertion_time()
    
    # Generate final report
    print_section_header("PERFORMANCE SUMMARY")
    
    print("Overall Performance Assessment:")
    
    # Retrieval performance
    print(f"\nRetrieval Performance:")
    for complexity, times in retrieval_results.items():
        if times:
            avg_time = statistics.mean(times)
            if avg_time < 500:
                status = "Excellent"
            elif avg_time < 1000:
                status = "Good"
            elif avg_time < 2000:
                status = "Moderate"
            else:
                status = "Poor"
            print(f"   {complexity.capitalize()} queries: {avg_time:.2f}ms - {status}")
    
    # Insertion performance
    print(f"\nInsertion Performance:")
    successful_insertions = [r for r in insertion_results if r['success']]
    if successful_insertions:
        avg_insertion_time = statistics.mean([r['total_time'] for r in successful_insertions])
        if avg_insertion_time < 5000:
            status = "Excellent"
        elif avg_insertion_time < 10000:
            status = "Good"
        elif avg_insertion_time < 20000:
            status = "Moderate"
        else:
            status = "Poor"
        print(f"   Average insertion time: {avg_insertion_time:.2f}ms - {status}")
        print(f"   Success rate: {len(successful_insertions)}/{len(insertion_results)} ({len(successful_insertions)/len(insertion_results)*100:.1f}%)")
    

    
    return {
        "retrieval_results": retrieval_results,
        "insertion_results": insertion_results
    }

if __name__ == "__main__":
    try:
        results = generate_performance_report()
        print(f"\nPerformance evaluation completed!")
        
    except KeyboardInterrupt:
        print(f"\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
