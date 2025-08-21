#!/usr/bin/env python3
"""
Internal Consistency Test Script
Tests Question 2: Are the memories internally consistent across time, characters, and world state?
Detects "cross-talk" or forgotten updates
"""

import os
import json
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from memory_manager import MemoryManager

# Load environment variables
load_dotenv()

# Initialize OpenRouter for LLM
llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}")

def retrieve_memories_for_character(character, query=None, limit=None):
    """Retrieve all memories for a character and sort by chapter (earliest to latest)"""
    try:
        # Create memory manager instance
        memory_manager = MemoryManager()
        
        # Get all memories for the character (no limit to get all)
        memories = memory_manager.list_memories_for_character(character, limit=1000)  # Large limit to get all
        
        # Sort memories by chapter number (earliest to latest)
        memories.sort(key=lambda x: x['chapter'])
        
        # Format memories to match expected structure
        formatted_memories = []
        for i, memory in enumerate(memories, 1):
            formatted_memories.append({
                "rank": i,
                "memory_text": memory['memory_text'],
                "salience": memory['salience'],
                "memory_type": memory['type'],
                "source_char": memory['source_char'],
                "target_char": memory['target_char'],
                "chapter": memory['chapter'],
                "access_count": memory.get('access_count', 0),
                "id": str(memory['id'])
            })
        
        return formatted_memories
        
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return []

def create_consistency_evaluator():
    """Create LLM evaluator for internal consistency - evaluating all characters on same 5 fields"""
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert memory consistency evaluator. Evaluate ALL characters and ALL their memories on the same 5 consistency fields.
        
        CHARACTER: {character}
        MEMORY SET:
        {memory_set}
        
        EVALUATION CRITERIA (Same 5 fields for ALL characters):
        1. **Temporal Consistency (T)**: Are memories in logical chronological order? Check for time-related inconsistencies.
        2. **Knowledge Boundaries (K)**: Does the character know things they shouldn't? Check for cross-talk and private information access.
        3. **World State Consistency (W)**: Are memories consistent with established facts about the world, setting, and relationships?
        4. **Forgotten Updates (F)**: Are there outdated memories that should have been updated based on later events?
        5. **Contradictions (C)**: Do memories contradict each other within the same character's memory set?
        
        ANALYSIS INSTRUCTIONS:
        - Evaluate ALL characters using the SAME 5 criteria
        - Count violations in each category: T, K, W, F, C
        - Be consistent and fair across all characters
        - Focus on objective consistency issues, not character-specific knowledge
        - Look for patterns across the entire memory system
        
        Respond with a JSON object:
        {{
            "consistency_score": 0.0-1.0,
            "temporal_violations": 0,
            "knowledge_violations": 0,
            "world_state_violations": 0,
            "forgotten_update_violations": 0,
            "contradiction_violations": 0,
            "total_violations": 0,
            "violation_details": {{
                "temporal": ["List of temporal issues"],
                "knowledge": ["List of knowledge boundary violations"],
                "world_state": ["List of world state conflicts"],
                "forgotten_updates": ["List of forgotten updates"],
                "contradictions": ["List of contradictions"]
            }},
            "reasoning": "Detailed explanation of consistency analysis across all 5 fields",
            "recommendations": ["Suggestions for fixing consistency issues"]
        }}
        
        Remember: Evaluate ALL characters on the SAME 5 fields (T:0, K:0, W:0, F:0, C:0) for fair comparison.
        """
    )
    return prompt | llm | JsonOutputParser()

def create_test_scenarios():
    """Create test scenarios for internal consistency - evaluating main characters on same 5 fields"""
    
    # Only test the main characters to reduce output length
    main_characters = ["Byleth", "Dimitri", "Sylvain"]
    
    consistency_tests = []
    
    for character in main_characters:
        consistency_tests.append({
            "character": character,
            "query": "all memories",
            "description": f"Comprehensive consistency analysis for {character}",
            "focus": "all_consistency_fields"
        })
    
    return consistency_tests

def test_internal_consistency():
    """Test internal consistency with cross-talk and forgotten updates detection"""
    print_section_header("INTERNAL CONSISTENCY TESTING")
    
    # Create evaluator and test scenarios
    evaluator = create_consistency_evaluator()
    test_scenarios = create_test_scenarios()
    
    print(f"Testing {len(test_scenarios)} internal consistency scenarios...")
    
    results = []
    total_consistency = 0
    cross_talk_count = 0
    total_violations = 0
    
    for i, test_case in enumerate(test_scenarios, 1):
        character = test_case["character"]
        query = test_case["query"]
        description = test_case["description"]
        focus = test_case["focus"]
        
        print(f"\n🔍 Test {i}: {description}")
        print(f"   Character: {character}")
        print(f"   Focus: {focus}")
        print(f"   Analysis: All memories evaluated on 5 consistency fields (T, K, W, F, C)")
        
        # Retrieve memories for consistency analysis
        retrieved_memories = retrieve_memories_for_character(character)
        
        if not retrieved_memories:
            print(f"   ⚠️ No memories found for consistency analysis")
            
            # Evaluate with LLM even for empty results
            memories_text = "No memories retrieved"
            evaluation = evaluator.invoke({
                "character": character,
                "memory_set": memories_text
            })
            
            consistency_score = evaluation.get("consistency_score", 0.0)
            cross_talk_detected = evaluation.get("cross_talk_detected", False)
            
            print(f"   📊 Consistency Score: {consistency_score:.2f}")
            print(f"   🚨 Cross-talk Detected: {cross_talk_detected}")
            print(f"   ✅ No consistency issues (no memories to analyze)")
            
            test_result = "PASS"
        else:
            print(f"   ✅ Retrieved {len(retrieved_memories)} memories (sorted by chapter)")
            
            # Print each retrieved memory with access count
            print(f"   📋 All Memories (Chronological Order):")
            for mem in retrieved_memories:
                print(f"      Rank: {mem['rank']}")
                print(f"      ID: {mem['id']}")
                print(f"      Source: {mem['source_char']} -> Target: {mem['target_char']}")
                print(f"      Type: {mem['memory_type']}")
                print(f"      Salience: {mem['salience']}")
                print(f"      Chapter: {mem['chapter']}")
                print(f"      Access Count: {mem['access_count']}")
                print(f"      Memory: {mem['memory_text']}")
                print(f"      ---")
            
            # Format memories for evaluation with access count
            memories_text = "\n".join([
                f"{mem['rank']}. [{mem['memory_type']}] (Ch.{mem['chapter']}, Access: {mem['access_count']}) {mem['memory_text']}"
                for mem in retrieved_memories
            ])
            
            # Evaluate consistency with LLM
            evaluation = evaluator.invoke({
                "character": character,
                "memory_set": memories_text
            })
            
            consistency_score = evaluation.get("consistency_score", 0.0)
            temporal_violations = evaluation.get("temporal_violations", 0)
            knowledge_violations = evaluation.get("knowledge_violations", 0)
            world_state_violations = evaluation.get("world_state_violations", 0)
            forgotten_update_violations = evaluation.get("forgotten_update_violations", 0)
            contradiction_violations = evaluation.get("contradiction_violations", 0)
            total_violations_count = evaluation.get("total_violations", 0)
            
            total_consistency += consistency_score
            total_violations += total_violations_count
            
            print(f"   📊 Consistency Score: {consistency_score:.2f}")
            print(f"   📋 Violations Summary: (T:{temporal_violations}, K:{knowledge_violations}, W:{world_state_violations}, F:{forgotten_update_violations}, C:{contradiction_violations})")
            print(f"   ⚠️ Total Violations: {total_violations_count}")
            
            # Report violation details
            violation_details = evaluation.get("violation_details", {})
            
            if violation_details.get("temporal"):
                print(f"   ⏰ Temporal Issues ({len(violation_details['temporal'])}):")
                for issue in violation_details["temporal"]:
                    print(f"      - {issue}")
            
            if violation_details.get("knowledge"):
                print(f"   🧠 Knowledge Violations ({len(violation_details['knowledge'])}):")
                for violation in violation_details["knowledge"]:
                    print(f"      - {violation}")
            
            if violation_details.get("world_state"):
                print(f"   🌍 World State Conflicts ({len(violation_details['world_state'])}):")
                for conflict in violation_details["world_state"]:
                    print(f"      - {conflict}")
            
            if violation_details.get("forgotten_updates"):
                print(f"   🔄 Forgotten Updates ({len(violation_details['forgotten_updates'])}):")
                for update in violation_details["forgotten_updates"]:
                    print(f"      - {update}")
            
            if violation_details.get("contradictions"):
                print(f"   ⚠️ Contradictions ({len(violation_details['contradictions'])}):")
                for contradiction in violation_details["contradictions"]:
                    print(f"      - {contradiction}")
            
            print(f"   💭 Reasoning: {evaluation.get('reasoning', 'No reasoning provided')[:150]}...")
            
            # Determine if test passed based on total violations
            if consistency_score >= 0.7 and total_violations_count == 0:
                print(f"   ✅ PASS: Good consistency, no violations")
                test_result = "PASS"
            elif consistency_score >= 0.5 and total_violations_count <= 2:
                print(f"   ⚠️ WARNING: Moderate consistency issues detected")
                test_result = "WARNING"
            else:
                print(f"   ❌ FAIL: Significant consistency issues detected")
                test_result = "FAIL"
        
        results.append({
            "test_case": i,
            "character": character,
            "description": description,
            "focus": focus,
            "memories_analyzed": len(retrieved_memories),
            "consistency_score": consistency_score,
            "temporal_violations": temporal_violations,
            "knowledge_violations": knowledge_violations,
            "world_state_violations": world_state_violations,
            "forgotten_update_violations": forgotten_update_violations,
            "contradiction_violations": contradiction_violations,
            "total_violations": total_violations_count,
            "test_result": test_result,
            "reasoning": evaluation.get("reasoning", "")
        })
    
    # Summary
    print_section_header("INTERNAL CONSISTENCY RESULTS")
    
    passed = sum(1 for r in results if r["test_result"] == "PASS")
    warnings = sum(1 for r in results if r["test_result"] == "WARNING")
    failed = sum(1 for r in results if r["test_result"] == "FAIL")
    
    # Calculate averages
    num_tests = len(test_scenarios)
    avg_consistency = total_consistency / num_tests if num_tests > 0 else 0
    
    print(f"📊 Test Results: {passed} PASS, {warnings} WARNING, {failed} FAIL")
    print(f"📊 Average Consistency Score: {avg_consistency:.3f}")
    print(f"📋 Total Violations: {total_violations}")
    print(f"⚠️ Total Violations Found: {total_violations}")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        if result["test_result"] == "PASS":
            status = "✅ PASS"
        elif result["test_result"] == "WARNING":
            status = "⚠️ WARNING"
        else:
            status = "❌ FAIL"
        
        print(f"   {status} - {result['description']}")
        print(f"      Character: {result['character']}, Focus: {result['focus']}")
        print(f"      Consistency: {result['consistency_score']:.2f}")
        print(f"      Violations: {result['total_violations']} (T:{result['temporal_violations']}, K:{result['knowledge_violations']}, W:{result['world_state_violations']}, F:{result['forgotten_update_violations']}, C:{result['contradiction_violations']})")
    
    if passed == len(results):
        print(f"\n🎉 All internal consistency tests passed!")
        print(f"✅ Memories are internally consistent with no cross-talk")
    elif failed == 0:
        print(f"\n⚠️ {warnings} tests have warnings but no failures")
        print(f"⚠️ Some consistency issues detected but not critical")
    else:
        print(f"\n❌ {failed} internal consistency tests failed")
        print(f"❌ Significant cross-talk or consistency issues detected")
    
    return {
        "avg_consistency": avg_consistency,
        "total_violations": total_violations,
        "passed_tests": passed,
        "warning_tests": warnings,
        "failed_tests": failed,
        "total_tests": len(results),
        "detailed_results": results
    }

def main():
    """Main function"""
    print("🔍 Internal Consistency Test")
    print("=" * 80)
    print("Question 2: Are the memories internally consistent across time, characters, and world state?")
    print("Detecting cross-talk and forgotten updates")
    print("=" * 80)
    
    # Check prerequisites
    print("\n🔍 Checking database connection...")
    
    try:
        memory_manager = MemoryManager()
        stats = memory_manager.get_database_stats()
        memory_count = stats.get('total_memories', 0)
        print(f"✅ Database connected. Found {memory_count} memories for testing.")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    
    if memory_count == 0:
        print("❌ No memories found in database. Please populate the database first.")
        return False
    
    # Run internal consistency tests
    results = test_internal_consistency()
    
    return results["failed_tests"] == 0

if __name__ == "__main__":
    try:
        success = main()
        print(f"\n👋 Internal consistency test completed!")
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
