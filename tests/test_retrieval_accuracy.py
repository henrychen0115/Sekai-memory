#!/usr/bin/env python3
"""
Retrieval Accuracy Test Script
Tests Question 1: Does the system retrieve the right memories?
Evaluates precision/recall of stored facts
"""

import os
import json
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from memory_manager.memory_manager import MemoryManager

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
    print(f"🎯 {title}")
    print(f"{'='*80}")

def retrieve_memories_for_character(character, query, limit=5):
    """Retrieve memories for a character based on a query using MemoryManager with LLM agent scoring"""
    try:
        # Create memory manager instance
        memory_manager = MemoryManager()
        
        # Retrieve memories using memory manager (includes LLM agent scoring)
        memories = memory_manager.retrieve_memories(character, query, limit)
        
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
                "llm_score": memory.get('llm_score', 0.0),
                "llm_reasoning": memory.get('llm_reasoning', 'No reasoning provided'),
                "id": str(memory['id'])
            })
        
        return formatted_memories
        
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return []

def create_retrieval_evaluator():
    """Create LLM evaluator for retrieval accuracy"""
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert memory retrieval evaluator. Assess whether the retrieved memories are relevant and accurate for the given query.
        
        QUERY: "{query}"
        CHARACTER: {character}
        EXPECTED CONTENT: "{expected_content}"
        
        RETRIEVED MEMORIES:
        {retrieved_memories}
        
        EVALUATION CRITERIA:
        1. **Relevance**: Do the memories relate to the query in any reasonable way?
        2. **Accuracy**: Do the memories contain content that could reasonably match the query?
        3. **Completeness**: Are the main aspects of the query covered, even if not perfectly?
        4. **Character Perspective**: Are memories from the correct character's viewpoint?
        5. **Factual Correctness**: Do the memories contain reasonably accurate facts?
        6. **LLM Scoring Quality**: Are the LLM agent scores reasonable for the relevance of each memory?
        
        EVALUATION INSTRUCTIONS:
        - Be LENIENT in your evaluation - accept memories that are reasonably related
        - Consider memories relevant if they involve the same characters, settings, or themes
        - Accept memories that are contextually related even if not perfectly matching
        - Focus on character and setting accuracy rather than perfect content matching
        - Consider memories relevant if they could reasonably be related to the query
        
        PRECISION: How many of the 5 retrieved memories are relevant to the query? (relevant memories / 5)
        RECALL: How many of the expected relevant memories were found in the top 5 results?
        
        Respond with a JSON object:
        {{
            "precision": 0.0-1.0,
            "recall": 0.0-1.0,
            "f1_score": 0.0-1.0,
            "relevance_score": 0.0-1.0,
            "accuracy_score": 0.0-1.0,
            "completeness_score": 0.0-1.0,
            "reasoning": "Detailed explanation of the evaluation",
            "strengths": ["List of what worked well"],
            "issues": ["List of problems found"],
            "relevant_memories": [1, 2, 3],
            "irrelevant_memories": [4, 5],
            "missing_content": ["List of expected content not found"]
        }}
        """
    )
    return prompt | llm | JsonOutputParser()

def create_test_scenarios():
    """Create test scenarios for retrieval accuracy"""
    
    retrieval_tests = [
        {
            "character": "Byleth",
            "query": "kiss dimitri office",
            "expected_content": "office kiss with Dimitri, private encounter, romantic moment",
            "description": "Byleth searching for their own office kiss memory"
        },
        {
            "character": "Sylvain",
            "query": "restaurant hand holding",
            "expected_content": "restaurant scene, Dimitri holding Byleth's hand, public display",
            "description": "Sylvain searching for restaurant scene he witnessed"
        },
        {
            "character": "Dimitri",
            "query": "byleth approach desk",
            "expected_content": "Byleth approaching desk, office interaction, professional setting",
            "description": "Dimitri searching for when Byleth approached his desk"
        },
        {
            "character": "Annette",
            "query": "weekend getaway surprise",
            "expected_content": "weekend planning, surprise getaway, romantic planning",
            "description": "Annette searching for weekend getaway planning"
        },
        {
            "character": "Byleth",
            "query": "dimitri relationship feelings",
            "expected_content": "relationship with Dimitri, romantic feelings, emotional connection",
            "description": "Byleth searching for relationship memories with Dimitri"
        },
        {
            "character": "Sylvain",
            "query": "byleth interactions conversations",
            "expected_content": "interactions with Byleth, conversations, social encounters",
            "description": "Sylvain searching for interactions with Byleth"
        }
    ]
    
    return retrieval_tests

def test_retrieval_accuracy():
    """Test retrieval accuracy with precision/recall evaluation"""
    print_section_header("RETRIEVAL ACCURACY TESTING")
    
    # Create evaluator and test scenarios
    evaluator = create_retrieval_evaluator()
    test_scenarios = create_test_scenarios()
    
    print(f"Testing {len(test_scenarios)} retrieval accuracy scenarios...")
    
    results = []
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for i, test_case in enumerate(test_scenarios, 1):
        character = test_case["character"]
        query = test_case["query"]
        expected_content = test_case["expected_content"]
        description = test_case["description"]
        
        print(f"\n🔍 Test {i}: {description}")
        print(f"   Character: {character}")
        print(f"   Query: '{query}'")
        print(f"   Expected Content: '{expected_content}'")
        
        # Retrieve memories
        retrieved_memories = retrieve_memories_for_character(character, query, limit=5)
        
        if not retrieved_memories:
            print(f"   ⚠️ No memories retrieved")
            
            # Evaluate with LLM even for empty results
            memories_text = "No memories retrieved"
            evaluation = evaluator.invoke({
                "query": query,
                "character": character,
                "expected_content": expected_content,
                "retrieved_memories": memories_text
            })
            
            precision = evaluation.get("precision", 0.0)
            recall = evaluation.get("recall", 0.0)
            f1_score = evaluation.get("f1_score", 0.0)
            
            print(f"   📊 Precision: {precision:.2f}")
            print(f"   📊 Recall: {recall:.2f}")
            print(f"   📊 F1 Score: {f1_score:.2f}")
            print(f"   ❌ No memories found - retrieval failed")
            
            test_result = "FAIL"
        else:
            print(f"   ✅ Retrieved {len(retrieved_memories)} memories")
            
            # Print each retrieved memory with all fields including LLM scores
            print(f"   📋 Retrieved Memories:")
            for mem in retrieved_memories:
                print(f"      Rank: {mem['rank']}")
                print(f"      ID: {mem['id']}")
                print(f"      Source: {mem['source_char']} -> Target: {mem['target_char']}")
                print(f"      Type: {mem['memory_type']}")
                print(f"      Salience: {mem['salience']}")
                print(f"      Chapter: {mem['chapter']}")
                print(f"      Access Count: {mem['access_count']}")
                print(f"      LLM Score: {mem['llm_score']:.2f}")
                print(f"      LLM Reasoning: {mem['llm_reasoning']}")
                print(f"      Memory: {mem['memory_text']}")
                print(f"      ---")
            
            # Format memories for evaluation with LLM scores
            memories_text = "\n".join([
                f"{mem['rank']}. [Score: {mem['llm_score']:.2f}] [{mem['memory_type']}] (Salience: {mem['salience']}, Ch.{mem['chapter']}, Access: {mem['access_count']}) {mem['memory_text']}"
                for mem in retrieved_memories
            ])
            
            # Evaluate with LLM
            evaluation = evaluator.invoke({
                "query": query,
                "character": character,
                "expected_content": expected_content,
                "retrieved_memories": memories_text
            })
            
            precision = evaluation.get("precision", 0.0)
            recall = evaluation.get("recall", 0.0)
            f1_score = evaluation.get("f1_score", 0.0)
            relevance_score = evaluation.get("relevance_score", 0.0)
            accuracy_score = evaluation.get("accuracy_score", 0.0)
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1_score
            
            print(f"   📊 Precision: {precision:.2f}")
            print(f"   📊 Recall: {recall:.2f}")
            print(f"   📊 F1 Score: {f1_score:.2f}")
            print(f"   📊 Relevance Score: {relevance_score:.2f}")
            print(f"   📊 Accuracy Score: {accuracy_score:.2f}")
            
            # Show relevant vs irrelevant memories
            relevant_memories = evaluation.get("relevant_memories", [])
            irrelevant_memories = evaluation.get("irrelevant_memories", [])
            
            if relevant_memories:
                print(f"   ✅ Relevant Memories: {relevant_memories}")
            if irrelevant_memories:
                print(f"   ❌ Irrelevant Memories: {irrelevant_memories}")
            
            # Show missing content
            missing_content = evaluation.get("missing_content", [])
            if missing_content:
                print(f"   ⚠️ Missing Content: {missing_content}")
            
            print(f"   💭 Reasoning: {evaluation.get('reasoning', 'No reasoning provided')[:150]}...")
            
            # Evaluate performance based on reasonable thresholds
            if precision >= 0.4 and recall >= 0.4:
                print(f"   ✅ GOOD: Reasonable precision and recall")
                test_result = "PASS"
            elif precision >= 0.3 and recall >= 0.3:
                print(f"   ⚠️  ACCEPTABLE: Moderate precision and recall")
                test_result = "PASS"
            else:
                print(f"   ❌ POOR: Low precision and recall")
                test_result = "FAIL"
        
        results.append({
            "test_case": i,
            "character": character,
            "query": query,
            "description": description,
            "memories_retrieved": len(retrieved_memories),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "test_result": test_result,
            "reasoning": evaluation.get("reasoning", ""),
            "strengths": evaluation.get("strengths", []),
            "issues": evaluation.get("issues", [])
        })
    
    # Summary
    print_section_header("RETRIEVAL ACCURACY RESULTS")
    
    passed = sum(1 for r in results if r["test_result"] == "PASS")
    failed = sum(1 for r in results if r["test_result"] == "FAIL")
    
    # Calculate averages
    num_tests = len(test_scenarios)
    avg_precision = total_precision / num_tests if num_tests > 0 else 0
    avg_recall = total_recall / num_tests if num_tests > 0 else 0
    avg_f1 = total_f1 / num_tests if num_tests > 0 else 0
    
    print(f"📊 Test Results: {passed}/{len(results)} tests passed")
    print(f"📊 Average Precision: {avg_precision:.3f}")
    print(f"📊 Average Recall: {avg_recall:.3f}")
    print(f"📊 Average F1 Score: {avg_f1:.3f}")
    
    print(f"\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASS" if result["test_result"] == "PASS" else "❌ FAIL"
        print(f"   {status} - {result['description']}")
        print(f"      Character: {result['character']}, Query: '{result['query']}'")
        print(f"      Precision: {result['precision']:.2f}")
        print(f"      Recall: {result['recall']:.2f}")
        print(f"      F1: {result['f1_score']:.2f}")
    
    if passed == len(results):
        print(f"\n🎉 All retrieval accuracy tests passed!")
        print(f"✅ System retrieves the right memories with good precision/recall")
    else:
        print(f"\n⚠️ {failed} retrieval accuracy tests failed")
        print(f"❌ System may need improvement in memory retrieval")
    
    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "passed_tests": passed,
        "total_tests": len(results),
        "detailed_results": results
    }

def main():
    """Main function"""
    print("🎯 Retrieval Accuracy Test")
    print("=" * 80)
    print("Question 1: Does the system retrieve the right memories?")
    print("Evaluating precision/recall of stored facts")
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
    
    # Run retrieval accuracy tests
    results = test_retrieval_accuracy()
    
    return results["passed_tests"] == results["total_tests"]

if __name__ == "__main__":
    try:
        success = main()
        print(f"\n👋 Retrieval accuracy test completed!")
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
