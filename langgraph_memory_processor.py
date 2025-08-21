#!/usr/bin/env python3
"""
LangGraph Memory Processing System
Uses LangGraph to process memory insertion with multiple nodes:
1. Extract atomic memories from synopsis (using Gemini Flash)
2. Validate memories against synopsis (using GPT-3.5-turbo)
3. Check for conflicts/redundancy with existing memories
4. Insert into database
"""

import os
import json
import uuid
import psycopg2
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Initialize models
gemini_llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

gpt_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-lite-001",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# State definition
class MemoryState(TypedDict):
    """State for the memory processing workflow"""
    chapter_data: Dict[str, Any]
    extracted_memories: List[Dict[str, Any]]
    validated_memories: List[Dict[str, Any]]
    conflict_checked_memories: List[Dict[str, Any]]
    inserted_memories: List[Dict[str, Any]]
    errors: List[str]
    processing_log: List[str]
    step_times: Dict[str, float]

def get_db_connection():
    """Get database connection"""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    register_vector(conn)
    return conn

# Node 1: Extract Atomic Memories (using Gemini Flash)
def extract_memories(state: MemoryState) -> MemoryState:
    """Extract atomic memories from synopsis using Gemini Flash"""
    
    import time
    step_start_time = time.time()
    
    chapter_data = state["chapter_data"]
    chapter_num = chapter_data['chapter_number']
    synopsis = chapter_data['synopsis']
    
    print(f"🔍 Extracting memories from Chapter {chapter_num}...")
    
    # Create extraction prompt
    extraction_prompt = ChatPromptTemplate.from_template(
        """
        You are a precise narrative analysis engine. Extract atomic memories for each character present in the events.
        
        CRITICAL RULES:
        1. Create memories for ALL characters who were present or directly involved
        2. Each memory must be from the perspective of a character who was actually present
        3. DO NOT invent events not mentioned in the synopsis
        4. Focus on concrete actions and observable events
        5. If multiple characters are present, create memories for each perspective
        
        CHARACTER NAME STANDARDIZATION:
        - Use ONLY first names for all characters
        - "Byleth Eisner" → "Byleth"
        - "Dimitri Alexandre Blaiddyd" → "Dimitri"
        - "Sylvain Jose Gautier" → "Sylvain"
        - "Felix Hugo Fraldarius" → "Felix"
        - "Annette Fantine Dominic" → "Annette"
        - "Dedue Molinaro" → "Dedue"
        - Any other character: use their first name only
        
        Synopsis from Chapter {chapter_number}:
        "{synopsis}"
        
        MEMORY FORMAT:
        - source: character whose memory this is (FIRST NAME ONLY)
        - target: other character involved (if any) (FIRST NAME ONLY)
        - type: C2U (involves Byleth), IC (between NPCs), WM (world state)
        - salience: 1-10 (importance level)
        - memory_text: First-person description of what the character experienced
        
        Respond with JSON list:
        [
            {{
                "source": "first_name_only",
                "target": "first_name_only_or_null",
                "type": "C2U|IC|WM",
                "salience": 1-10,
                "memory_text": "First-person memory description"
            }}
        ]
        
        REMEMBER: Create memories for ALL characters present in the scene. Use FIRST NAMES ONLY.
        """
    )
    
    try:
        # Extract memories using Gemini Flash
        chain = extraction_prompt | gemini_llm | JsonOutputParser()
        extracted_memories = chain.invoke({
            "chapter_number": chapter_num,
            "synopsis": synopsis
        })
        
        # Add chapter number to each memory
        for memory in extracted_memories:
            memory['chapter'] = chapter_num
        
        step_end_time = time.time()
        step_time = (step_end_time - step_start_time) * 1000  # Convert to milliseconds
        
        print(f"✅ Extracted {len(extracted_memories)} memories")
        
        return {
            **state,
            "extracted_memories": extracted_memories,
            "processing_log": state["processing_log"] + [f"Extracted {len(extracted_memories)} memories from Chapter {chapter_num}"],
            "step_times": {**state.get("step_times", {}), "extraction": step_time}
        }
        
    except Exception as e:
        error_msg = f"Error extracting memories: {e}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "processing_log": state["processing_log"] + [error_msg]
        }

# Node 2: Validate Memories Against Synopsis (using GPT-3.5-turbo)
def validate_memories(state: MemoryState) -> MemoryState:
    """Validate extracted memories against the original synopsis using GPT-3.5-turbo"""
    
    import time
    step_start_time = time.time()
    
    if not state["extracted_memories"]:
        return state
    
    chapter_data = state["chapter_data"]
    synopsis = chapter_data['synopsis']
    extracted_memories = state["extracted_memories"]
    
    print(f"✅ Validating {len(extracted_memories)} memories against synopsis...")
    
    # Create validation prompt
    validation_prompt = ChatPromptTemplate.from_template(
        """
        You are a memory validation expert. Validate if each memory is faithful to the source synopsis.
        
        ORIGINAL SYNOPSIS:
        "{synopsis}"
        
        MEMORIES TO VALIDATE:
        {memories_text}
        
        VALIDATION CRITERIA:
        1. **Factual Accuracy**: Does the memory describe events that actually happened in the synopsis?
        2. **Character Presence**: Was the source character actually present during these events?
        3. **No Hallucination**: Does the memory avoid inventing events not mentioned?
        4. **Perspective Accuracy**: Is the memory from a perspective the character could reasonably have?
        5. **Contextual Fit**: Does the memory fit the overall context and setting?
        6. **Character Name Standardization**: Are character names using first names only? (Byleth, Dimitri, Sylvain, etc.)
        
        JUDGMENT INSTRUCTIONS:
        - Be LENIENT - accept memories that are reasonably accurate
        - Accept memories that describe events the character could have experienced
        - Only reject memories that are completely wrong or involve wrong characters/settings
        - Focus on character and setting accuracy rather than perfect factual precision
        
        For each memory, respond with:
        {{
            "valid_memories": [list of valid memory indices],
            "rejected_memories": [list of rejected memory indices],
            "reasoning": "Explanation of validation decisions"
        }}
        """
    )
    
    try:
        # Format memories for validation
        memories_text = ""
        for i, memory in enumerate(extracted_memories):
            memories_text += f"{i+1}. [{memory['source']}] {memory['memory_text']}\n"
        
        # Validate memories using GPT-3.5-turbo
        chain = validation_prompt | gpt_llm | JsonOutputParser()
        validation_result = chain.invoke({
            "synopsis": synopsis,
            "memories_text": memories_text
        })
        
        # Filter valid memories
        valid_indices = validation_result.get("valid_memories", [])
        validated_memories = [extracted_memories[i-1] for i in valid_indices if 1 <= i <= len(extracted_memories)]
        
        step_end_time = time.time()
        step_time = (step_end_time - step_start_time) * 1000  # Convert to milliseconds
        
        print(f"✅ Validated: {len(validated_memories)} valid, {len(extracted_memories) - len(validated_memories)} rejected")
        
        return {
            **state,
            "validated_memories": validated_memories,
            "processing_log": state["processing_log"] + [f"Validated {len(validated_memories)}/{len(extracted_memories)} memories"],
            "step_times": {**state.get("step_times", {}), "validation": step_time}
        }
        
    except Exception as e:
        error_msg = f"Error validating memories: {e}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "processing_log": state["processing_log"] + [error_msg]
        }

# Node 3: Check for Conflicts/Redundancy
def check_conflicts(state: MemoryState) -> MemoryState:
    """Check each memory for conflicts or redundancy with existing memories"""
    
    import time
    step_start_time = time.time()
    
    if not state["validated_memories"]:
        return state
    
    validated_memories = state["validated_memories"]
    
    print(f"🔍 Checking {len(validated_memories)} memories for conflicts...")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        conflict_checked_memories = []
        
        for memory in validated_memories:
            source_char = memory['source']
            
            # Get existing memories for this character
            cursor.execute("""
                SELECT memory_text, embedding, chapter
                FROM memories
                WHERE source_char = %s
                ORDER BY chapter DESC, salience DESC
                LIMIT 15
            """, (source_char,))
            
            existing_memories = cursor.fetchall()
            
            if not existing_memories:
                # No existing memories, accept this one
                conflict_checked_memories.append(memory)
                continue
            
            # Create embedding for new memory
            new_memory_embedding = embeddings_model.embed_query(memory['memory_text'])
            
            # Check semantic similarity with existing memories
            max_similarity = 0.0
            most_similar_text = ""
            most_similar_chapter = 0
            
            for existing_text, existing_embedding, existing_chapter in existing_memories:
                similarity = calculate_cosine_similarity(new_memory_embedding, existing_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_text = existing_text
                    most_similar_chapter = existing_chapter
            
            # More intelligent conflict detection
            is_duplicate = False
            
            # Only reject if it's an extremely close match (near-duplicate)
            if max_similarity > 0.98:
                is_duplicate = True
                print(f"   ❌ Rejected {source_char} memory: Near-duplicate (similarity: {max_similarity:.3f})")
                print(f"      Similar to: Ch.{most_similar_chapter} - {most_similar_text[:80]}...")
            
            # For high similarity but not duplicate, use LLM to make final decision
            elif max_similarity > 0.90:
                # Use LLM to determine if it's truly a duplicate
                is_duplicate = check_if_duplicate_with_llm(memory['memory_text'], most_similar_text)
                if is_duplicate:
                    print(f"   ❌ Rejected {source_char} memory: LLM determined duplicate (similarity: {max_similarity:.3f})")
                else:
                    print(f"   ✅ Accepted {source_char} memory: Similar but different (similarity: {max_similarity:.3f})")
            
            if not is_duplicate:
                conflict_checked_memories.append(memory)
        
        cursor.close()
        conn.close()
        
        step_end_time = time.time()
        step_time = (step_end_time - step_start_time) * 1000  # Convert to milliseconds
        
        print(f"✅ Conflict check: {len(conflict_checked_memories)} accepted, {len(validated_memories) - len(conflict_checked_memories)} rejected")
        
        return {
            **state,
            "conflict_checked_memories": conflict_checked_memories,
            "processing_log": state["processing_log"] + [f"Conflict check: {len(conflict_checked_memories)}/{len(validated_memories)} accepted"],
            "step_times": {**state.get("step_times", {}), "conflict_check": step_time}
        }
        
    except Exception as e:
        error_msg = f"Error checking conflicts: {e}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "processing_log": state["processing_log"] + [error_msg]
        }

# Node 4: Insert into Database
def insert_memories(state: MemoryState) -> MemoryState:
    """Insert conflict-checked memories into database"""
    
    import time
    step_start_time = time.time()
    
    if not state["conflict_checked_memories"]:
        return state
    
    memories_to_insert = state["conflict_checked_memories"]
    
    print(f"💾 Inserting {len(memories_to_insert)} memories into database...")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create embeddings for all memories
        memory_texts = [memory['memory_text'] for memory in memories_to_insert]
        memory_embeddings = embeddings_model.embed_documents(memory_texts)
        
        inserted_memories = []
        
        for i, memory in enumerate(memories_to_insert):
            memory_id = str(uuid.uuid4())
            embedding_vector = memory_embeddings[i]
            
            cursor.execute("""
                INSERT INTO memories (id, chapter, source_char, target_char, memory_type, salience, memory_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                memory_id,
                memory.get('chapter', 999),
                memory['source'],
                memory.get('target'),
                memory['type'],
                memory['salience'],
                memory['memory_text'],
                embedding_vector
            ))
            
            inserted_memories.append({
                **memory,
                'id': memory_id
            })
        
        conn.commit()
        cursor.close()
        conn.close()
        
        step_end_time = time.time()
        step_time = (step_end_time - step_start_time) * 1000  # Convert to milliseconds
        
        print(f"✅ Successfully inserted {len(inserted_memories)} memories")
        
        return {
            **state,
            "inserted_memories": inserted_memories,
            "processing_log": state["processing_log"] + [f"Inserted {len(inserted_memories)} memories into database"],
            "step_times": {**state.get("step_times", {}), "insertion": step_time}
        }
        
    except Exception as e:
        error_msg = f"Error inserting memories: {e}"
        print(f"❌ {error_msg}")
        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "processing_log": state["processing_log"] + [error_msg]
        }

def check_if_duplicate_with_llm(new_memory_text, existing_memory_text):
    """Use GPT-3.5-turbo to determine if two memories are truly duplicates"""
    
    duplicate_check_prompt = ChatPromptTemplate.from_template(
        """
        You are a memory duplicate detection expert. Determine if two memories describe the same event or are duplicates.
        
        MEMORY 1: "{new_memory}"
        MEMORY 2: "{existing_memory}"
        
        ANALYSIS CRITERIA:
        1. **Same Event**: Do both memories describe the same specific event, action, or interaction?
        2. **Same Characters**: Do both memories involve the same characters in the same roles?
        3. **Same Setting**: Do both memories take place in the same location/context?
        4. **Same Outcome**: Do both memories describe the same result or consequence?
        5. **Same Perspective**: Are both memories from the same character's viewpoint?
        
        JUDGMENT INSTRUCTIONS:
        - Be LENIENT - only mark as duplicate if they are clearly describing the exact same event
        - Consider memories different if they describe similar but distinct events
        - Consider memories different if they have different emotional tones or details
        - Consider memories different if they focus on different aspects of the same event
        
        Respond with JSON:
        {{
            "is_duplicate": true/false,
            "reasoning": "Explanation of why they are or are not duplicates"
        }}
        """
    )
    
    try:
        chain = duplicate_check_prompt | gpt_llm | JsonOutputParser()
        result = chain.invoke({
            "new_memory": new_memory_text,
            "existing_memory": existing_memory_text
        })
        
        return result.get("is_duplicate", False)
        
    except Exception as e:
        print(f"   ⚠️  LLM duplicate check failed: {e}")
        return False  # Default to accepting if LLM check fails

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

# Create LangGraph workflow
def create_memory_workflow():
    """Create the LangGraph workflow for memory processing"""
    
    # Create the graph
    workflow = StateGraph(MemoryState)
    
    # Add nodes
    workflow.add_node("extract_memories", extract_memories)
    workflow.add_node("validate_memories", validate_memories)
    workflow.add_node("check_conflicts", check_conflicts)
    workflow.add_node("insert_memories", insert_memories)
    
    # Define the flow
    workflow.set_entry_point("extract_memories")
    workflow.add_edge("extract_memories", "validate_memories")
    workflow.add_edge("validate_memories", "check_conflicts")
    workflow.add_edge("check_conflicts", "insert_memories")
    workflow.add_edge("insert_memories", END)
    
    # Compile the graph
    return workflow.compile()

class LangGraphMemoryProcessor:
    """LangGraph-based memory processing system"""
    
    def __init__(self):
        self.workflow = create_memory_workflow()
    
    def process_chapter(self, chapter_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single chapter through the LangGraph workflow
        
        Args:
            chapter_data: Dictionary with 'chapter_number' and 'synopsis'
            
        Returns:
            Dict containing processing results
        """
        
        print(f"\n🚀 LangGraph Memory Processing for Chapter {chapter_data['chapter_number']}")
        print("=" * 60)
        
        # Initialize state
        initial_state = MemoryState(
            chapter_data=chapter_data,
            extracted_memories=[],
            validated_memories=[],
            conflict_checked_memories=[],
            inserted_memories=[],
            errors=[],
            processing_log=[],
            step_times={}
        )
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Prepare results
            results = {
                "chapter_number": chapter_data['chapter_number'],
                "extracted_count": len(final_state["extracted_memories"]),
                "validated_count": len(final_state["validated_memories"]),
                "conflict_checked_count": len(final_state["conflict_checked_memories"]),
                "inserted_count": len(final_state["inserted_memories"]),
                "errors": final_state["errors"],
                "processing_log": final_state["processing_log"],
                "step_times": final_state.get("step_times", {}),
                "success": len(final_state["errors"]) == 0
            }
            
            print(f"\n📊 Processing Results:")
            print(f"   Extracted: {results['extracted_count']}")
            print(f"   Validated: {results['validated_count']}")
            print(f"   Conflict Checked: {results['conflict_checked_count']}")
            print(f"   Inserted: {results['inserted_count']}")
            
            if results['errors']:
                print(f"   Errors: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"      - {error}")
            
            return results
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "chapter_number": chapter_data['chapter_number'],
                "extracted_count": 0,
                "validated_count": 0,
                "conflict_checked_count": 0,
                "inserted_count": 0,
                "errors": [error_msg],
                "processing_log": [],
                "success": False
            }

def test_langgraph_processor():
    """Test the LangGraph memory processor"""
    
    # Test chapter data
    test_chapter = {
        "chapter_number": 99,
        "synopsis": "Byleth and Dimitri have a private meeting in Dimitri's office. They discuss company strategy and share a moment of mutual understanding. The conversation reveals their growing professional respect for each other."
    }
    
    # Create processor and test
    processor = LangGraphMemoryProcessor()
    results = processor.process_chapter(test_chapter)
    
    print(f"\n🎉 Test completed: {'Success' if results['success'] else 'Failed'}")
    return results

if __name__ == "__main__":
    test_langgraph_processor()
