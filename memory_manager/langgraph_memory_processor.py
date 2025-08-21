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
import logging
import psycopg2
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END

# Configure logging - only important information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize models
gemini_llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# gpt_llm = ChatOpenAI(
#     model="google/gemini-2.0-flash-lite-001",
#     temperature=0,
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1"
# )

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
    
    # Removed excessive logging - only log important steps
    
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
        
        # Removed excessive logging
        
        return {
            **state,
            "extracted_memories": extracted_memories,
            "processing_log": state["processing_log"] + [f"Extracted {len(extracted_memories)} memories from Chapter {chapter_num}"],
            "step_times": {**state.get("step_times", {}), "extraction": step_time}
        }
        
    except Exception as e:
        error_msg = f"Error extracting memories: {e}"
        logger.error(f"{error_msg}")
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
    
    # Removed excessive logging
    
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
        chain = validation_prompt | gemini_llm | JsonOutputParser()
        validation_result = chain.invoke({
            "synopsis": synopsis,
            "memories_text": memories_text
        })
        
        # Filter valid memories
        valid_indices = validation_result.get("valid_memories", [])
        validated_memories = [extracted_memories[i-1] for i in valid_indices if 1 <= i <= len(extracted_memories)]
        
        step_end_time = time.time()
        step_time = (step_end_time - step_start_time) * 1000  # Convert to milliseconds
        
        # Removed excessive logging
        
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

# Node 3: Check for Conflicts/Redundancy (HYBRID - Adaptive approach)
def check_conflicts(state: MemoryState) -> MemoryState:
    """
    Hybrid conflict checking with adaptive approach:
    - ≤50 memories: Use efficient bulk loading approach
    - >50 memories: Use database-first approach for scalability
    """
    
    import time
    step_start_time = time.time()
    
    if not state["validated_memories"]:
        return state
    
    validated_memories = state["validated_memories"]
    
    # Removed excessive logging
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        conflict_checked_memories = []
        
        # HYBRID APPROACH: Choose strategy based on database size
        # Check total number of memories in database to determine approach
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories_in_db = cursor.fetchone()[0]
        
        if total_memories_in_db <= 1000:  # Small database - use bulk loading
            conflict_checked_memories = _bulk_loading_conflict_check(cursor, validated_memories)
        else:  # Large database - use database-first approach
            conflict_checked_memories = _database_first_conflict_check(cursor, validated_memories)
        
        cursor.close()
        conn.close()
        
        step_end_time = time.time()
        step_time = (step_end_time - step_start_time) * 1000
        
        # Removed excessive logging
        
        return {
            **state,
            "conflict_checked_memories": conflict_checked_memories,
            "processing_log": state["processing_log"] + [f"Hybrid conflict check: {len(conflict_checked_memories)}/{len(validated_memories)} accepted"],
            "step_times": {**state.get("step_times", {}), "conflict_check": step_time}
        }
        
    except Exception as e:
        error_msg = f"Error in hybrid conflict checking: {e}"
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
    
    # Removed excessive logging
    
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
        
        # Removed excessive logging
        
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
        chain = duplicate_check_prompt | gemini_llm | JsonOutputParser()
        result = chain.invoke({
            "new_memory": new_memory_text,
            "existing_memory": existing_memory_text
        })
        
        return result.get("is_duplicate", False)
        
    except Exception as e:
        logger.warning(f"LLM duplicate check failed: {e}")
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


def _bulk_loading_conflict_check(cursor, validated_memories: List[Dict]) -> List[Dict]:
    """
    Efficient bulk loading approach for small datasets (≤50 memories).
    Single database query + in-memory processing.
    """
    conflict_checked_memories = []
    
    # Get unique characters to fetch existing memories
    unique_characters = list(set(memory['source'] for memory in validated_memories))
    
    # Single bulk query to fetch ALL existing memories for ALL characters
    # Removed excessive logging
    cursor.execute("""
        SELECT source_char, memory_text, embedding, chapter, salience
        FROM memories
        WHERE source_char = ANY(%s)
        ORDER BY source_char, chapter DESC, salience DESC
    """, (unique_characters,))
    
    # Group existing memories by character
    existing_memories_by_char = {}
    for row in cursor.fetchall():
        source_char, memory_text, embedding, chapter, salience = row
        if source_char not in existing_memories_by_char:
            existing_memories_by_char[source_char] = []
        
        # Limit to top 15 memories per character (like original logic)
        if len(existing_memories_by_char[source_char]) < 15:
            existing_memories_by_char[source_char].append((memory_text, embedding, chapter, salience))
    
    # Generate all embeddings in batch to avoid multiple API calls
    memory_texts = [memory['memory_text'] for memory in validated_memories]
    new_memory_embeddings = embeddings_model.embed_documents(memory_texts)
    
    # Process each new memory against existing ones in memory
    for i, memory in enumerate(validated_memories):
        source_char = memory['source']
        existing_memories = existing_memories_by_char.get(source_char, [])
        
        if not existing_memories:
            # No existing memories for this character
            conflict_checked_memories.append(memory)
            continue
        
        # Use pre-generated embedding
        new_memory_embedding = new_memory_embeddings[i]
        
        # Check semantic similarity with existing memories (in-memory)
        max_similarity = 0.0
        most_similar_text = ""
        most_similar_chapter = 0
        
        for existing_text, existing_embedding, existing_chapter, existing_salience in existing_memories:
            similarity = calculate_cosine_similarity(new_memory_embedding, existing_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_text = existing_text
                most_similar_chapter = existing_chapter
        
        # Conflict detection logic
        is_duplicate = False
        
        # Only reject if it's an extremely close match (near-duplicate)
        if max_similarity > 0.98:
            is_duplicate = True
            # Removed excessive logging
        
        # For high similarity but not duplicate, use LLM to make final decision
        elif max_similarity > 0.90:
            is_duplicate = check_if_duplicate_with_llm(memory['memory_text'], most_similar_text)
            if is_duplicate:
                # Removed excessive logging
                pass
            else:
                # Removed excessive logging
                pass
        
        if not is_duplicate:
            conflict_checked_memories.append(memory)
    
    return conflict_checked_memories


def _database_first_conflict_check(cursor, validated_memories: List[Dict]) -> List[Dict]:
    """
    Database-first approach for large datasets (>50 memories).
    Individual queries with vector search for scalability.
    """
    conflict_checked_memories = []
    
    # Generate all embeddings in batch to avoid multiple API calls
    memory_texts = [memory['memory_text'] for memory in validated_memories]
    new_memory_embeddings = embeddings_model.embed_documents(memory_texts)
    
    for i, memory in enumerate(validated_memories):
        source_char = memory['source']
        
        # Use pre-generated embedding
        new_memory_embedding = new_memory_embeddings[i]
        
        # Use database vector search to find the most similar memory
        cursor.execute("""
            SELECT memory_text, embedding, chapter, salience
            FROM memories
            WHERE source_char = %s
            ORDER BY embedding <=> %s::vector
            LIMIT 1
        """, (source_char, new_memory_embedding))
        
        result = cursor.fetchone()
        
        if not result:
            # No existing memories for this character
            conflict_checked_memories.append(memory)
            continue
        
        existing_text, existing_embedding, existing_chapter, existing_salience = result
        
        # Calculate similarity
        similarity = calculate_cosine_similarity(new_memory_embedding, existing_embedding)
        
        # Conflict detection logic
        is_duplicate = False
        
        # Only reject if it's an extremely close match (near-duplicate)
        if similarity > 0.98:
            is_duplicate = True
            # Removed excessive logging
        
        # For high similarity but not duplicate, use LLM to make final decision
        elif similarity > 0.90:
            is_duplicate = check_if_duplicate_with_llm(memory['memory_text'], existing_text)
            if is_duplicate:
                # Removed excessive logging
                pass
            else:
                # Removed excessive logging
                pass
        
        if not is_duplicate:
            conflict_checked_memories.append(memory)
    
    return conflict_checked_memories

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
        
        logger.info(f"Processing Chapter {chapter_data['chapter_number']}")
        
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
            
            logger.info(f"Processing Results: {results['extracted_count']} extracted, {results['validated_count']} validated, {results['conflict_checked_count']} conflict-checked, {results['inserted_count']} inserted")
            
            if results['errors']:
                logger.error(f"   Errors: {len(results['errors'])}")
                for error in results['errors']:
                    logger.error(f"      - {error}")
            
            return results
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {e}"
            logger.error(f"{error_msg}")
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
    
    logger.info(f"Test completed: {'Success' if results['success'] else 'Failed'}")
    return results

if __name__ == "__main__":
    test_langgraph_processor()
