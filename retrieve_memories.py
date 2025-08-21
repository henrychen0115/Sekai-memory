#!/usr/bin/env python3
"""
Simplified Memory Retrieval Interface for Sekai Memory System
Handles memory retrieval and scoring using MemoryManager
"""

import os
import time
from dotenv import load_dotenv
from memory_manager.memory_manager import MemoryManager

# Load environment variables
load_dotenv()

def retrieve_memories_with_custom_ranking(character_name, query, k=5):
    """
    Retrieves memories for a character using LLM agent scoring from MemoryManager
    """
    try:
        memory_manager = MemoryManager()
        
        # Start timing the retrieval process
        start_time = time.time()
        
        # Get memories with LLM agent scoring (already includes access_count in scoring)
        memories = memory_manager.retrieve_memories(character_name, query, limit=k)
        
        # Calculate retrieval time
        retrieval_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if not memories:
            return f"No relevant memories found. (Retrieval time: {retrieval_time:.2f}ms)"
        
        # Format results with LLM scores
        memory_list = []
        memory_list.append(f"⏱️ Retrieval Time: {retrieval_time:.2f}ms")
        memory_list.append("")  # Empty line for spacing
        
        for i, memory in enumerate(memories, 1):
            llm_score = memory.get('llm_score', 'N/A')
            llm_reasoning = memory.get('llm_reasoning', 'No reasoning provided')
            memory_list.append(
                f"{i}. [Score: {llm_score:.2f}] [{memory['type']}] (Salience: {memory['salience']}, Ch.{memory['chapter']}, Access: {memory.get('access_count', 0)})"
            )
            memory_list.append(f"   Text: {memory['memory_text']}")
            memory_list.append(f"   Reasoning: {llm_reasoning}")
            memory_list.append("")  # Empty line for readability
        
        return "\n".join(memory_list)
        
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return "Error retrieving memories."

# Removed LLM response generation - now only showing memory scores

def simple_memory_query(character_name, query):
    """
    Simple function to query a character's memories and show LLM agent scores
    """
    print(f"🔍 Querying {character_name} about: '{query}'")
    print(f"📋 Retrieving top 5 memories with LLM agent scoring...")
    
    # Start timing the entire query process
    start_time = time.time()
    
    # Retrieve memories with LLM agent scoring
    retrieved_memories = retrieve_memories_with_custom_ranking(character_name, query)
    
    # Calculate total query time
    total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    print(f"\n📊 Memory Scores for {character_name}:")
    print("=" * 80)
    print(retrieved_memories)
    print("=" * 80)
    print(f"⏱️ Total Query Time: {total_time:.2f}ms")
    
    return retrieved_memories

def interactive_memory_query():
    """
    Interactive memory query session
    """
    print("🎭 Sekai Memory System - Memory Query Interface")
    print("=" * 50)
    print("Available characters: Byleth, Dimitri, Sylvain, Annette, Felix, Dedue")
    print("Format: <character>: <query>")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nEnter query (e.g., 'Byleth: What happened in the office?'): ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Parse input: "Character: Query"
            if ':' not in user_input:
                print("Please use format: <character>: <query>")
                continue
            
            character_name, query = user_input.split(':', 1)
            character_name = character_name.strip()
            query = query.strip()
            
            if not character_name or not query:
                print("Please provide both character name and query")
                continue
            
            # Display memory scores
            simple_memory_query(character_name, query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        # Command line usage: python retrieve_memories.py <character> <query>
        character_name = sys.argv[1]
        query = sys.argv[2]
        simple_memory_query(character_name, query)
    else:
        interactive_memory_query()
