#!/usr/bin/env python3
"""
Central Memory Manager
Provides core functionalities for the Sekai Memory System:
- Write new memories (using LangGraph workflow)
- Update existing memories
- Retrieve memories based on context
"""

import os
import json
import uuid
import psycopg2
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph_memory_processor import LangGraphMemoryProcessor

# Load environment variables
load_dotenv()

class MemoryManager:
    """Central memory management system for the Sekai Memory System"""
    
    def __init__(self):
        """Initialize the memory manager with database connection and LangGraph processor"""
        self.processor = LangGraphMemoryProcessor()
        
        # Initialize LLM and embeddings for retrieval
        self.llm = ChatOpenAI(
            model="google/gemini-2.5-flash-lite",
            temperature=0,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"
        )
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Database connection parameters
        self.db_config = {
            'host': os.getenv("DB_HOST"),
            'dbname': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD")
        }
    
    def _get_db_connection(self):
        """Get database connection with pgvector support"""
        conn = psycopg2.connect(**self.db_config)
        register_vector(conn)
        return conn
    
    def write_new_memories(self, chapter_data: Dict) -> Dict:
        """
        Write new memories using LangGraph workflow
        
        Args:
            chapter_data: Dictionary with 'chapter_number' and 'synopsis'
            
        Returns:
            Dict: Processing results with counts and status
        """
        print(f"📝 Writing new memories for Chapter {chapter_data['chapter_number']}...")
        
        try:
            # Use LangGraph processor to handle the entire workflow
            result = self.processor.process_chapter(chapter_data)
            
            print(f"✅ Memory writing completed:")
            print(f"   Extracted: {result['extracted_count']}")
            print(f"   Validated: {result['validated_count']}")
            print(f"   Inserted: {result['inserted_count']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error writing memories: {e}")
            return {
                'success': False,
                'extracted_count': 0,
                'validated_count': 0,
                'inserted_count': 0,
                'errors': [str(e)]
            }
    
    def update_existing_memory(self, memory_id: int, new_memory_text: str) -> bool:
        """
        Update an existing memory by ID
        
        Args:
            memory_id: Database ID of the memory to update
            new_memory_text: New memory text content
            
        Returns:
            bool: Success status
        """
        print(f"🔄 Updating memory ID {memory_id}...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Get the existing memory to preserve other fields
            cursor.execute("SELECT * FROM memories WHERE id = %s", (memory_id,))
            existing_memory = cursor.fetchone()
            
            if not existing_memory:
                print(f"❌ Memory ID {memory_id} not found")
                return False
            
            # Generate new embedding for the updated text
            new_embedding = self.embeddings_model.embed_query(new_memory_text)
            
            # Update the memory
            cursor.execute("""
                UPDATE memories 
                SET memory_text = %s, embedding = %s, updated_at = NOW()
                WHERE id = %s
            """, (new_memory_text, new_embedding, memory_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"✅ Successfully updated memory ID {memory_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating memory: {e}")
            return False
    
    def update_memory_by_character_and_content(self, character: str, old_content: str, new_content: str) -> bool:
        """
        Update a memory by finding it using character and content
        
        Args:
            character: Source character name
            old_content: Current memory text to match
            new_content: New memory text
            
        Returns:
            bool: Success status
        """
        print(f"🔄 Updating memory for {character}...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Find the memory by character and content
            cursor.execute("""
                SELECT id FROM memories 
                WHERE source_char = %s AND memory_text = %s
            """, (character, old_content))
            
            memory = cursor.fetchone()
            
            if not memory:
                print(f"❌ Memory not found for {character} with content: {old_content[:50]}...")
                return False
            
            memory_id = memory[0]
            
            # Generate new embedding
            new_embedding = self.embeddings_model.embed_query(new_content)
            
            # Update the memory
            cursor.execute("""
                UPDATE memories 
                SET memory_text = %s, embedding = %s, updated_at = NOW()
                WHERE id = %s
            """, (new_content, new_embedding, memory_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"✅ Successfully updated memory ID {memory_id} for {character}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating memory: {e}")
            return False
    
    def retrieve_memories(self, character: str, query: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve memories for a specific character based on context query with LLM agent scoring
        
        Args:
            character: Character to retrieve memories for
            query: Context query for semantic search
            limit: Maximum number of memories to return
            
        Returns:
            List[Dict]: List of memory dictionaries with LLM agent scores
        """
        print(f"🔍 Retrieving memories for {character} with query: '{query}'...")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_model.embed_query(query)
            
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # First, retrieve more memories than needed for LLM agent selection
            retrieval_limit = max(limit * 3, 20)  # Get at least 20 memories for LLM to choose from
            
            # Retrieve memories using semantic similarity with access_count
            cursor.execute("""
                SELECT id, source_char, target_char, memory_text, memory_type, salience, chapter, access_count
                FROM memories 
                WHERE source_char = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (character, query_embedding, retrieval_limit))
            
            candidate_memories = []
            for row in cursor.fetchall():
                memory = {
                    'id': row[0],
                    'source_char': row[1],
                    'target_char': row[2],
                    'memory_text': row[3],
                    'type': row[4],
                    'salience': row[5],
                    'chapter': row[6],
                    'access_count': row[7]
                }
                candidate_memories.append(memory)
            
            # Update access_count for all retrieved memories
            if candidate_memories:
                memory_ids = [memory['id'] for memory in candidate_memories]
                # Convert UUIDs to proper format for PostgreSQL
                cursor.execute("""
                    UPDATE memories 
                    SET access_count = access_count + 1 
                    WHERE id = ANY(%s::uuid[])
                """, (memory_ids,))
                conn.commit()
            
            cursor.close()
            conn.close()
            
            # Use LLM agent to score and select top memories
            scored_memories = self._score_memories_with_llm(candidate_memories, query, limit)
            
            print(f"✅ Retrieved {len(scored_memories)} memories for {character}")
            return scored_memories
            
        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
            return []
    
    def _score_memories_with_llm(self, candidate_memories: List[Dict], query: str, limit: int) -> List[Dict]:
        """
        Use LLM agent to score memories based on salience, chapter, and access_count
        
        Args:
            candidate_memories: List of candidate memories from vector search
            query: Original search query
            limit: Number of top memories to return
            
        Returns:
            List of top memories with LLM agent scores
        """
        if not candidate_memories:
            return []
        

        
        # Prepare memory data for LLM
        memory_data = []
        for i, memory in enumerate(candidate_memories):
            memory_data.append({
                'index': i,
                'memory_text': memory['memory_text'],
                'salience': memory['salience'],
                'chapter': memory['chapter'],
                'access_count': memory['access_count'],
                'type': memory['type']
            })
        
        # Create LLM scoring prompt
        scoring_prompt = f"""
        You are an intelligent memory selection agent. Given a query and a list of candidate memories, 
        score each memory from 0.0 to 10.0 based on relevance to the query and memory quality.
        
        Query: "{query}"
        
        Scoring criteria:
        1. Relevance to query (40% weight): How well does the memory answer the query?
        2. Salience (25% weight): Higher salience = more important memory
        3. Recency (20% weight): Lower chapter numbers = older memories (less weight)
        4. Access frequency (15% weight): Higher access_count = more frequently accessed (more weight)
        
        Memory data:
        {json.dumps(memory_data, indent=2)}
        
        For each memory, provide:
        1. A score from 0.0 to 10.0
        2. Brief reasoning for the score
        
        Return your response as a JSON array with objects containing:
        {{"index": memory_index, "score": float_score, "reasoning": "brief explanation"}}
        
        Select the top {limit} memories with the highest scores.
        """
        
        try:
            # Get LLM response
            response = self.llm.invoke(scoring_prompt)
            response_text = response.content
            
            # Parse LLM response
            try:
                # Extract JSON from response
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_str = response_text[json_start:json_end]
                    scored_indices = json.loads(json_str)
                else:
                    # Fallback: try to parse the entire response
                    scored_indices = json.loads(response_text)
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse LLM response, using fallback scoring")
                scored_indices = self._fallback_scoring(candidate_memories, limit)
            
            # Sort by score and get top memories
            scored_indices.sort(key=lambda x: x['score'], reverse=True)
            top_indices = scored_indices[:limit]
            
            # Create final result with scores
            final_memories = []
            for scored_item in top_indices:
                memory_index = scored_item['index']
                if memory_index < len(candidate_memories):
                    memory = candidate_memories[memory_index].copy()
                    memory['llm_score'] = scored_item['score']
                    memory['llm_reasoning'] = scored_item.get('reasoning', 'No reasoning provided')
                    final_memories.append(memory)
                    

            
            return final_memories
            
        except Exception as e:
            print(f"⚠️ LLM scoring failed: {e}, using fallback scoring")
            return self._fallback_scoring(candidate_memories, limit)
    
    def _fallback_scoring(self, candidate_memories: List[Dict], limit: int) -> List[Dict]:
        """
        Fallback scoring method when LLM scoring fails
        
        Args:
            candidate_memories: List of candidate memories
            limit: Number of top memories to return
            
        Returns:
            List of top memories with fallback scores
        """

        
        scored_memories = []
        for i, memory in enumerate(candidate_memories):
            # Simple scoring formula: (salience * 0.4) + (chapter * 0.3) + (access_count * 0.3)
            score = (memory['salience'] * 0.4) + (memory['chapter'] * 0.3) + (memory['access_count'] * 0.3)
            
            memory_copy = memory.copy()
            memory_copy['llm_score'] = score
            memory_copy['llm_reasoning'] = f"Fallback score: salience({memory['salience']:.2f}) + chapter({memory['chapter']}) + access({memory['access_count']})"
            scored_memories.append(memory_copy)
            

        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x['llm_score'], reverse=True)
        return scored_memories[:limit]
    
    def get_memory_by_id(self, memory_id: int) -> Optional[Dict]:
        """
        Get a specific memory by its database ID
        
        Args:
            memory_id: Database ID of the memory
            
        Returns:
            Optional[Dict]: Memory dictionary or None if not found
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, source_char, target_char, memory_text, memory_type, salience, chapter, access_count
                FROM memories 
                WHERE id = %s
            """, (memory_id,))
            
            row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'source_char': row[1],
                    'target_char': row[2],
                    'memory_text': row[3],
                    'type': row[4],
                    'salience': row[5],
                    'chapter': row[6],
                    'access_count': row[7]
                }
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error getting memory by ID: {e}")
            return None
    
    def list_memories_for_character(self, character: str, limit: int = 20) -> List[Dict]:
        """
        List all memories for a specific character
        
        Args:
            character: Character name
            limit: Maximum number of memories to return
            
        Returns:
            List[Dict]: List of memory dictionaries
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, source_char, target_char, memory_text, memory_type, salience, chapter, access_count
                FROM memories 
                WHERE source_char = %s
                ORDER BY chapter DESC
                LIMIT %s
            """, (character, limit))
            
            memories = []
            for row in cursor.fetchall():
                memory = {
                    'id': row[0],
                    'source_char': row[1],
                    'target_char': row[2],
                    'memory_text': row[3],
                    'type': row[4],
                    'salience': row[5],
                    'chapter': row[6],
                    'access_count': row[7]
                }
                memories.append(memory)
            
            cursor.close()
            conn.close()
            
            print(f"📋 Listed {len(memories)} memories for {character}")
            return memories
            
        except Exception as e:
            print(f"❌ Error listing memories: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dict: Database statistics
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Total memories
            cursor.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]
            
            # Memories by character with access statistics
            cursor.execute("""
                SELECT source_char, COUNT(*), AVG(access_count), SUM(access_count)
                FROM memories 
                GROUP BY source_char 
                ORDER BY COUNT(*) DESC
            """)
            character_data = cursor.fetchall()
            character_counts = {}
            character_access_stats = {}
            for row in character_data:
                character_counts[row[0]] = row[1]
                character_access_stats[row[0]] = {
                    'avg_access': round(row[2], 2) if row[2] else 0,
                    'total_access': row[3] if row[3] else 0
                }
            
            # Memories by type
            cursor.execute("""
                SELECT memory_type, COUNT(*) 
                FROM memories 
                GROUP BY memory_type 
                ORDER BY COUNT(*) DESC
            """)
            type_counts = dict(cursor.fetchall())
            
            cursor.close()
            conn.close()
            
            return {
                'total_memories': total_memories,
                'character_counts': character_counts,
                'type_counts': type_counts,
                'character_access_stats': character_access_stats
            }
            
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}
    
    def clear_all_memories(self) -> bool:
        """
        Clear all memories from the database
        
        Returns:
            bool: Success status
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM memories")
            deleted_count = cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"🗑️ Cleared {deleted_count} memories from database")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing memories: {e}")
            return False

def test_memory_manager():
    """Test the memory manager functionality"""
    print("🧪 Testing Memory Manager")
    print("=" * 50)
    
    # Create memory manager
    manager = MemoryManager()
    
    # Test database stats
    print("\n📊 Database Statistics:")
    stats = manager.get_database_stats()
    print(f"   Total memories: {stats.get('total_memories', 0)}")
    print(f"   Character counts: {stats.get('character_counts', {})}")
    
    # Test writing new memories
    print("\n📝 Testing memory writing:")
    test_chapter = {
        'chapter_number': 999,
        'synopsis': 'Test chapter for memory manager. Byleth and Dimitri have a brief conversation about work.'
    }
    result = manager.write_new_memories(test_chapter)
    print(f"   Write result: {result['success']}")
    
    # Test memory retrieval
    print("\n🔍 Testing memory retrieval:")
    memories = manager.retrieve_memories('Byleth', 'conversation with Dimitri', limit=5)
    print(f"   Retrieved {len(memories)} memories")
    
    # Test memory listing
    print("\n📋 Testing memory listing:")
    character_memories = manager.list_memories_for_character('Byleth', limit=5)
    print(f"   Listed {len(character_memories)} memories for Byleth")
    
    print("\n✅ Memory manager test completed")

if __name__ == "__main__":
    test_memory_manager()
