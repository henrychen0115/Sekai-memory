# Multi-Character Memory System Architecture

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Design](#architecture-design)
3. [Core Components](#core-components)
4. [Data Model](#data-model)
5. [Memory Processing Pipeline](#memory-processing-pipeline)
6. [Retrieval System](#retrieval-system)
7. [Evaluation Framework](#evaluation-framework)
8. [Performance Optimizations](#performance-optimizations)
9. [Technical Stack](#technical-stack)

---

## System Overview

The Multi-Character Memory System is a sophisticated AI-powered memory management system designed for interactive storytelling and character-driven narratives. It enables characters to maintain persistent, contextually relevant memories that evolve over time while preserving character-specific knowledge boundaries.

### Key Features

- **Character-Specific Memory Isolation**: Each character maintains their own memory space
- **Semantic Memory Retrieval**: Vector-based similarity search for contextually relevant memories
- **Temporal Consistency**: Chronological ordering and temporal relationship tracking
- **Cross-Talk Prevention**: Ensures characters only know what they should reasonably know
- **Memory Conflict Resolution**: Automatic detection and resolution of contradictory memories
- **Real-time Memory Updates**: Dynamic memory insertion and modification capabilities

### Design Principles

1. **Character Perspective Preservation**: Memories are stored from the character's viewpoint
2. **Semantic Relevance**: Retrieval prioritizes meaning over exact keyword matching
3. **Temporal Coherence**: Memories maintain logical chronological relationships
4. **Knowledge Boundary Enforcement**: Characters cannot access information they shouldn't know
5. **Scalability**: System designed to handle large numbers of characters and memories

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │  Processing     │    │   Storage       │
│                 │    │   Pipeline      │    │   Layer         │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Chapter Data  │───▶│ • LangGraph     │───▶│ • PostgreSQL    │
│   (Synopsis)    │    │   Workflow      │    │ • pgvector      │
│ • Character     │    │ • Memory        │    │ • Vector        │
│   Names         │    │   Extraction    │    │   Embeddings    │
│ • Memory        │    │ • Validation    │    │ • Memory        │
│   Updates       │    │ • Conflict      │    │   Metadata      │
│                 │    │   Resolution    │    │ • Access Count  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Query Layer   │───▶│   Retrieval     │
                       │                 │    │   Layer         │
                       ├─────────────────┤    ├─────────────────┤
                       │ • User Queries  │    │ • Semantic      │
                       │ • Character     │    │   Search        │
                       │   Selection     │    │ • LLM Agent     │
                       │ • Populate      │    │   Scoring       │
                       │   Database      │    │ • Character     │
                       │ • Test Scripts  │    │   Filtering     │
                       │                 │    │ • Access Count  │
                       │                 │    │   Tracking      │
                       └─────────────────┘    └─────────────────┘
```

#### **Detailed Component Breakdown:**

**Input Layer:**

- **Chapter Data**: Synopsis text processed through `langgraph_memory_processor.py`
- **Character Names**: Direct character queries (Byleth, Dimitri, Sylvain, etc.)
- **Memory Updates**: Updates to existing memories via `memory_manager.py`

**Processing Pipeline:**

- **LangGraph Workflow**: `langgraph_memory_processor.py` orchestrates the entire pipeline
- **Memory Extraction**: LLM extracts atomic memories from synopsis text
- **Validation**: Dual-LLM validation against original synopsis
- **Conflict Resolution**: Checks for duplicates and inconsistencies before insertion

**Storage Layer:**

- **PostgreSQL**: Primary database with `memories` table
- **pgvector**: Vector similarity search using `text-embedding-ada-002`
- **Memory Metadata**: Character, type, salience, chapter, access_count, created_at
- **Indexes**: B-tree on character/salience/access_count, IVFFlat on embeddings

**Retrieval Layer:**

- **Semantic Search**: Vector similarity using cosine distance
- **LLM Agent Scoring**: Multi-criteria ranking (relevance, salience, recency, access)
- **Character Filtering**: First filter by character, then semantic search
- **Access Tracking**: Automatic increment of access_count on retrieval

**Query Layer:**

- **User Queries**: Direct character + query combinations
- **Memory Retrieval Interface**: `retrieve_memories.py` for interactive memory queries
- **Test Scripts**: `test_retrieval_accuracy.py`, `test_internal_consistency.py`, `test_system_performance.py`
- **Memory Manager**: Central interface `memory_manager.py` for all operations

### Component Interaction Flow

1. **Memory Ingestion**: Chapter data → LangGraph processing → Database storage
2. **Memory Retrieval**: Query → filter based on character → Semantic search → Filtered results (LLM scoring) → Custom ranking (retrieve top K memories)
3. **Memory Updates**: Modification request → Validation → Database update
4. **Consistency Checking**: Memory set → LLM evaluation → Violation detection

---

## Core Components

### 1. LangGraph Memory Processor (`langgraph_memory_processor.py`)

The central processing engine that orchestrates the entire memory lifecycle using LangGraph's stateful workflow.

#### Key Functions:

- **Memory Extraction**: Breaks down story content into atomic memories
- **Memory Validation**: Ensures memories are accurate and properly formatted
- **Conflict Detection**: Identifies and resolves memory contradictions
- **Database Insertion**: Stores validated memories with proper metadata

#### Processing Nodes:

```
┌─────────────┐
│   __start__ │
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────┐
│              a                  │
│    Chapter Data Processing      │
│  (Input: chapter_data)          │
└─────┬───────────────────┬───────┘
      │                   │
      │ (success)         │ (error)
      ▼                   ▼
┌─────────────┐    ┌─────────────┐
│      b      │    │   __end__   │
│  Extract    │    │   (Error)   │
│ Memories    │    │             │
└─────┬───────┘    └─────────────┘
      │
      ▼
┌─────────────────────────────────┐
│              c                  │
│         Validate                │
│        Memories                 │
└─────┬───────────────────┬───────┘
      │                   │
      │ (valid)           │ (invalid)
      ▼                   ▼
┌─────────────┐    ┌─────────────┐
│      d      │    │   __end__   │
│   Check     │    │ (Rejected)  │
│ Conflicts   │    │             │
└─────┬───────┘    └─────────────┘
      │
      ▼
┌─────────────────────────────────┐
│              e                  │
│         Insert                  │
│        Memories                 │
└─────┬───────────────────┬───────┘
      │                   │
      │ (success)         │ (error)
      ▼                   ▼
┌─────────────┐    ┌─────────────┐
│   __end__   │    │   __end__   │
│ (Success)   │    │   (Error)   │
│             │    │             │
└─────────────┘    └─────────────┘
```

#### State Management:

```python
class MemoryState(TypedDict):
    chapter_data: Dict
    extracted_memories: List[Dict]
    validated_memories: List[Dict]
    inserted_count: int
    errors: List[str]
```

### 2. Memory Manager (`memory_manager.py`)

Central interface for all memory operations

#### Core Methods:

```python
class MemoryManager:
    def write_new_memories(self, chapter_data: Dict) -> Dict
    def retrieve_memories(self, character: str, query: str, limit: int = 10) -> List[Dict]
    def update_existing_memory(self, memory_id: int, new_memory_text: str) -> bool
    def list_memories_for_character(self, character: str, limit: int = 20) -> List[Dict]
    def clear_all_memories(self) -> bool
    def get_memory_statistics(self) -> Dict
```

````

### 3. Memory Retrieval Interface (`retrieve_memories.py`)

Interactive interface for character memory queries with custom ranking.

#### LLM Agent Scoring Algorithm:

```python
def _score_memories_with_llm(self, candidate_memories: List[Dict], query: str, limit: int):
    """
    Use LLM agent to score memories based on multiple criteria including access_count
    """
    # LLM scoring criteria:
    # 1. Relevance to query (40% weight): How well does the memory answer the query?
    # 2. Salience (25% weight): Higher salience = more important memory
    # 3. Recency (20% weight): Lower chapter numbers = older memories (less weight)
    # 4. Access frequency (15% weight): Higher access_count = more frequently accessed (more weight)

    # LLM agent evaluates each memory and returns scores from 0.0 to 10.0
    # Includes reasoning for each score and selects top memories
    return scored_memories_with_llm_scores
```

**Key Features:**
- **Access Count Integration**: Access frequency is weighted at 15% in the scoring algorithm
- **Multi-Criteria Scoring**: Combines relevance, salience, recency, and access frequency
- **LLM Reasoning**: Provides explanations for why each memory was scored as it was
- **Automatic Updates**: Access count is incremented every time a memory is retrieved`

---

## Data Model

### Database Schema

```sql
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    source_char VARCHAR(100) NOT NULL,
    target_char VARCHAR(100),
    memory_text TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    salience FLOAT DEFAULT 0.5,
    chapter INTEGER NOT NULL,
    embedding vector(1536),
    access_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_memories_source_char ON memories(source_char);
CREATE INDEX idx_memories_chapter ON memories(chapter);
CREATE INDEX idx_memories_salience ON memories(salience);
CREATE INDEX idx_memories_access_count ON memories(access_count);
CREATE INDEX idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops);
```

### Memory Structure

```python
Memory = {
    'id': int,                    # Database ID
    'source_char': str,           # Character who owns the memory
    'target_char': str,           # Character the memory is about
    'memory_text': str,           # The actual memory content
    'type': str,                  # Memory type (observation, interaction, etc.)
    'salience': float,            # Importance score (0.0-1.0)
    'chapter': int,               # Story chapter number
    'embedding': List[float],     # Vector representation (1536 dimensions)
    'access_count': int,          # Number of times memory was accessed
    'created_at': datetime,       # Creation timestamp
    'llm_score': float,           # LLM agent scoring (0.0-10.0) - added during retrieval
    'llm_reasoning': str          # LLM reasoning for score - added during retrieval
}
```

### Memory Types

- **Observation**: Character observes something happening
- **Interaction**: Direct interaction between characters
- **Emotional**: Character's emotional response or feeling
- **Knowledge**: Factual information learned
- **Decision**: Character's decision or choice
- **Reflection**: Character's thoughts or introspection

---

## Memory Processing Pipeline

### 1. Memory Extraction

#### LangGraph Node A: Chapter Data Processing

```python
def process_chapter_data(state: MemoryState) -> MemoryState:
    """Process chapter data and extract initial memories"""
    chapter_data = state["chapter_data"]

    # Extract memories using LLM
    extraction_prompt = f"""
    Extract atomic memories from this chapter synopsis:
    Chapter {chapter_data['chapter_number']}: {chapter_data['synopsis']}

    Format each memory as: [Character] [Action/Observation] [Other Characters/Context]
    """

    # Process with LLM and extract structured memories
    return state
```

#### LangGraph Node B: Memory Extraction

```python
def extract_memories(state: MemoryState) -> MemoryState:
    """Extract atomic memories from chapter content"""
    chapter_data = state["chapter_data"]

    # Use LLM to break down synopsis into atomic memories
    # Each memory follows: subject -> action -> other characters format
    # Validate character perspectives and knowledge boundaries

    return state
```

### 2. Memory Validation

#### LangGraph Node C: Validate Memories

```python
def validate_memories(state: MemoryState) -> MemoryState:
    """Validate extracted memories against source material"""
    extracted_memories = state["extracted_memories"]

    # Check each memory against the original synopsis
    # Ensure no hallucinated content
    # Verify character knowledge boundaries
    # Validate temporal consistency

    return state
```

### 3. Conflict Resolution

#### LangGraph Node D: Check Conflicts

```python
def check_conflicts(state: MemoryState) -> MemoryState:
    """Check for conflicts with existing memories"""
    validated_memories = state["validated_memories"]

    # For each new memory, check against character's existing memories
    # Use vector similarity to detect potential duplicates
    # Use LLM to evaluate conflicts and contradictions
    # Reject or modify conflicting memories

    return state
```

### 4. Database Insertion

#### LangGraph Node E: Insert Memories

```python
def insert_memories(state: MemoryState) -> MemoryState:
    """Insert validated memories into database"""
    validated_memories = state["validated_memories"]

    # Generate embeddings for each memory
    # Insert into PostgreSQL with pgvector
    # Update insertion count and statistics

    return state
```

---

## Retrieval System

### Semantic Search Implementation

```python
def retrieve_memories(self, character: str, query: str, limit: int = 10):
    # Generate query embedding
    query_embedding = self.embeddings_model.embed_query(query)

    # Database query with semantic similarity
    cursor.execute("""
        SELECT id, source_char, target_char, memory_text, memory_type, salience, chapter, access_count
        FROM memories
        WHERE source_char = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (character, query_embedding, limit))

    # Update access_count for retrieved memories
    if memories:
        memory_ids = [str(memory['id']) for memory in memories]
        cursor.execute("""
            UPDATE memories
            SET access_count = access_count + 1
            WHERE id = ANY(%s::uuid[])
        """, (memory_ids,))

    return memories
```

### LLM Agent Scoring System

The system implements an intelligent LLM agent to score and select the most relevant memories from vector search results.

#### Retrieval Process:

1. **Vector Search**: Retrieve candidate memories using semantic similarity
2. **Access Tracking**: Update `access_count` for all retrieved memories
3. **LLM Scoring**: Use LLM agent to score memories based on multiple criteria
4. **Top Selection**: Return top-scored memories with detailed reasoning

#### LLM Scoring Criteria:

```python
# Scoring weights for LLM agent
scoring_criteria = {
    'relevance_to_query': 0.40,    # How well memory answers the query
    'salience': 0.25,              # Memory importance (higher = better)
    'recency': 0.20,               # Chapter number (lower = older, less weight)
    'access_frequency': 0.15       # Access count (higher = more popular)
}
```

#### LLM Agent Implementation:

```python
def _score_memories_with_llm(self, candidate_memories: List[Dict], query: str, limit: int):
    """Use LLM agent to score memories based on multiple criteria"""

    # Prepare memory data for LLM evaluation
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

    # LLM scoring prompt with detailed criteria
    scoring_prompt = f"""
    Score each memory from 0.0 to 10.0 based on:
    1. Relevance to query (40%): How well does it answer "{query}"?
    2. Salience (25%): Higher salience = more important
    3. Recency (20%): Lower chapter = older (less weight)
    4. Access frequency (15%): Higher access_count = more popular

    Return JSON: [{{"index": 0, "score": 8.5, "reasoning": "..."}}]
    """

    # Parse LLM response and return top memories
    return scored_memories[:limit]
```

#### Fallback Scoring:

When LLM scoring fails, the system uses a mathematical fallback:

```python
def _fallback_scoring(self, candidate_memories: List[Dict], limit: int):
    """Fallback scoring when LLM is unavailable"""

    for memory in candidate_memories:
        # Simple formula: (salience * 0.4) + (chapter * 0.3) + (access_count * 0.3)
        score = (memory['salience'] * 0.4) + (memory['chapter'] * 0.3) + (memory['access_count'] * 0.3)
        memory['llm_score'] = score
        memory['llm_reasoning'] = f"Fallback: salience({memory['salience']:.2f}) + chapter({memory['chapter']}) + access({memory['access_count']})"

    return sorted(candidate_memories, key=lambda x: x['llm_score'], reverse=True)[:limit]
```

### Character-First Filtering

The system implements character-first filtering to optimize performance:

1. **Filter by Character**: `WHERE source_char = %s` - Reduces search space
2. **Semantic Search**: `ORDER BY embedding <=> %s::vector` - Finds relevant memories
3. **Limit Results**: `LIMIT %s` - Controls response size

This approach ensures:

- **Performance**: Character filtering reduces vector search space by ~90%
- **Security**: Characters can only access their own memories
- **Accuracy**: Semantic search within character's knowledge domain

---

## Evaluation Framework

### 1. Retrieval Accuracy Testing (`test_retrieval_accuracy.py`)

**Purpose**: Measure precision and recall of memory retrieval

#### Test Process:

1. **Query Generation**: Create test queries for each character
2. **Memory Retrieval**: Get top 5 memories for each query
3. **LLM Evaluation**: Use LLM as judge to evaluate relevance
4. **Metrics Calculation**: Calculate precision and recall scores

#### Evaluation Metrics:

```python
# Precision: How many retrieved memories are relevant?
precision = relevant_retrieved / total_retrieved

# Recall: How many relevant memories were retrieved?
recall = relevant_retrieved / total_relevant

# Relevance Score: LLM-judged relevance (0-10 scale)
relevance_score = llm_evaluation_score / 10
```

### 2. Internal Consistency Testing (`test_internal_consistency.py`)

**Purpose**: Detect temporal, knowledge, and world state violations

#### Violation Types:

- **T (Temporal Issues)**: Chronological inconsistencies
- **K (Knowledge Violations)**: Characters knowing impossible information
- **W (World State Conflicts)**: Contradictions with established facts
- **F (Forgotten Updates)**: Outdated information
- **C (Contradictions)**: Direct memory contradictions

#### Test Process:

1. Retrieve all memories for character (chronological order)
2. LLM evaluation against expected knowledge boundaries
3. Violation detection and categorization
4. Consistency scoring and reporting

### 3. System Performance Testing (`test_system_performance.py`)

**Purpose**: Measure system efficiency and scalability

#### Performance Metrics:

- **Retrieval Speed**: Response time for memory queries
- **Insertion Time**: Time to process and store new memories
- **Step-by-step Timing**: Detailed breakdown of processing steps
- **Scalability**: Performance with large memory sets

#### Test Categories:

- **Retrieval Speed**: Different query complexities
- **Insertion Time**: Memory processing pipeline timing
- **Memory Deduplication**: Conflict resolution efficiency

---

## Performance Optimizations

### Database Optimization

#### Indexing Strategy

- **Vector Index**: IVFFlat for semantic search performance
- **Character Index**: B-tree for character-specific queries
- **Temporal Index**: B-tree for chronological ordering
- **Composite Indexes**: For complex query patterns

#### Query Optimization

```sql
-- Optimized retrieval query
SELECT memory_text, salience, memory_type
FROM memories
WHERE source_char = %s
  AND embedding <=> %s < 0.8  -- Similarity threshold
ORDER BY embedding <=> %s
LIMIT %s;
```

### Memory Management

#### Embedding Storage

- **Vector Dimensions**: 1536 (OpenAI text-embedding-ada-002)
- **Storage Format**: PostgreSQL vector type
- **Compression**: Automatic by pgvector extension

#### Character-First Filtering Strategy

- **Primary Filter**: Filter by character before vector search
- **Reduced Search Space**: Only search within character's memories
- **Performance Gain**: Significant reduction in vector search time

**Implementation Details:**

```python
# Character-first filtering
cursor.execute("""
    SELECT id, source_char, target_char, memory_text, memory_type, salience, chapter, access_count
    FROM memories
    WHERE source_char = %s  # Primary filter by character
    ORDER BY embedding <=> %s::vector  # Then vector similarity
    LIMIT %s
""", (character, query_embedding, limit))
```

**Performance Impact:**

- **Query Time**: ~50-200ms (embedding generation + filtered vector search)
- **Search Space**: Reduced by ~90% (only character's memories)
- **Scalability**: Linear performance with character memory count

### Scalability Considerations


### Monitoring and Metrics

#### Key Performance Indicators

- **Query Response Time**: Target < 500ms for queries
- **Insertion Throughput**: Target > 100 memories/minute
- **Memory Usage**: Monitor embedding storage size

#### Logging and Debugging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance logging
logger.info(f"Query completed in {response_time}ms")
logger.info(f"Retrieved {memory_count} memories")
```

---

## Technical Stack

### Core Technologies

#### Backend Framework

- **Python 3.8+**: Primary programming language
- **LangGraph**: Stateful workflow orchestration

#### Database

- **PostgreSQL 13+**: Primary relational database
- **pgvector**: Vector similarity search extension

#### AI/ML

- **OpenAI text-embedding-ada-002**: Text embeddings (1536 dimensions)
- **Google Gemini 2.5 Flash Lite**: LLM for processing and evaluation
- **Google Gemini 2.0 Flash Lite**: LLM for faster processing without reasoning
- **OpenRouter**: API proxy for LLM access

#### Performance Optimization

- **Character-First Filtering**: Reduce search space by filtering by character
- **Database Indexing**: Optimized indexes for fast queries

### Development Tools

#### Testing Framework

- **Custom Test Scripts**: Specialized evaluation scripts
- **LLM-as-Judge**: Automated evaluation using AI
- **Performance Benchmarking**: Custom timing and metrics

#### Monitoring

- **Custom Logging**: Performance and error tracking
- **Database Analytics**: Query performance monitoring

### Architecture Patterns

### Key Optimizations

- **Semantic Understanding**: Deep comprehension of memory content
- **Character Isolation**: Strong boundaries between character knowledge
- **Temporal Coherence**: Logical chronological relationships
- **Scalable Design**: Handles large datasets efficiently
- **Comprehensive Testing**: Multi-dimensional evaluation framework

### Current System Improvements

#### 1. Dynamic LLM Selection via OpenRouter

**Implementation**: The system uses OpenRouter for dynamic agent choosing, allowing optimal model selection for different tasks.

**Benefits**:

- **Task-Specific Optimization**: Choose expensive/reasoning models for complex tasks (memory extraction, conflict resolution)
- **Cost Efficiency**: Use cheaper models for simpler tasks (validation, basic queries)
- **Performance Optimization**: Select fastest, most suitable, and token-efficient LLM for each task
- **Flexibility**: Easy switching between models without code changes

**Current Model Assignments**:

```python
# Complex Tasks (Reasoning Models)
- Memory Extraction: google/gemini-2.5-flash-lite
- Conflict Resolution: google/gemini-2.5-flash-lite
- LLM-as-Judge Evaluation: google/gemini-2.5-flash-lite

# Simple Tasks (Efficient Models)
- Validation: gpt-3.5-turbo
- Basic Queries: gpt-3.5-turbo
- Embedding Generation: text-embedding-ada-002
```

#### 2. LangGraph Integration

**Migration from LangChain**: Replaced LangChain with LangGraph for enhanced functionality.

**Key Improvements**:

- **State Management**: Global state storage across workflow nodes
- **Better Error Handling**: Comprehensive error logging and recovery
- **LangSmith Integration**: Advanced monitoring and debugging capabilities
- **Workflow Orchestration**: Stateful, multi-step processing with conditional flows

#### 3. Enhanced Atomic Memory Validation

**Dual-LLM Validation System**: Added dedicated validation agent to check processed atomic memories against full synopsis to minimize hallucinations.

#### 4. LLM Agent Memory Scoring

**Intelligent Memory Selection**: Implemented LLM agent to score and select the most relevant memories from vector search results.

**Key Features**:

- **Multi-criteria Scoring**: Relevance (40%), Salience (25%), Recency (20%), Access Frequency (15%)
- **Access Tracking**: Automatic `access_count` updates for all retrieved memories
- **Detailed Reasoning**: LLM provides reasoning for each memory score
- **Fallback System**: Mathematical scoring when LLM is unavailable

**Implementation Benefits**:

- **Improved Relevance**: Better memory selection based on query context
- **Usage Analytics**: Track which memories are accessed most frequently
- **Transparency**: Detailed scoring explanations for debugging
- **Robustness**: Graceful degradation when LLM scoring fails


````
