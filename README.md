# Sekai Memory System

A multi-character memory management system built with PostgreSQL, pgvector, LangGraph, and LLMs. This system stores, retrieves, and manages character-specific memories with semantic search capabilities.

## 📁 Project Structure

```
Sekai-memory/
├── memory_manager/           # Core memory management functionality
│   ├── __init__.py          # Package initialization
│   ├── memory_manager.py    # Main memory manager class
│   └── langgraph_memory_processor.py  # LangGraph workflow processor
├── utils/                   # Utility scripts
│   ├── clear_database.py    # Database clearing utility
│   └── database_stats.py    # Database statistics and reporting
├── tests/                   # System testing and evaluation
│   ├── test_internal_consistency.py
│   ├── test_retrieval_accuracy.py
│   └── test_system_performance.py
├── docs/                    # Documentation
│   └── MEMORY_SYSTEM_ARCHITECTURE.md
├── logs/                    # Application logs
├── retrieve_memories.py     # Memory retrieval interface
├── populate_database.py     # Database population script
└── docker-compose.yml       # Docker configuration
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- OpenAI API key or OpenRouter API key

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Sekai-memory
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Override default settings
DB_HOST=postgres
DB_NAME=sekai_memory
DB_USER=memory_user
DB_PASSWORD=memory_password
```

### 3. Start the System

```bash
# Build and start all containers
docker-compose up --build -d

# Check container status
docker-compose ps
```

You should see three containers running:

- `memory_postgres` (PostgreSQL database)
- `memory_system_app` (Python application)
- `memory_pgadmin` (Database management interface)

## 📊 System Architecture

### Services

- **PostgreSQL + pgvector**: Vector database for semantic memory storage
- **Memory System App**: Python application with LangGraph workflow
- **pgAdmin**: Web-based database management interface

### Key Features

- **Atomic Memory Generation**: Breaks down story content into character-specific memories
- **Semantic Search**: Vector-based memory retrieval using embeddings
- **LLM Agent Scoring**: Intelligent memory ranking based on relevance, salience, and recency
- **Conflict Resolution**: Deduplication and consistency checking
- **Access Tracking**: Memory usage monitoring and scoring

## 🔧 API Routes and Functions

### Core Memory Manager Functions

The `MemoryManager` class provides the following key functions:

#### Memory Operations

- `write_new_memories(chapter_data)` - Write new memories using LangGraph workflow
- `update_existing_memory(memory_id, new_memory_text)` - Update memory by ID
- `update_memory_by_character_and_content(character, old_content, new_content)` - Update memory by content
- `retrieve_memories(character, query, limit=10)` - Retrieve memories with LLM scoring
- `get_memory_by_id(memory_id)` - Get specific memory by ID

#### Database Management

- `list_memories_for_character(character, limit=20)` - List all memories for a character
- `get_database_stats()` - Get comprehensive database statistics
- `clear_all_memories()` - Clear all memories from database

### LangGraph Memory Processor

The `LangGraphMemoryProcessor` class handles the workflow:

- `process_chapter(chapter_data)` - Process chapter data through the complete workflow
- Workflow nodes: Extract → Validate → Conflict Check → Insert

### Utility Functions

#### Database Utilities (`utils/`)

- `clear_database()` - Clear all database contents
- `get_database_statistics()` - Get detailed database statistics

#### Testing Functions (`tests/`)

- `test_retrieval_accuracy()` - Test memory retrieval precision/recall
- `test_internal_consistency()` - Test for memory contradictions
- `test_system_performance()` - Performance benchmarking

## 🗄️ Database Management

### Access pgAdmin Web Interface

1. Open browser: `http://localhost:8080`
2. Login: `admin@admin.com` / `admin`
3. Add server:
   - **General**: Name = "Memory System DB"
   - **Connection**:
     - Host = `postgres`
     - Port = `5432`
     - Database = `sekai_memory`
     - Username = `memory_user`
     - Password = `memory_password`

### Direct Database Connection

```bash
# Connect to PostgreSQL container
docker-compose exec memory_postgres psql -U memory_user -d memory_system

# Check database stats
docker-compose exec memory_system python -c "from memory_manager.memory_manager import MemoryManager; print(MemoryManager().get_database_stats())"
```

## 📚 Populate Database

There are a total of 50 chapters pre-stored in memory_data.json, typing chapter range 1-50 will load all the pre-saved stories. Make sure to populate the database first if the database is empty.

### Interactive Mode

```bash
docker-compose exec memory_system python populate_database.py
```

**Input Formats:**

1. **Chapter Range**: `1-20` (processes chapters 1 through 20)
2. **Single Chapter**: `5` (processes only chapter 5)
3. **Custom Chapter**: `25,This is a custom synopsis`

### Example Usage

```bash
# Process chapters 1-5
echo "1-5" | docker-compose exec -T memory_system python populate_database.py

# Process single chapter
echo "3" | docker-compose exec -T memory_system python populate_database.py
```

### Processing Workflow

1. **Extract**: Breaks synopsis into atomic memories
2. **Validate**: Checks memories against original synopsis
3. **Conflict Check**: Removes duplicates and inconsistencies
4. **Insert**: Stores valid memories in database

## 🔍 Memory Retrieval

Make sure to populate database first if database is empty.

### Interactive Mode

```bash
docker-compose exec memory_system python retrieve_memories.py
```

**Format**: `<character>: <query>`
**Example**: `Byleth: office meeting with Dimitri`

### Direct Query

```bash
docker-compose exec memory_system python retrieve_memories.py "Byleth" "office meeting"
```

### Available Characters

- Byleth, Dimitri, Sylvain, Annette, Felix, Dedue

### Output Format

```
🔍 Querying Byleth about: 'office meeting'
📋 Retrieving top 5 memories with LLM agent scoring...

📊 Memory Scores for Byleth:
================================================================================
⏱️ Retrieval Time: 1150.00ms

1. [Score: 8.50] [C2U] (Salience: 8.0, Ch.3, Access: 2)
   Text: I approached Dimitri's desk after hours...
   Reasoning: Highly relevant as it describes an office interaction...

2. [Score: 6.20] [WM] (Salience: 7.0, Ch.1, Access: 1)
   Text: I observed the office ecosystem...
   Reasoning: Relevant office context...
================================================================================
⏱️ Total Query Time: 1163.62ms
```

## 🧪 Testing and Evaluation

### 1. Retrieval Accuracy Test

For these tests, it will work better if you've populated at least 20 chapters worth of Synopsis.

Tests precision and recall of memory retrieval:

```bash
docker-compose exec memory_system python test_retrieval_accuracy.py
```

**What it tests:**

- Memory relevance to queries
- Precision/recall metrics
- LLM scoring quality
- Character-specific knowledge boundaries

### 2. Internal Consistency Test

Checks for memory contradictions and inconsistencies:

```bash
docker-compose exec memory_system python test_internal_consistency.py
```

**What it tests:**

- Temporal consistency (T)
- Knowledge consistency (K)
- World state consistency (W)
- Forgotten updates (F)
- Contradictions (C)

### 3. System Performance Test

Measures insertion and retrieval performance:

```bash
docker-compose exec memory_system python test_system_performance.py
```

**What it tests:**

- Memory insertion time
- Query retrieval time
- Performance across different query complexities

## 🔧 Management Commands

### Container Management

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs memory_system
docker-compose logs memory_postgres

# Restart specific service
docker-compose restart memory_system
```

### Database Management

```bash
# Clear all memories
docker-compose exec memory_system python utils/clear_database.py

# Get comprehensive database statistics
docker-compose exec memory_system python utils/database_stats.py

# Check basic database stats
docker-compose exec memory_system python -c "from memory_manager.memory_manager import MemoryManager; mm = MemoryManager(); print(mm.get_database_stats())"

# List memories for character
docker-compose exec memory_system python -c "from memory_manager.memory_manager import MemoryManager; mm = MemoryManager(); print(mm.list_memories_for_character('Byleth', limit=10))"
```
