# Sekai Memory System

A multi-character memory management system built with PostgreSQL, pgvector, LangGraph, and LLMs. This system stores, retrieves, and manages character-specific memories with semantic search capabilities.

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
DB_NAME=memory_system
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

## 🗄️ Database Management

### Access pgAdmin Web Interface

1. Open browser: `http://localhost:8080`
2. Login: `admin@admin.com` / `admin`
3. Add server:
   - **General**: Name = "Memory System DB"
   - **Connection**:
     - Host = `postgres`
     - Port = `5432`
     - Database = `memory_system`
     - Username = `memory_user`
     - Password = `memory_password`

### Direct Database Connection

```bash
# Connect to PostgreSQL container
docker-compose exec memory_postgres psql -U memory_user -d memory_system

# Check database stats
docker-compose exec memory_system python -c "from memory_manager import MemoryManager; print(MemoryManager().get_database_stats())"
```

## 📚 Populate Database

There are a total of 50 chapters pre-stored in memory_data.json, typing chapter range 1-50 will load all the pre-saved stories

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
docker-compose exec memory_system python clear_database.py

# Get comprehensive database statistics
docker-compose exec memory_system python database_stats.py

# Check basic database stats
docker-compose exec memory_system python -c "from memory_manager import MemoryManager; mm = MemoryManager(); print(mm.get_database_stats())"

# List memories for character
docker-compose exec memory_system python -c "from memory_manager import MemoryManager; mm = MemoryManager(); print(mm.list_memories_for_character('Byleth', limit=10))"
```

### Development and Debugging

```bash
# Access container shell
docker-compose exec memory_system bash

# Check file structure
docker-compose exec memory_system ls -la /app/

# Test imports
docker-compose exec memory_system python -c "from memory_manager import MemoryManager; print('✅ All imports successful')"
```

## 📈 Performance Monitoring

### Access Count Tracking

The system automatically tracks memory access:

- Increments `access_count` on each retrieval
- Influences LLM scoring algorithm
- Helps identify frequently accessed memories

### LLM Agent Scoring

Memories are scored based on:

- **Relevance** (40%): How well memory answers the query
- **Salience** (25%): Memory importance
- **Recency** (20%): Chapter number (lower = older)
- **Access Frequency** (15%): How often memory is accessed

### Performance Metrics

- **Insertion Time**: ~2-4 seconds per chapter
- **Retrieval Time**: ~1-4 seconds per query
- **Memory Count**: Varies by chapter (2-7 memories per chapter)

## 🛠️ Troubleshooting

### Common Issues

**1. Port 5432 already in use**

```bash
# Stop conflicting PostgreSQL
sudo systemctl stop postgresql
# Or
docker stop $(docker ps -q --filter ancestor=postgres)
```

**2. API Key errors**

```bash
# Check environment variables
docker-compose exec memory_system env | grep API
```

**3. Database connection issues**

```bash
# Check PostgreSQL health
docker-compose exec memory_postgres pg_isready -U memory_user -d memory_system
```

**4. Memory retrieval returns empty results**

```bash
# Check if database is populated
docker-compose exec memory_system python -c "from memory_manager import MemoryManager; print(MemoryManager().get_database_stats())"
```

### Logs and Debugging

```bash
# View application logs
docker-compose logs -f memory_system

# View database logs
docker-compose logs -f memory_postgres

# Check container status
docker-compose ps
```
