-- Setup script for pgvector and Sekai Memory System
-- Run these commands in your PostgreSQL instance

-- Enable the pgvector extension
-- Note: This requires pgvector to be installed in PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;

-- 5. Create the memories table
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chapter INT NOT NULL,
    source_char VARCHAR(100),
    target_char VARCHAR(100),
    memory_type VARCHAR(50),
    salience FLOAT DEFAULT 0.5,
    memory_text TEXT,
    embedding VECTOR(1536), -- OpenAI text-embedding-ada-002 dimensions
    access_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Create indexes for better performance
CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON memories (source_char, target_char);
CREATE INDEX ON memories (chapter);
CREATE INDEX ON memories (memory_type);
CREATE INDEX ON memories (access_count);
CREATE INDEX ON memories (salience);

-- 7. Verify the setup
SELECT 'pgvector extension status:' as info;
SELECT * FROM pg_extension WHERE extname = 'vector';

SELECT 'memories table created:' as info;
SELECT COUNT(*) FROM memories;
