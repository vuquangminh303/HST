# Data Schema Agent API - Usage Guide

Complete REST API for data schema analysis pipeline with AI agent.

## 🚀 Quick Start

### 1. Start Server

```bash
# Install dependencies
pip install fastapi uvicorn pandas pydantic

# Start server
python server.py

# Server runs on http://localhost:8000
```

### 2. Run Example Client

```bash
# Edit file path in client_example.py
python client_example.py
```

## 📋 Complete Workflow

### Step 1: Create Session

```bash
POST /sessions
Content-Type: application/json

{
  "model": "gpt-4o-mini"
}
```

**Response:**
```json
{
  "session_id": "session_20240115_143022",
  "created_at": "2024-01-15T14:30:22",
  "sources": [],
  "current_source_id": "",
  "has_schema": false,
  "has_agent": false
}
```

### Step 2: Ingest Data

```bash
POST /sessions/{session_id}/ingest
Content-Type: multipart/form-data

file: data.csv
```

**Response:**
```json
{
  "session_id": "session_20240115_143022",
  "sources": [...],
  "current_source": {
    "source_id": "data_csv_0",
    "file_path": "/tmp/data.csv",
    "shape": [1000, 15]
  }
}
```

### Step 3: Analyze Structure

```bash
POST /sessions/{session_id}/analyze-structure
```

**Response:**
```json
{
  "session_id": "session_20240115_143022",
  "is_clean": true,
  "structure_info": {...},
  "transformations": [],
  "issues": []
}
```

### Step 4: Apply Transformations (if needed)

```bash
POST /sessions/{session_id}/apply-transformations
Content-Type: application/json

["trans_1", "trans_2"]
```

### Step 5: Clean Data Types

```bash
POST /sessions/{session_id}/clean-types
```

**Response:**
```json
{
  "session_id": "session_20240115_143022",
  "cleaning_rules": [...],
  "applied_count": 5,
  "shape": [1000, 15],
  "dtypes": {
    "price": "float64",
    "quantity": "int64",
    ...
  }
}
```

### Step 6: Set Question Context (Optional)

```bash
POST /sessions/{session_id}/question-set
Content-Type: application/json

{
  "questions": [
    "What is the total revenue by category?",
    "Show me top 10 products by price"
  ],
  "output_fields": ["total_revenue", "category_name", "product_count"],
  "additional_notes": "For monthly financial reports"
}
```

### Step 7: Generate Clarification Questions

```bash
POST /sessions/{session_id}/generate-clarifications
```

**Response:**
```json
{
  "session_id": "session_20240115_143022",
  "questions": [
    {
      "id": "q1",
      "question": "What is the currency unit for the 'price' column?",
      "suggested_answer": "USD",
      "target": "price.unit",
      "question_type": "semantic"
    },
    {
      "id": "q2",
      "question": "What is the unit of measurement for 'area'?",
      "suggested_answer": "m²",
      "target": "area.unit",
      "question_type": "semantic"
    }
  ],
  "count": 2
}
```

### Step 8: Answer Clarifications

```bash
POST /sessions/{session_id}/answer-clarifications
Content-Type: application/json

{
  "answers": {
    "q1": "USD",
    "q2": "m²"
  }
}
```

### Step 9: Generate Final Schema

```bash
POST /sessions/{session_id}/generate-schema
```

**Response:**
```json
{
  "session_id": "session_20240115_143022",
  "schema": {
    "price": {
      "name": "price",
      "semantic_type": "numeric",
      "physical_type": "float64",
      "unit": "USD",
      "description": "Product price in US Dollars",
      "is_required": true,
      "constraints": {"min": 0}
    },
    ...
  },
  "column_count": 15,
  "agent_ready": true
}
```

### Step 10: Query Agent (Main Endpoint)

#### Streaming Response (Recommended)

```bash
POST /responses
Content-Type: application/json

{
  "question": "What is the average price by category?",
  "stream": true
}
```

**Response (Server-Sent Events):**
```
data: {"chunk": "Let"}
data: {"chunk": " me"}
data: {"chunk": " query"}
data: {"chunk": " the"}
data: {"chunk": " data..."}
...
data: {"usage": {"prompt_tokens": 150, "completion_tokens": 300}}
```

#### Non-Streaming Response

```bash
POST /responses
Content-Type: application/json

{
  "question": "Show me the schema summary",
  "stream": false
}
```

**Response:**
```json
{
  "session_id": "session_20240115_143022",
  "question": "Show me the schema summary",
  "response": "The dataset has 15 columns including price (USD), quantity (integer), category (categorical)..."
}
```

## 🔍 Query Endpoints

### Get Session Info

```bash
GET /sessions/{session_id}
```

### Get Schema

```bash
GET /sessions/{session_id}/schema
```

### List All Sessions

```bash
GET /sessions
```

## 💡 Usage Examples

### Python Client

```python
import requests
import json

# Create session
response = requests.post("http://localhost:8000/sessions")
session_id = response.json()["session_id"]

# Ingest data
with open("data.csv", "rb") as f:
    files = {"file": f}
    requests.post(f"http://localhost:8000/sessions/{session_id}/ingest", files=files)

# Clean data
requests.post(f"http://localhost:8000/sessions/{session_id}/clean-types")

# Generate schema
requests.post(f"http://localhost:8000/sessions/{session_id}/generate-schema")

# Query agent (streaming)
response = requests.post(
    "http://localhost:8000/responses",
    json={"question": "What columns are in the dataset?", "stream": True},
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8')[6:])  # Remove "data: "
        if "chunk" in data:
            print(data["chunk"], end="", flush=True)
```

### cURL Examples

```bash
# Create session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini"}'

# Upload data
curl -X POST http://localhost:8000/sessions/SESSION_ID/ingest \
  -F "file=@data.csv"

# Generate schema
curl -X POST http://localhost:8000/sessions/SESSION_ID/generate-schema

# Query agent
curl -X POST http://localhost:8000/responses \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this data about?", "stream": false}'
```

### JavaScript/TypeScript

```typescript
// Create session
const sessionResponse = await fetch('http://localhost:8000/sessions', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({model: 'gpt-4o-mini'})
});
const {session_id} = await sessionResponse.json();

// Query agent (streaming)
const response = await fetch('http://localhost:8000/responses', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    question: 'What columns are available?',
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.chunk) {
        console.log(data.chunk);
      }
    }
  }
}
```

## 🎯 Integration with Other Services

### Use from FastAPI Service

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

SCHEMA_AGENT_URL = "http://localhost:8000"

@app.post("/analyze-file")
async def analyze_file(file_path: str):
    async with httpx.AsyncClient() as client:
        # Create session
        session_resp = await client.post(f"{SCHEMA_AGENT_URL}/sessions")
        session_id = session_resp.json()["session_id"]

        # Ingest and process
        with open(file_path, "rb") as f:
            files = {"file": f}
            await client.post(
                f"{SCHEMA_AGENT_URL}/sessions/{session_id}/ingest",
                files=files
            )

        await client.post(f"{SCHEMA_AGENT_URL}/sessions/{session_id}/clean-types")
        schema_resp = await client.post(
            f"{SCHEMA_AGENT_URL}/sessions/{session_id}/generate-schema"
        )

        return schema_resp.json()
```

## 📊 Response Formats

All responses follow standard JSON format:

**Success:**
```json
{
  "session_id": "...",
  "data": {...}
}
```

**Error:**
```json
{
  "detail": "Error message"
}
```

## 🔒 Configuration

Set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export DEFAULT_MODEL="gpt-4o-mini"
```

## 🛠️ Advanced Usage

### Custom System Prompt

Modify the agent initialization in `generate_schema` endpoint:

```python
agent = DataSchemaAgent(
    session=session,
    api_key=OPENAI_API_KEY,
    model=DEFAULT_MODEL,
    df_cleaned=df,
    system_prompt="Your custom prompt..."
)
```

### Conversation History

Pass conversation history to `/responses`:

```python
{
  "question": "What about prices?",
  "conversation_history": [
    {"role": "user", "content": "Show me the schema"},
    {"role": "assistant", "content": "The schema has..."}
  ]
}
```

## 📝 Notes

- Sessions are stored in memory (use Redis/DB for production)
- Files are temporarily stored during processing
- Agent uses streaming by default for better UX
- SQL queries are executed on the cleaned data automatically

## 🐛 Troubleshooting

**Connection Refused:**
- Check server is running: `ps aux | grep server.py`
- Verify port 8000 is not in use

**Schema Not Generated:**
- Ensure data is ingested first
- Check type cleaning was completed
- Review logs for errors

**Agent Not Available:**
- Generate schema first via `/generate-schema`
- Check session_id is correct
