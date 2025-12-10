"""
Example client to use Data Schema Agent API
Demonstrates complete workflow from data ingestion to querying
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"


def print_step(step: str, emoji: str = "📌"):
    """Print step header"""
    print(f"\n{emoji} {step}")
    print("=" * 60)


def create_session():
    """Step 1: Create a new session"""
    print_step("Step 1: Create Session", "🎯")

    response = requests.post(f"{API_BASE}/sessions", json={
        "model": "gpt-4o-mini"
    })

    if response.status_code == 200:
        data = response.json()
        session_id = data["session_id"]
        print(f"✅ Session created: {session_id}")
        return session_id
    else:
        print(f"❌ Failed: {response.text}")
        return None


def ingest_data(session_id: str, file_path: str):
    """Step 2: Upload and ingest data"""
    print_step("Step 2: Ingest Data", "📤")

    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(
            f"{API_BASE}/sessions/{session_id}/ingest",
            files=files
        )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Data ingested:")
        print(f"   Shape: {data['current_source']['shape']}")
        print(f"   File: {data['current_source']['file_path']}")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def analyze_structure(session_id: str):
    """Step 3: Analyze data structure"""
    print_step("Step 3: Analyze Structure", "🔍")

    response = requests.post(f"{API_BASE}/sessions/{session_id}/analyze-structure")

    if response.status_code == 200:
        data = response.json()
        is_clean = data["is_clean"]

        if is_clean:
            print("✅ Structure is clean!")
        else:
            print(f"⚠️  Found {len(data['transformations'])} transformations needed")
            for trans in data["transformations"]:
                print(f"   - {trans['transformation_type']}: {trans['description']}")

        return data
    else:
        print(f"❌ Failed: {response.text}")
        return None


def clean_types(session_id: str):
    """Step 4: Clean data types"""
    print_step("Step 4: Clean Data Types", "🧹")

    response = requests.post(f"{API_BASE}/sessions/{session_id}/clean-types")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Applied {data['applied_count']} cleaning rules")
        print(f"   Final shape: {data['shape']}")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def set_question_set(session_id: str):
    """Step 5: Define expected questions and output fields"""
    print_step("Step 5: Set Question Context", "❓")

    question_set = {
        "questions": [
            "What is the total revenue by category?",
            "Show me top 10 products by price",
            "How many items per region?"
        ],
        "output_fields": [
            "total_revenue",
            "category_name",
            "product_count",
            "average_price"
        ],
        "additional_notes": "This data will be used for monthly financial reports"
    }

    response = requests.post(
        f"{API_BASE}/sessions/{session_id}/question-set",
        json=question_set
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Set {data['questions_count']} questions and {data['fields_count']} output fields")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def generate_clarifications(session_id: str):
    """Step 6: Generate clarification questions"""
    print_step("Step 6: Generate Clarification Questions", "💡")

    response = requests.post(f"{API_BASE}/sessions/{session_id}/generate-clarifications")

    if response.status_code == 200:
        data = response.json()
        questions = data["questions"]

        print(f"✅ Generated {len(questions)} clarification questions:")
        for i, q in enumerate(questions, 1):
            print(f"\n   Q{i}: {q['question']}")
            print(f"   Target: {q['target']}")
            print(f"   Suggested: {q['suggested_answer']}")

        return questions
    else:
        print(f"❌ Failed: {response.text}")
        return []


def answer_clarifications(session_id: str, questions: list):
    """Step 7: Answer clarification questions"""
    print_step("Step 7: Answer Clarification Questions", "✍️")

    # Use suggested answers (in real app, user would provide these)
    answers = {
        q["id"]: q["suggested_answer"] or "Unknown"
        for q in questions
    }

    response = requests.post(
        f"{API_BASE}/sessions/{session_id}/answer-clarifications",
        json={"answers": answers}
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Answered {data['answers_count']} questions")
        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def generate_schema(session_id: str):
    """Step 8: Generate final schema"""
    print_step("Step 8: Generate Final Schema", "📋")

    response = requests.post(f"{API_BASE}/sessions/{session_id}/generate-schema")

    if response.status_code == 200:
        data = response.json()
        schema = data["schema"]

        print(f"✅ Generated schema with {data['column_count']} columns")
        print(f"   Agent ready: {data['agent_ready']}")

        # Show first 3 columns
        print("\n   Sample columns:")
        for i, (col_name, col_schema) in enumerate(list(schema.items())[:3]):
            print(f"   - {col_name}:")
            print(f"     Type: {col_schema['semantic_type']} ({col_schema['physical_type']})")
            if col_schema.get('unit'):
                print(f"     Unit: {col_schema['unit']}")
            print(f"     Description: {col_schema['description']}")

        return True
    else:
        print(f"❌ Failed: {response.text}")
        return False


def query_agent(session_id: str, question: str, stream: bool = True):
    """Step 9: Query the agent via /responses endpoint"""
    print_step(f"Query Agent: {question}", "🤖")

    payload = {
        "question": question,
        "stream": stream
    }

    if stream:
        # Streaming response
        response = requests.post(
            f"{API_BASE}/responses",
            json=payload,
            stream=True
        )

        if response.status_code == 200:
            print("Response: ", end="", flush=True)
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data = json.loads(line_str[6:])
                        if "chunk" in data:
                            print(data["chunk"], end="", flush=True)
                        elif "usage" in data:
                            # Final metadata
                            print(f"\n\n📊 Usage: {data['usage']}")

            print()  # New line
        else:
            print(f"❌ Failed: {response.text}")
    else:
        # Non-streaming
        response = requests.post(f"{API_BASE}/responses", json=payload)

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data['response']}")
        else:
            print(f"❌ Failed: {response.text}")


def get_schema(session_id: str):
    """Get generated schema"""
    print_step("Get Schema", "📄")

    response = requests.get(f"{API_BASE}/sessions/{session_id}/schema")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Schema has {data['column_count']} columns")
        return data["schema"]
    else:
        print(f"❌ Failed: {response.text}")
        return None


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """Run complete workflow"""
    print("\n" + "="*60)
    print("🚀 Data Schema Agent API - Complete Workflow")
    print("="*60)

    # Step 1: Create session
    session_id = create_session()
    if not session_id:
        return

    time.sleep(0.5)

    # Step 2: Ingest data (replace with your file path)
    file_path = "data.csv"  # Change this to your file

    if not ingest_data(session_id, file_path):
        print("\n⚠️  Note: Using example file path 'data.csv'")
        print("   Please replace with your actual file path")
        return

    time.sleep(0.5)

    # Step 3: Analyze structure
    structure_result = analyze_structure(session_id)
    if not structure_result:
        return

    time.sleep(0.5)

    # Step 4: Clean types
    if not clean_types(session_id):
        return

    time.sleep(0.5)

    # Step 5: Set question context (optional)
    set_question_set(session_id)
    time.sleep(0.5)

    # Step 6: Generate clarification questions
    questions = generate_clarifications(session_id)
    if not questions:
        return

    time.sleep(1)

    # Step 7: Answer clarifications
    if not answer_clarifications(session_id, questions):
        return

    time.sleep(0.5)

    # Step 8: Generate final schema
    if not generate_schema(session_id):
        return

    time.sleep(1)

    # Step 9: Query the agent
    print_step("Agent Q&A Session", "💬")

    # Example queries
    queries = [
        "What columns are in this dataset?",
        "What is the data about?",
        "Show me a sample of the data"
    ]

    for query in queries:
        query_agent(session_id, query, stream=True)
        time.sleep(1)

    # Get final schema
    schema = get_schema(session_id)

    print("\n" + "="*60)
    print("✅ Workflow completed successfully!")
    print("="*60)
    print(f"\nSession ID: {session_id}")
    print(f"You can now query the agent at: POST {API_BASE}/responses")


if __name__ == "__main__":
    main()
