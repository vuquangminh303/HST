"""
FastAPI Server for Data Schema Agent
Provides REST API endpoints for the complete schema pipeline
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import pandas as pd
import logging
import os
import json
from pathlib import Path
from datetime import datetime
import tempfile

from schema_pipeline import (
    DataSchemaAgent,
    DataIngestor,
    StructureAnalyzer,
    TypeInferenceEngine,
    ProfileGenerator,
    SchemaGenerator,
    SchemaValidator,
    SessionManager,
    Session,
    QuestionSet,
    UserQuestion,
    OutputField,
    Answer,
    SQLQueryTool
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Schema Agent API",
    description="Complete data schema analysis pipeline with AI agent",
    version="1.0.0"
)

# Global storage (in production, use Redis or database)
sessions = {}
agents = {}
dataframes = {}

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"
OUTPUT_DIR = "./output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ============================================================================
# Request/Response Models
# ============================================================================

class SessionCreate(BaseModel):
    """Request to create a new session"""
    model: Optional[str] = DEFAULT_MODEL


class QuestionSetRequest(BaseModel):
    """Request to set user questions and output fields"""
    questions: List[str]
    output_fields: List[str]
    additional_notes: Optional[str] = ""


class AnswerRequest(BaseModel):
    """Request to answer clarification questions"""
    answers: Dict[str, str]  # question_id -> answer


class QueryRequest(BaseModel):
    """Request to query the agent"""
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    stream: Optional[bool] = True


class SessionResponse(BaseModel):
    """Session information"""
    session_id: str
    created_at: str
    sources: List[Dict[str, Any]]
    current_source_id: str
    has_schema: bool
    has_agent: bool


# ============================================================================
# Helper Functions
# ============================================================================

def get_session(session_id: str) -> Session:
    """Get session or raise 404"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return sessions[session_id]


def get_agent(session_id: str) -> DataSchemaAgent:
    """Get agent or raise 404"""
    if session_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent for session {session_id} not found")
    return agents[session_id]


def get_dataframe(session_id: str, key: str = "cleaned") -> pd.DataFrame:
    """Get dataframe or raise 404"""
    if session_id not in dataframes:
        raise HTTPException(status_code=404, detail=f"No data for session {session_id}")

    if key not in dataframes[session_id]:
        raise HTTPException(status_code=404, detail=f"No {key} dataframe for session {session_id}")

    return dataframes[session_id][key]


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Data Schema Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /sessions": "Create new session",
            "POST /sessions/{id}/ingest": "Upload and ingest data",
            "POST /sessions/{id}/analyze-structure": "Analyze structure",
            "POST /sessions/{id}/clean-types": "Clean data types",
            "POST /sessions/{id}/question-set": "Set user questions",
            "POST /sessions/{id}/generate-clarifications": "Generate clarification questions",
            "POST /sessions/{id}/answer-clarifications": "Answer clarifications",
            "POST /sessions/{id}/generate-schema": "Generate final schema",
            "POST /responses": "Query agent (streaming)",
            "GET /sessions/{id}": "Get session info",
            "GET /sessions/{id}/schema": "Get schema",
        }
    }


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """
    Create a new session

    Returns session ID
    """
    try:
        # Create session
        session_manager = SessionManager(OUTPUT_DIR)
        session = Session(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
            sources=[],
            current_source_id="",
            raw_structure_info={},
            transformations=[],
            applied_transformations=[],
            cleaning_rules=[],
            applied_cleaning_rules=[],
            schema={},
            profiles={},
            questions=[],
            answers=[],
            agent_conversations=[],
            history=[],
            checkpoints=[]
        )

        sessions[session.session_id] = session
        dataframes[session.session_id] = {}

        logger.info(f"Created session: {session.session_id}")

        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            sources=[],
            current_source_id="",
            has_schema=False,
            has_agent=False
        )

    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/ingest")
async def ingest_data(session_id: str, file: UploadFile = File(...)):
    """
    Upload and ingest data file (CSV or Excel)

    Steps:
    1. Save uploaded file
    2. Discover data sources
    3. Load data
    4. Return source info
    """
    session = get_session(session_id)

    try:
        # Save uploaded file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Saved uploaded file to {tmp_path}")

        # Discover sources
        ingestor = DataIngestor()
        sources = ingestor.discover_sources([tmp_path])

        if not sources:
            raise HTTPException(status_code=400, detail="No valid data sources found in file")

        # Update session
        session.sources = sources
        session.current_source_id = sources[0].source_id

        # Load raw data
        raw_df = ingestor.load_raw(sources[0])
        dataframes[session_id]["raw"] = raw_df

        logger.info(f"Ingested data: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols")

        return {
            "session_id": session_id,
            "sources": [s.to_dict() for s in sources],
            "current_source": {
                "source_id": sources[0].source_id,
                "file_path": sources[0].file_path,
                "sheet_name": sources[0].sheet_name,
                "shape": raw_df.shape
            }
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/analyze-structure")
async def analyze_structure(session_id: str):
    """
    Analyze data structure and propose transformations

    Returns:
    - Structure issues detected
    - Proposed transformations
    - Whether structure is clean
    """
    session = get_session(session_id)
    raw_df = get_dataframe(session_id, "raw")

    try:
        analyzer = StructureAnalyzer(api_key=OPENAI_API_KEY, model=DEFAULT_MODEL)

        structure_info, transformations, questions, is_clean = analyzer.analyze_structure(raw_df)

        session.raw_structure_info = structure_info
        session.transformations = transformations
        session.is_clean_structure = is_clean

        # If clean, use raw as clean
        if is_clean:
            dataframes[session_id]["clean"] = raw_df.copy()

        logger.info(f"Structure analysis: {'clean' if is_clean else f'{len(transformations)} transformations needed'}")

        return {
            "session_id": session_id,
            "is_clean": is_clean,
            "structure_info": structure_info,
            "transformations": [t.to_dict() for t in transformations],
            "issues": session.structure_issues if hasattr(session, 'structure_issues') else []
        }

    except Exception as e:
        logger.error(f"Structure analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/apply-transformations")
async def apply_transformations(session_id: str, transformation_ids: List[str]):
    """
    Apply selected structure transformations

    Args:
        transformation_ids: List of transformation IDs to apply
    """
    session = get_session(session_id)
    raw_df = get_dataframe(session_id, "raw")

    try:
        analyzer = StructureAnalyzer(api_key=OPENAI_API_KEY, model=DEFAULT_MODEL)

        df = raw_df.copy()
        for trans_id in transformation_ids:
            trans = next((t for t in session.transformations if t.id == trans_id), None)
            if trans:
                df = analyzer.apply_transformation(df, trans)
                session.applied_transformations.append(trans_id)

        dataframes[session_id]["clean"] = df

        logger.info(f"Applied {len(transformation_ids)} transformations")

        return {
            "session_id": session_id,
            "applied_count": len(transformation_ids),
            "shape": df.shape
        }

    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/clean-types")
async def clean_types(session_id: str):
    """
    Analyze and clean data types

    Steps:
    1. Generate column profiles
    2. Infer correct types
    3. Generate cleaning rules
    4. Apply cleaning rules

    Returns cleaned data info
    """
    session = get_session(session_id)

    # Get dataframe (cleaned or raw)
    if "clean" in dataframes[session_id]:
        df = dataframes[session_id]["clean"]
    else:
        df = get_dataframe(session_id, "raw")

    try:
        # Generate profiles
        profiles = ProfileGenerator.generate_profiles(df)
        session.profiles.update(profiles)

        # Generate cleaning rules
        engine = TypeInferenceEngine()
        cleaning_rules = engine.generate_cleaning_rules(df, profiles)
        session.cleaning_rules = cleaning_rules

        # Apply all cleaning rules
        cleaned_df = df.copy()
        for rule in cleaning_rules:
            cleaned_df = engine.apply_cleaning_rule(cleaned_df, rule)
            session.applied_cleaning_rules.append(rule.id)

        dataframes[session_id]["cleaned"] = cleaned_df

        logger.info(f"Type cleaning: applied {len(cleaning_rules)} rules")

        return {
            "session_id": session_id,
            "cleaning_rules": [r.to_dict() for r in cleaning_rules],
            "applied_count": len(cleaning_rules),
            "shape": cleaned_df.shape,
            "dtypes": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()}
        }

    except Exception as e:
        logger.error(f"Type cleaning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/question-set")
async def set_question_set(session_id: str, request: QuestionSetRequest):
    """
    Set user questions and expected output fields

    This helps guide schema generation
    """
    session = get_session(session_id)

    try:
        # Parse questions
        user_questions = [
            UserQuestion(
                id=f"uq_{i}",
                question=q,
                description="",
                priority="medium"
            ) for i, q in enumerate(request.questions)
        ]

        # Parse output fields
        output_fields = [
            OutputField(
                field_name=f,
                description=f"Field: {f}",
                data_type="string",
                required=True
            ) for f in request.output_fields
        ]

        # Create question set
        session.question_set = QuestionSet(
            user_questions=user_questions,
            output_fields=output_fields,
            additional_notes=request.additional_notes
        )

        logger.info(f"Set question set: {len(user_questions)} questions, {len(output_fields)} fields")

        return {
            "session_id": session_id,
            "questions_count": len(user_questions),
            "fields_count": len(output_fields)
        }

    except Exception as e:
        logger.error(f"Failed to set question set: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/generate-clarifications")
async def generate_clarifications(session_id: str):
    """
    Step 1 of schema generation: Generate clarification questions

    LLM analyzes data and generates questions about:
    - Units (VND, USD, m², etc.)
    - Formats
    - Constraints
    - Business context
    """
    session = get_session(session_id)
    df = get_dataframe(session_id, "cleaned")

    try:
        # Generate profiles if not exists
        if not session.profiles:
            profiles = ProfileGenerator.generate_profiles(df)
            session.profiles.update(profiles)

        sample_rows = ProfileGenerator.get_sample_rows(df)

        # Generate clarification questions
        schema_gen = SchemaGenerator(api_key=OPENAI_API_KEY, model=DEFAULT_MODEL)
        questions = schema_gen.generate_clarification_questions(
            session.profiles,
            sample_rows,
            question_set=session.question_set
        )

        session.questions = questions

        logger.info(f"Generated {len(questions)} clarification questions")

        return {
            "session_id": session_id,
            "questions": [q.to_dict() for q in questions],
            "count": len(questions)
        }

    except Exception as e:
        logger.error(f"Clarification generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/answer-clarifications")
async def answer_clarifications(session_id: str, request: AnswerRequest):
    """
    Answer clarification questions

    Stores answers to be used in schema generation
    """
    session = get_session(session_id)

    try:
        # Convert answers to Answer objects
        answers = [
            Answer(
                question_id=q_id,
                answer=ans,
                timestamp=datetime.now().isoformat(),
                applied=True
            )
            for q_id, ans in request.answers.items()
        ]

        session.answers.extend(answers)

        logger.info(f"Received {len(answers)} answers")

        return {
            "session_id": session_id,
            "answers_count": len(answers),
            "total_answers": len(session.answers)
        }

    except Exception as e:
        logger.error(f"Failed to save answers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/generate-schema")
async def generate_schema(session_id: str):
    """
    Step 2 of schema generation: Generate final schema

    Uses:
    - Column profiles
    - User question set (if provided)
    - Clarification answers

    Returns final schema with semantic types, units, descriptions
    """
    session = get_session(session_id)
    df = get_dataframe(session_id, "cleaned")

    try:
        # Generate profiles if not exists
        if not session.profiles:
            profiles = ProfileGenerator.generate_profiles(df)
            session.profiles.update(profiles)

        sample_rows = ProfileGenerator.get_sample_rows(df)

        # Generate schema
        schema_gen = SchemaGenerator(api_key=OPENAI_API_KEY, model=DEFAULT_MODEL)
        schema, additional_questions = schema_gen.generate_schema(
            session.profiles,
            sample_rows,
            question_set=session.question_set,
            clarification_answers=session.answers if session.answers else None
        )

        session.schema.update(schema)

        # Create agent with schema
        agent = DataSchemaAgent(
            session=session,
            api_key=OPENAI_API_KEY,
            model=DEFAULT_MODEL,
            df_cleaned=df
        )

        agents[session_id] = agent

        logger.info(f"Generated schema with {len(schema)} columns")

        return {
            "session_id": session_id,
            "schema": {k: v.to_dict() for k, v in schema.items()},
            "column_count": len(schema),
            "agent_ready": True,
            "additional_questions": [q.to_dict() for q in additional_questions] if additional_questions else []
        }

    except Exception as e:
        logger.error(f"Schema generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/responses")
async def query_agent(request: QueryRequest):
    """
    Query the agent - main chat endpoint

    Supports:
    - Streaming or non-streaming
    - Conversation history
    - SQL query execution

    Returns agent response
    """
    # Extract session_id from conversation history or use latest session
    if request.conversation_history and len(request.conversation_history) > 0:
        # Try to extract from history metadata
        session_id = request.conversation_history[0].get("session_id")
    else:
        # Use most recent session
        if not sessions:
            raise HTTPException(status_code=400, detail="No active session. Create session first.")
        session_id = list(sessions.keys())[-1]

    agent = get_agent(session_id)

    try:
        if request.stream:
            # Streaming response
            async def generate():
                for chunk in agent.query(
                    question=request.question,
                    conversation_history=request.conversation_history
                ):
                    if isinstance(chunk, dict):
                        # Final metadata
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # Text chunk
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming
            response = agent.chat(request.question)

            return {
                "session_id": session_id,
                "question": request.question,
                "response": response
            }

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session information"""
    session = get_session(session_id)

    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        sources=[s.to_dict() for s in session.sources],
        current_source_id=session.current_source_id,
        has_schema=bool(session.schema),
        has_agent=session_id in agents
    )


@app.get("/sessions/{session_id}/schema")
async def get_schema(session_id: str):
    """Get generated schema"""
    session = get_session(session_id)

    if not session.schema:
        raise HTTPException(status_code=404, detail="Schema not generated yet")

    return {
        "session_id": session_id,
        "schema": {k: v.to_dict() for k, v in session.schema.items()},
        "column_count": len(session.schema)
    }


@app.get("/sessions")
async def list_sessions():
    """List all sessions"""
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "has_schema": bool(s.schema),
                "has_agent": sid in agents
            }
            for sid, s in sessions.items()
        ],
        "count": len(sessions)
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
