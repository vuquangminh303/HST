"""
Streamlit UI V5 - Question-Driven Schema Generation
Full workflow from V4 + User Question Collection + Schema Validation
Allows users to define expected questions and output format before schema generation
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import sqlite3

# Import from pipeline
from schema_pipeline import (
    DataIngestor, StructureAnalyzer, ProfileGenerator, SchemaGenerator,
    SessionManager, RefinementEngine, DataSource, Session, Transformation,
    Question, Answer, ColumnProfile, ColumnSchema, DataFrameCheckpoint,
    TypeInferenceEngine, CleaningRule, DataSchemaAgent, AgentMessage,
    UserQuestion, OutputField, QuestionSet, SchemaValidator
)

# Page config
st.set_page_config(
    page_title="Data Schema Pipeline V4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .cleaning-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #4CAF50;
        background-color: #f1f8e9;
        margin: 0.5rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'session' not in st.session_state:
        st.session_state.session = None
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = None
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'current_source' not in st.session_state:
        st.session_state.current_source = None
    if 'sources' not in st.session_state:
        st.session_state.sources = []
    if 'raw_dfs' not in st.session_state:
        st.session_state.raw_dfs = {}
    if 'clean_dfs' not in st.session_state:
        st.session_state.clean_dfs = {}
    if 'cleaned_dfs' not in st.session_state:
        st.session_state.cleaned_dfs = {}
    if 'agent' not in st.session_state:
        st.session_state.agent = None


# ============================================================================
# Tab 1: Data Ingestion
# ============================================================================

def tab_ingestion():
    """Data ingestion tab"""
    st.header("📁 Data Ingestion")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Data Files")
        uploaded_files = st.file_uploader(
            "Upload CSV or Excel files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="You can upload multiple CSV files or Excel files with multiple sheets"
        )
    
    with col2:
        st.subheader("Settings")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get('api_key', ''),
            help="Required for LLM analysis"
        )
        
        model = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=0
        )
        
        output_dir = st.text_input(
            "Output Directory",
            value="./output_streamlit_v4"
        )
    
    if st.button("🚀 Start Pipeline", type="primary", disabled=not uploaded_files or not api_key):
        with st.spinner("Discovering data sources..."):
            # Save uploaded files temporarily
            temp_dir = Path("./temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            file_paths = []
            for uploaded_file in uploaded_files:
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(str(temp_path))
            
            # Discover sources
            sources = DataIngestor.discover_sources(file_paths)
            
            if not sources:
                st.error("No valid data sources found")
                return
            
            # Initialize session
            session_manager = SessionManager(output_dir)
            session = session_manager.create_session(
                sources=sources,
                current_source_id=sources[0].source_id
            )
            
            # Store in session state
            st.session_state.session = session
            st.session_state.session_manager = session_manager
            st.session_state.sources = sources
            st.session_state.api_key = api_key
            st.session_state.model = model
            
            # Load all raw dataframes
            for source in sources:
                df_raw = DataIngestor.load_raw(source)
                st.session_state.raw_dfs[source.source_id] = df_raw
                
                # Save checkpoint
                session_manager.save_checkpoint(
                    session, df_raw, f"raw_{source.source_id}", 
                    f"Raw data from {source.source_id}"
                )
            
            # Initialize agent
            st.session_state.agent = DataSchemaAgent(
                session, api_key, model
            )
            
            st.success(f"✅ Discovered {len(sources)} data source(s)!")
            st.rerun()
    
    # Display discovered sources
    if st.session_state.sources:
        st.divider()
        st.subheader("📋 Discovered Data Sources")
        
        source_data = []
        for source in st.session_state.sources:
            df = st.session_state.raw_dfs.get(source.source_id)
            source_data.append({
                "Source ID": source.source_id,
                "File": Path(source.file_path).name,
                "Sheet": source.sheet_name or "N/A",
                "Type": source.source_type,
                "Rows": len(df) if df is not None else 0,
                "Columns": len(df.columns) if df is not None else 0
            })
        
        st.dataframe(pd.DataFrame(source_data), use_container_width=True)
        
        # Preview
        st.subheader("👁️ Data Preview")
        selected_source = st.selectbox(
            "Select source to preview",
            options=[s.source_id for s in st.session_state.sources]
        )
        
        if selected_source and selected_source in st.session_state.raw_dfs:
            df = st.session_state.raw_dfs[selected_source]
            st.dataframe(df.head(20), use_container_width=True)


# ============================================================================
# Tab 2: Structure Analysis
# ============================================================================

def tab_structure_analysis():
    """Structure analysis and transformation tab"""
    st.header("🔍 Structure Analysis & Transformation")
    
    if not st.session_state.session:
        st.warning("⚠️ Please ingest data first (go to 'Ingestion' tab)")
        return
    
    session = st.session_state.session
    
    # Select source
    st.subheader("Select Data Source")
    selected_source_id = st.selectbox(
        "Source",
        options=[s.source_id for s in session.sources],
        key="structure_source_select"
    )
    
    source = next(s for s in session.sources if s.source_id == selected_source_id)
    
    if st.button("🔍 Analyze Structure", type="primary"):
        with st.spinner("Analyzing structure..."):
            analyzer = StructureAnalyzer(
                api_key=st.session_state.api_key,
                model=st.session_state.model
            )
            
            df_raw = st.session_state.raw_dfs[selected_source_id]
            
            structure_info, transformations, questions, is_clean = analyzer.analyze_structure(df_raw)
            
            # Store results
            if 'structure_results' not in st.session_state:
                st.session_state.structure_results = {}
            
            st.session_state.structure_results[selected_source_id] = {
                'structure_info': structure_info,
                'transformations': transformations,
                'questions': questions,
                'is_clean': is_clean
            }
            
            session.raw_structure_info[selected_source_id] = structure_info
            session.is_clean_structure = is_clean
            
            if not is_clean:
                _, issues = analyzer._heuristic_structure_check(df_raw)
                session.structure_issues = issues
            
            st.rerun()
    
    # Display results
    if hasattr(st.session_state, 'structure_results') and selected_source_id in st.session_state.structure_results:
        results = st.session_state.structure_results[selected_source_id]
        
        # Status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅ Clean" if results['is_clean'] else "⚠️ Issues Detected"
            st.metric("Structure Status", status)
        
        with col2:
            st.metric("Transformations Proposed", len(results['transformations']))
        
        with col3:
            st.metric("Questions", len(results['questions']))
        
        st.divider()
        
        # Issues
        if not results['is_clean'] and session.structure_issues:
            st.subheader("⚠️ Detected Issues")
            for issue in session.structure_issues:
                st.warning(f"• {issue}")
            st.divider()
        
        # Transformations
        if results['transformations']:
            st.subheader("🔧 Proposed Transformations")
            
            for i, trans in enumerate(results['transformations']):
                with st.expander(f"**{trans.description}**", expanded=True):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Type:** `{trans.type}`")
                        st.write(f"**Parameters:** `{json.dumps(trans.params, indent=2)}`")
                    
                    with col2:
                        st.metric("Confidence", f"{trans.confidence:.0%}")
                    
                    with col3:
                        if st.button(f"✅ Apply", key=f"apply_{trans.id}"):
                            df_raw = st.session_state.raw_dfs[selected_source_id]
                            df_transformed = StructureAnalyzer.apply_transformation(df_raw, trans)
                            
                            # Save checkpoint
                            st.session_state.session_manager.save_checkpoint(
                                session, df_transformed, f"transform_{trans.id}",
                                f"After: {trans.description}"
                            )
                            
                            # Update current df
                            st.session_state.clean_dfs[selected_source_id] = df_transformed
                            trans.applied = True
                            session.applied_transformations.append(trans.id)
                            
                            st.success(f"✅ Applied: {trans.description}")
                            st.rerun()
        
        # Show before/after comparison
        if selected_source_id in st.session_state.clean_dfs:
            st.divider()
            st.subheader("📊 Before/After Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before (Raw)**")
                df_raw = st.session_state.raw_dfs[selected_source_id]
                st.dataframe(df_raw.head(10), use_container_width=True)
                st.caption(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
            
            with col2:
                st.write("**After (Transformed)**")
                df_clean = st.session_state.clean_dfs[selected_source_id]
                st.dataframe(df_clean.head(10), use_container_width=True)
                st.caption(f"Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")


# ============================================================================
# Tab 3: Type Inference & Data Cleaning
# ============================================================================

def tab_type_cleaning():
    """Type inference and data cleaning tab"""
    st.header("🧹 Type Inference & Data Cleaning")
    
    if not st.session_state.session:
        st.warning("⚠️ Please ingest data first")
        return
    
    session = st.session_state.session
    
    # Select source
    selected_source_id = st.selectbox(
        "Select Source",
        options=[s.source_id for s in session.sources],
        key="clean_source_select"
    )
    
    # Get current df (after structure transforms if any)
    df = st.session_state.clean_dfs.get(
        selected_source_id,
        st.session_state.raw_dfs.get(selected_source_id)
    )
    
    if df is None:
        st.error("No data available for this source")
        return
    
    if st.button("🔍 Analyze Types", type="primary"):
        with st.spinner("Analyzing types and generating cleaning rules..."):
            # Generate profiles with type inference
            profiles = ProfileGenerator.generate_profiles(df)
            
            # Generate cleaning rules
            cleaning_rules = TypeInferenceEngine.generate_cleaning_rules(df, profiles)
            
            # Store results
            if 'cleaning_results' not in st.session_state:
                st.session_state.cleaning_results = {}
            
            st.session_state.cleaning_results[selected_source_id] = {
                'profiles': profiles,
                'cleaning_rules': cleaning_rules
            }
            
            session.cleaning_rules.extend(cleaning_rules)
            
            st.success(f"✅ Found {len(cleaning_rules)} cleaning rule(s)")
            st.rerun()
    
    # Display results
    if hasattr(st.session_state, 'cleaning_results') and selected_source_id in st.session_state.cleaning_results:
        results = st.session_state.cleaning_results[selected_source_id]
        
        st.divider()
        st.subheader("📊 Type Inference Results")
        
        # Show inferred types table
        type_data = []
        for col_name, profile in results['profiles'].items():
            type_data.append({
                "Column": col_name,
                "Current Type": profile.pandas_dtype,
                "Inferred Type": profile.inferred_type,
                "Has Separator": "✅" if profile.has_thousand_separator else "❌",
                "Example Value": profile.sample_raw_values[0] if profile.sample_raw_values else ""
            })
        
        st.dataframe(pd.DataFrame(type_data), use_container_width=True)
        
        # Cleaning rules
        if results['cleaning_rules']:
            st.divider()
            st.subheader("🧹 Data Cleaning Rules")
            
            st.info(f"💡 Found {len(results['cleaning_rules'])} columns that need type conversion")
            
            for rule in results['cleaning_rules']:
                with st.expander(f"**{rule.description}**", expanded=True):
                    st.markdown(f"""
                    <div class="cleaning-card">
                        <strong>Column:</strong> {rule.column}<br>
                        <strong>Action:</strong> {rule.action}<br>
                        <strong>Target Type:</strong> {rule.column} will be converted to proper numeric type
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show before/after preview
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Before Cleaning (sample values):**")
                        before_sample = df[rule.column].dropna().head(5).tolist()
                        st.code("\n".join(str(v) for v in before_sample))
                    
                    with col2:
                        if st.button(f"✅ Apply Cleaning", key=f"apply_clean_{rule.id}"):
                            df_current = st.session_state.clean_dfs.get(
                                selected_source_id, df
                            )
                            
                            df_cleaned = TypeInferenceEngine.apply_cleaning_rule(df_current, rule)
                            
                            st.session_state.session_manager.save_checkpoint(
                                session, df_cleaned, f"clean_{rule.id}", rule.description
                            )
                            
                            st.session_state.cleaned_dfs[selected_source_id] = df_cleaned
                            session.applied_cleaning_rules.append(rule.id)
                            rule.applied = True
                            
                            st.success("✅ Cleaning applied!")
                            st.rerun()
        else:
            st.info("✅ No cleaning needed - all types are already correct!")
        
        # Show final cleaned data preview
        if selected_source_id in st.session_state.cleaned_dfs:
            st.divider()
            st.subheader("✨ Cleaned Data Preview")
            
            df_cleaned = st.session_state.cleaned_dfs[selected_source_id]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", df_cleaned.shape[0])
            with col2:
                st.metric("Columns", df_cleaned.shape[1])
            
            st.dataframe(df_cleaned.head(20), use_container_width=True)
            
            # Show dtype changes
            st.write("**Data Types After Cleaning:**")
            dtype_df = pd.DataFrame({
                "Column": df_cleaned.columns,
                "Type": [str(dtype) for dtype in df_cleaned.dtypes]
            })
            st.dataframe(dtype_df, use_container_width=True)


# ============================================================================
# Tab 4: Schema Generation
# ============================================================================

def tab_schema_generation():
    """Schema generation tab"""
    st.header("📋 Semantic Schema Generation")
    
    if not st.session_state.session:
        st.warning("⚠️ No active session")
        return
    
    session = st.session_state.session
    
    selected_source_id = st.selectbox(
        "Select Source",
        options=[s.source_id for s in session.sources],
        key="schema_source_select"
    )
    
    # Use cleaned df if available, otherwise clean, otherwise raw
    df = st.session_state.cleaned_dfs.get(
        selected_source_id,
        st.session_state.clean_dfs.get(
            selected_source_id,
            st.session_state.raw_dfs.get(selected_source_id)
        )
    )
    
    if df is None:
        st.error("No data available")
        return

    # ============================================================================
    # Question & Output Format Definition (Optional)
    # ============================================================================
    with st.expander("❓ Define Expected Questions & Output Format (Optional)", expanded=False):
        st.markdown("""
        **Help the system understand how you'll use this data:**

        Define questions you'll frequently ask and the output format you need.
        This helps generate a more relevant schema.
        """)

        # Initialize question_set if not exists
        if not session.question_set:
            session.question_set = QuestionSet()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Sample Questions")
            questions_text = st.text_area(
                "Enter questions (one per line)",
                value="\n".join([q.question for q in session.question_set.user_questions]) if session.question_set.user_questions else "",
                placeholder="What is the total revenue by category?\nShow top 10 products by price\nHow many items per region?",
                height=150,
                key="questions_input"
            )

        with col2:
            st.subheader("📊 Expected Output Fields")
            fields_text = st.text_area(
                "Enter field names (one per line or comma-separated)",
                value=", ".join([f.field_name for f in session.question_set.output_fields]) if session.question_set.output_fields else "",
                placeholder="total_revenue, category_name, product_count, average_price",
                height=150,
                key="fields_input"
            )

        additional_notes = st.text_area(
            "Additional context (optional)",
            value=session.question_set.additional_notes if session.question_set else "",
            placeholder="E.g., This data will be used for monthly financial reports...",
            height=80,
            key="notes_input"
        )

        if st.button("💾 Save Question Context"):
            # Parse questions
            questions_list = [q.strip() for q in questions_text.split('\n') if q.strip()]
            session.question_set.user_questions = [
                UserQuestion(
                    id=f"uq_{i}_{datetime.now().timestamp()}",
                    question=q,
                    description="",
                    priority="medium"
                ) for i, q in enumerate(questions_list)
            ]

            # Parse fields
            if ',' in fields_text:
                fields_list = [f.strip() for f in fields_text.split(',') if f.strip()]
            else:
                fields_list = [f.strip() for f in fields_text.split('\n') if f.strip()]

            session.question_set.output_fields = [
                OutputField(
                    field_name=f,
                    description=f"Field: {f}",
                    data_type="string",
                    required=True
                ) for f in fields_list
            ]

            session.question_set.additional_notes = additional_notes

            st.success(f"✓ Saved {len(session.question_set.user_questions)} questions and {len(session.question_set.output_fields)} fields")
            st.rerun()

        # Show summary
        if session.question_set and (session.question_set.user_questions or session.question_set.output_fields):
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Questions", len(session.question_set.user_questions))
            with col2:
                st.metric("Output Fields", len(session.question_set.output_fields))

    st.divider()

    if st.button("🧠 Generate Schema", type="primary"):
        with st.spinner("Generating semantic schema..."):
            # Generate profiles
            profiles = ProfileGenerator.generate_profiles(df)
            sample_rows = ProfileGenerator.get_sample_rows(df)

            # Generate schema (with question_set if available)
            schema_gen = SchemaGenerator(
                api_key=st.session_state.api_key,
                model=st.session_state.model
            )
            schema, questions = schema_gen.generate_schema(
                profiles,
                sample_rows,
                question_set=session.question_set
            )

            # Validate schema against user questions (if provided)
            validation_report = None
            is_sufficient = True
            if session.question_set and (session.question_set.user_questions or session.question_set.output_fields):
                with st.spinner("Validating schema against your questions..."):
                    validator = SchemaValidator(
                        api_key=st.session_state.api_key,
                        model=st.session_state.model
                    )
                    is_sufficient, additional_questions, validation_report = validator.validate_schema_for_questions(
                        schema, profiles, session.question_set, sample_rows
                    )

                    # Add additional questions to refinement questions
                    if not is_sufficient and additional_questions:
                        questions.extend(additional_questions)

            # Store results
            if 'schema_results' not in st.session_state:
                st.session_state.schema_results = {}

            st.session_state.schema_results[selected_source_id] = {
                'profiles': profiles,
                'schema': schema,
                'questions': questions,
                'validation_report': validation_report,
                'is_sufficient': is_sufficient
            }

            # Update session
            session.profiles.update(profiles)
            session.schema.update(schema)
            session.questions.extend(questions)

            st.success(f"✅ Generated schema for {len(schema)} columns")
            if not is_sufficient:
                st.warning("⚠️ Schema may not be sufficient to answer all your questions. Please review validation report and answer refinement questions below.")
            st.rerun()
    
    # Display schema
    if hasattr(st.session_state, 'schema_results') and selected_source_id in st.session_state.schema_results:
        results = st.session_state.schema_results[selected_source_id]
        
        st.divider()
        st.subheader("📊 Column Profiles & Schema")
        
        # Create tabs for each column
        column_names = list(results['schema'].keys())
        
        if column_names:
            tabs = st.tabs(column_names[:10])  # Limit to first 10 for UI
            
            for tab, col_name in zip(tabs, column_names[:10]):
                with tab:
                    profile = results['profiles'][col_name]
                    schema_col = results['schema'][col_name]
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Non-Null Count", profile.non_null_count)
                    with col2:
                        st.metric("Null Ratio", f"{profile.null_ratio:.1%}")
                    with col3:
                        st.metric("Unique Values", profile.n_unique)
                    with col4:
                        st.metric("Required", "✅" if schema_col.is_required else "❌")
                    
                    # Schema details
                    st.subheader("Schema Details")
                    
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.write(f"**Description:** {schema_col.description}")
                        st.write(f"**Semantic Type:** `{schema_col.semantic_type}`")
                        st.write(f"**Physical Type:** `{schema_col.physical_type}`")
                    
                    with col_right:
                        st.write(f"**Original Type:** `{schema_col.original_type}`")
                        st.write(f"**Unit:** {schema_col.unit or 'N/A'}")
                        if schema_col.constraints:
                            st.write(f"**Constraints:** `{json.dumps(schema_col.constraints)}`")
                    
                    # Sample values
                    st.subheader("Sample Values")
                    st.write(profile.sample_values[:10])

        # Validation Report
        if results.get('validation_report'):
            st.divider()
            st.subheader("🔍 Schema Validation Report")

            if results.get('is_sufficient'):
                st.success("✅ Schema is sufficient to answer your questions!")
            else:
                st.warning("⚠️ Schema needs refinement")

            with st.expander("📄 View Detailed Report", expanded=not results.get('is_sufficient')):
                st.markdown(results['validation_report'])

        # Questions
        if results['questions']:
            st.divider()
            st.subheader("❓ Clarification Questions")
            
            for q in results['questions']:
                with st.expander(f"**{q.question}**"):
                    st.write(f"**Target:** `{q.target}`")
                    st.write(f"**Suggested Answer:** {q.suggested_answer or 'N/A'}")
                    
                    # Answer input
                    answer_key = f"answer_{q.id}"
                    user_answer = st.text_input(
                        "Your answer",
                        value=q.suggested_answer or "",
                        key=answer_key
                    )
                    
                    if st.button("✅ Submit Answer", key=f"submit_{q.id}"):
                        answer = Answer(
                            question_id=q.id,
                            answer=user_answer,
                            timestamp=datetime.now().isoformat()
                        )
                        session.answers.append(answer)
                        
                        # Apply answer
                        engine = RefinementEngine(session)
                        if engine.apply_answer(answer):
                            st.success(f"✅ Applied: {user_answer}")
                            st.rerun()


# ============================================================================
# Tab 5: Agent Q&A
# ============================================================================
class SQLQueryTool:
    """Tool to execute SQL queries - thread-safe for Streamlit"""

    def __init__(self, db_path: str, df: pd.DataFrame, table_name: str = "data"):
        self.db_path = db_path
        self.table_name = table_name
        self.df_hash = hash(str(df.values.tobytes()))  # Track df changes
        self._create_database(df)

    def _create_database(self, df: pd.DataFrame):
        """Create/recreate database with data"""
        try:
            # Close any existing connection first
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()

            # Create new connection with check_same_thread=False for Streamlit
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            df.to_sql(self.table_name, self.conn, if_exists='replace', index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to create database: {str(e)}")

    def _ensure_connection(self):
        """Ensure connection is valid, recreate if needed"""
        try:
            if not self.conn:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # Test connection
            self.conn.execute("SELECT 1")
        except:
            # Recreate connection if failed
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def execute_query(self, query: str):
        try:
            if not query.upper().strip().startswith('SELECT'):
                return pd.DataFrame(), "Only SELECT allowed"

            self._ensure_connection()
            result_df = pd.read_sql_query(query, self.conn)
            return result_df, None
        except Exception as e:
            return pd.DataFrame(), f"Error: {str(e)}"

    def get_schema_info(self):
        try:
            self._ensure_connection()
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns = cursor.fetchall()
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            row_count = cursor.fetchone()[0]
            return {
                "table_name": self.table_name,
                "columns": [{"name": col[1], "type": col[2]} for col in columns],
                "row_count": row_count
            }
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
def tab_agent_qa():
    """Enhanced Agent Q&A with SQL capability"""
    st.header("🤖 Agent Q&A - Chat & Query Your Data")
    
    if not st.session_state.session:
        st.warning("⚠️ No active session")
        return
    
    if not st.session_state.session.schema:
        st.warning("⚠️ Please generate schema first (Tab 4)")
        return
    
    session = st.session_state.session
    
    # Get cleaned DataFrame
    selected_source_id = session.current_source_id
    df_cleaned = st.session_state.cleaned_dfs.get(
        selected_source_id,
        st.session_state.clean_dfs.get(
            selected_source_id,
            st.session_state.raw_dfs.get(selected_source_id)
        )
    )

    if df_cleaned is None:
        st.warning("⚠️ No data available. Please complete data ingestion and cleaning first.")
        return

    # Calculate current df hash to detect changes
    current_df_hash = hash(str(df_cleaned.values.tobytes()))

    # Initialize or recreate SQL capability if df changed
    need_recreate = False
    if 'sql_tool' not in st.session_state or st.session_state.sql_tool is None:
        need_recreate = True
    elif 'sql_tool_df_hash' not in st.session_state:
        need_recreate = True
    elif st.session_state.sql_tool_df_hash != current_df_hash:
        need_recreate = True
        st.info("🔄 Data has changed, recreating SQL database...")

    if need_recreate:
        with st.spinner("Setting up SQL database..."):
            db_dir = Path("./agent_databases")
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / f"data_{session.session_id}.db"

            # Close old connection if exists
            if 'sql_tool' in st.session_state and st.session_state.sql_tool:
                try:
                    st.session_state.sql_tool.close()
                except:
                    pass

            sql_tool = SQLQueryTool(str(db_path), df_cleaned)
            st.session_state.sql_tool = sql_tool
            st.session_state.sql_tool_df_hash = current_df_hash

            st.success("✅ SQL database created! Agent can now query your data.")

    sql_tool = st.session_state.get('sql_tool')
    
    # Info banner
    if sql_tool:
        schema_info = sql_tool.get_schema_info()
        if schema_info and 'table_name' in schema_info and 'error' not in schema_info:
            st.info(f"""
            💡 **SQL Capability Enabled!**
            - Table: `{schema_info['table_name']}`
            - Rows: {schema_info['row_count']}
            - Columns: {len(schema_info['columns'])}

            You can ask questions like:
            - "What is the average price?"
            - "Show me the top 5 most expensive properties"
            - "How many properties per district?"
            """)
        elif 'error' in schema_info:
            st.error(f"⚠️ SQL database error: {schema_info['error']}")
        else:
            st.warning("⚠️ SQL tool initialized but schema info not available")
    else:
        st.info("💬 Ask questions about your schema (SQL queries not yet available)")
    
    # Display conversation
    st.divider()
    st.subheader("💬 Conversation")
    
    if not session.agent_conversations:
        st.caption("👋 Start by asking a question below!")
    else:
        for msg in session.agent_conversations:
            if msg.role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>{msg.content}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check if SQL was used
                used_sql = msg.context.get('used_sql', False) if msg.context else False
                sql_badge = " 🔍" if used_sql else ""
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Agent{sql_badge}:</strong><br>{msg.content}
                </div>
                """, unsafe_allow_html=True)
    
    # Input form
    st.divider()
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question:",
            placeholder="e.g., What is the average price? Show me properties in District 1.",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submitted = st.form_submit_button("📤 Send", type="primary")
        with col2:
            if sql_tool:
                show_sql = st.form_submit_button("🔍 View SQL")
        
        if submitted and user_input:
            with st.spinner("Agent is thinking..."):
                # Call agent with SQL capability
                agent = st.session_state.get('agent')
                if not agent:
                    agent = DataSchemaAgent(
                        session,
                        st.session_state.api_key,
                        st.session_state.model
                    )
                    st.session_state.agent = agent
                
                # Enable SQL if available
                if sql_tool and df_cleaned is not None:
                    # Monkey patch to add SQL tool
                    agent.sql_tool = sql_tool
                    agent.df_cleaned = df_cleaned
                    agent.db_path = sql_tool.db_path
                
                # Get response
                try:
                    response = agent.chat(user_input)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Quick questions
    st.divider()
    st.subheader("💡 Quick Questions")
    
    if sql_tool:
        quick_questions = [
            ("📊 Data Overview", "Summarize the data for me"),
            ("💰 Average Price", "What is the average price?"),
            ("📈 Top 5 Items", "Show me the top 5 most expensive items"),
            ("🏘️ Count by Category", "How many items in each category?"),
            ("🔍 Missing Values", "Which columns have missing values?"),
            ("💡 Insights", "What are key insights about this dataset?")
        ]
    else:
        quick_questions = [
            ("📋 Schema Summary", "Summarize the schema"),
            ("❓ Missing Values", "Which columns have missing values?"),
            ("🧹 Cleaning Done", "What cleaning was applied?"),
            ("💡 Improvements", "Suggest improvements")
        ]
    
    cols = st.columns(3)
    for i, (label, question) in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(label, key=f"qq_{i}", use_container_width=True):
                with st.spinner("Thinking..."):
                    agent = st.session_state.get('agent')
                    if not agent:
                        agent = DataSchemaAgent(
                            session,
                            st.session_state.api_key,
                            st.session_state.model
                        )
                        st.session_state.agent = agent
                    
                    if sql_tool:
                        agent.sql_tool = sql_tool
                        agent.df_cleaned = df_cleaned
                        agent.db_path = sql_tool.db_path
                    
                    agent.chat(question)
                    st.rerun()
    
    # SQL Query section
    if sql_tool:
        st.divider()
        st.subheader("🔍 Direct SQL Query")
        st.caption("Advanced: Execute SQL directly (for debugging)")
        
        with st.expander("Execute Custom SQL"):
            sql_query = st.text_area(
                "SQL Query",
                value="SELECT * FROM data LIMIT 10",
                height=100,
                help="Only SELECT queries allowed"
            )
            
            if st.button("▶️ Execute"):
                result_df, error = sql_tool.execute_query(sql_query)
                
                if error:
                    st.error(error)
                else:
                    st.success(f"✅ Query returned {len(result_df)} rows")
                    st.dataframe(result_df, use_container_width=True)

# ============================================================================
# Tab 6: Checkpoints & History
# ============================================================================

def tab_checkpoints():
    """Checkpoints and history tab"""
    st.header("💾 Checkpoints & History")
    
    if not st.session_state.session:
        st.warning("⚠️ No active session")
        return
    
    session = st.session_state.session
    
    # Checkpoints
    if session.checkpoints:
        st.subheader("📸 Data Checkpoints")
        
        checkpoint_data = []
        for cp in session.checkpoints:
            checkpoint_data.append({
                "Checkpoint ID": cp.checkpoint_id,
                "Stage": cp.stage,
                "Timestamp": cp.timestamp[:19],
                "Rows": cp.shape[0],
                "Columns": cp.shape[1],
                "Description": cp.description
            })
        
        df_checkpoints = pd.DataFrame(checkpoint_data)
        st.dataframe(df_checkpoints, use_container_width=True)
        
        # Load checkpoint
        st.divider()
        selected_cp = st.selectbox(
            "Select checkpoint to view",
            options=[cp.checkpoint_id for cp in session.checkpoints]
        )
        
        if selected_cp and st.button("📂 Load Checkpoint"):
            checkpoint = next(cp for cp in session.checkpoints if cp.checkpoint_id == selected_cp)
            
            with st.spinner("Loading checkpoint..."):
                df_checkpoint = st.session_state.session_manager.load_checkpoint(checkpoint)
                st.success(f"✅ Loaded: {checkpoint.description}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", df_checkpoint.shape[0])
                with col2:
                    st.metric("Columns", df_checkpoint.shape[1])
                
                st.dataframe(df_checkpoint.head(20), use_container_width=True)
                
                # Download option
                csv = df_checkpoint.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download as CSV",
                    data=csv,
                    file_name=f"{selected_cp}.csv",
                    mime="text/csv"
                )
    else:
        st.info("No checkpoints yet")
    
    # History
    st.divider()
    st.subheader("📜 Refinement History")
    
    if session.history:
        for entry in reversed(session.history[-20:]):  # Show last 20
            with st.expander(f"**{entry.action}** - {entry.timestamp[:19]}"):
                st.write(f"**Schema Version:** {entry.schema_version}")
                st.json(entry.details)
    else:
        st.info("No history yet")


# ============================================================================
# Tab 7: Export
# ============================================================================

def tab_export():
    """Export tab"""
    st.header("📤 Export Results")
    
    if not st.session_state.session:
        st.warning("⚠️ No active session")
        return
    
    session = st.session_state.session
    
    st.subheader("📋 Final Schema")
    
    if session.schema:
        # Schema as table
        schema_data = []
        for col_name, schema_col in session.schema.items():
            schema_data.append({
                "Column": col_name,
                "Description": schema_col.description,
                "Semantic Type": schema_col.semantic_type,
                "Physical Type": schema_col.physical_type,
                "Original Type": schema_col.original_type,
                "Unit": schema_col.unit or "N/A",
                "Required": schema_col.is_required
            })
        
        df_schema = pd.DataFrame(schema_data)
        st.dataframe(df_schema, use_container_width=True)
        
        # Summary stats
        st.divider()
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Columns", len(session.schema))
        with col2:
            st.metric("Transformations Applied", len(session.applied_transformations))
        with col3:
            st.metric("Cleaning Rules Applied", len(session.applied_cleaning_rules))
        with col4:
            st.metric("Schema Version", session.schema_version)
        
        # Download schema JSON
        st.divider()
        schema_json = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "schema_version": session.schema_version,
            "applied_transformations": session.applied_transformations,
            "applied_cleaning_rules": session.applied_cleaning_rules,
            "columns": {k: v.to_dict() for k, v in session.schema.items()}
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="⬇️ Download Schema (JSON)",
                data=json.dumps(schema_json, indent=2, ensure_ascii=False),
                file_name=f"schema_{session.session_id}.json",
                mime="application/json"
            )
        
        # Download full session
        with col2:
            st.download_button(
                label="⬇️ Download Full Session (JSON)",
                data=json.dumps(session.to_dict(), indent=2, ensure_ascii=False, default=str),
                file_name=f"session_{session.session_id}.json",
                mime="application/json"
            )
    else:
        st.info("No schema generated yet")


# ============================================================================
# Main App
# ============================================================================

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Data Schema Pipeline V5")
        st.write("Complete workflow + Question-Driven Schema Generation")

        st.divider()
        
        # Session info
        if st.session_state.session:
            st.success("✅ Session Active")
            st.write(f"**Session ID:** {st.session_state.session.session_id}")
            st.write(f"**Sources:** {len(st.session_state.session.sources)}")
            st.write(f"**Schema Version:** {st.session_state.session.schema_version}")
            
            st.divider()
            
            # Quick stats
            st.metric("Transformations", len(st.session_state.session.applied_transformations))
            st.metric("Cleaning Rules", len(st.session_state.session.applied_cleaning_rules))
            st.metric("Questions Answered", len(st.session_state.session.answers))
            st.metric("Checkpoints", len(st.session_state.session.checkpoints))
            st.metric("Agent Messages", len(st.session_state.session.agent_conversations))
        else:
            st.info("No active session")
        
        st.divider()
        
        # Help
        with st.expander("❓ Help"):
            st.markdown("""
            **Workflow:**
            1. **Ingestion**: Upload files
            2. **Structure**: Fix structure issues
            3. **Type Cleaning**: Convert formatted numbers
            4. **Schema**: Generate schema (define questions/fields if needed)
            5. **Agent Q&A**: Chat & query your data with SQL
            6. **Checkpoints**: Review transformations
            7. **Export**: Download results
            """)
    
    # Main content - tabs
    tabs = st.tabs([
        "📁 Ingestion",
        "🔍 Structure Analysis",
        "🧹 Type Cleaning",
        "📋 Schema Generation",
        "🤖 Agent Q&A",
        "💾 Checkpoints",
        "📤 Export"
    ])

    with tabs[0]:
        tab_ingestion()

    with tabs[1]:
        tab_structure_analysis()

    with tabs[2]:
        tab_type_cleaning()

    with tabs[3]:
        tab_schema_generation()

    with tabs[4]:
        tab_agent_qa()

    with tabs[5]:
        tab_checkpoints()

    with tabs[6]:
        tab_export()


if __name__ == "__main__":
    main()