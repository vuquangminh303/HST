import os
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sqlite3
import re
import logging

# Import from pipeline
from hst_agent import (
    DataIngestor, StructureAnalyzer, ProfileGenerator, SchemaGenerator,
    SessionManager, RefinementEngine, DataSource, Session, Transformation,
    Question, Answer, ColumnProfile, ColumnSchema, DataFrameCheckpoint,
    TypeInferenceEngine, CleaningRule, DataSchemaAgent, AgentMessage,
    UserQuestion, OutputField, QuestionSet, SchemaValidator, Scenario
)
# Import chat validation extension
from validation_chat_extension import (
    SQLQueryTool,
    setup_sql_tool,
    add_chat_validation_to_questions_tab,
    add_chat_validation_to_scenarios_tab,
    render_export_chat_button,
    render_chat_statistics
)
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=os.getenv("LOGLEVEL","INFO").upper())
logger = logging.getLogger(__name__)
# Page config
st.set_page_config(
    page_title="Text2SQL Agent",
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
    .validation-success {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #4CAF50;
        background-color: #e8f5e9;
        margin: 0.5rem 0;
    }
    .validation-error {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #f44336;
        background-color: #ffebee;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

def safe_display_dataframe(df, *args, **kwargs):
    """
    Safely display DataFrame in Streamlit, handling PyArrow conversion errors.
    Converts problematic columns to string to avoid mixed-type issues.
    """
    try:
        # Try direct display first
        st.dataframe(df, *args, **kwargs)
    except Exception as e:
        # If fails, convert object columns to string
        if "ArrowTypeError" in str(type(e).__name__) or "Expected bytes" in str(e):
            df_display = df.copy()
            for col in df_display.columns:
                if df_display[col].dtype == 'object':
                    df_display[col] = df_display[col].astype(str)
            st.dataframe(df_display, *args, **kwargs)
        else:
            # Re-raise if different error
            raise


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
    if 'validation_agent' not in st.session_state:
        st.session_state.validation_agent = None
    # Multi-table state
    if 'multi_sql_tool' not in st.session_state:
        st.session_state.multi_sql_tool = None
    if 'selected_tables_for_query' not in st.session_state:
        st.session_state.selected_tables_for_query = []


# ============================================================================
# Multi-Table SQL Query Tool - HỖ TRỢ QUERY NHIỀU BẢNG
# ============================================================================

class MultiTableSQLQueryTool:
    """
    SQL Query Tool hỗ trợ nhiều bảng.
    Cho phép người dùng query trên nhiều DataFrame/bảng cùng lúc.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tables: Dict[str, Dict[str, Any]] = {}
        self.conn = None
        self._create_connection()
    
    def _create_connection(self):
        try:
            if self.conn:
                self.conn.close()
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        except Exception as e:
            raise RuntimeError(f"Failed to create database connection: {str(e)}")
    
    def _ensure_connection(self):
        try:
            if not self.conn:
                self._create_connection()
            self.conn.execute("SELECT 1")
        except:
            self._create_connection()
    
    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table name for SQL"""
        safe_name = re.sub(r'[^\w]', '_', str(name))
        if safe_name and safe_name[0].isdigit():
            safe_name = 't_' + safe_name
        return safe_name or 'data'
    
    def add_table(self, table_name: str, df: pd.DataFrame, schema_info: Dict = None) -> bool:
        """Thêm hoặc cập nhật một bảng vào database"""
        try:
            safe_name = self._sanitize_table_name(table_name)
            df_hash = hash(str(df.values.tobytes()))
            
            self._ensure_connection()
            df.to_sql(safe_name, self.conn, if_exists='replace', index=False)
            
            self.tables[safe_name] = {
                "df_hash": df_hash,
                "columns": list(df.columns),
                "row_count": len(df),
                "original_name": table_name,
                "schema_info": schema_info or {}
            }
            return True
        except Exception as e:
            logger.error(f"Failed to add table '{table_name}': {str(e)}")
            return False
    
    def remove_table(self, table_name: str) -> bool:
        """Xóa một bảng khỏi database"""
        try:
            safe_name = self._sanitize_table_name(table_name)
            if safe_name not in self.tables:
                return False
            self._ensure_connection()
            self.conn.execute(f"DROP TABLE IF EXISTS [{safe_name}]")
            del self.tables[safe_name]
            return True
        except Exception as e:
            return False
    
    def execute_query(self, query: str) -> tuple:
        """Thực thi SQL query"""
        try:
            if not query.upper().strip().startswith('SELECT'):
                return pd.DataFrame(), "Chỉ cho phép SELECT queries"
            self._ensure_connection()
            result_df = pd.read_sql_query(query, self.conn)
            return result_df, None
        except Exception as e:
            return pd.DataFrame(), f"SQL Error: {str(e)}"
    
    def get_tables_info(self) -> Dict[str, Any]:
        """Lấy thông tin tất cả các bảng"""
        return {
            "tables": [
                {
                    "name": name,
                    "original_name": info.get("original_name", name),
                    "columns": info.get("columns", []),
                    "row_count": info.get("row_count", 0),
                }
                for name, info in self.tables.items()
            ],
            "total_tables": len(self.tables)
        }
    
    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None


def setup_multi_table_sql_tool(session, sources_dfs: Dict[str, pd.DataFrame], 
                                schemas: Dict[str, Dict] = None) -> MultiTableSQLQueryTool:
    """
    Setup hoặc cập nhật multi-table SQL tool.
    
    Args:
        session: Session hiện tại
        sources_dfs: Dict mapping source_id -> DataFrame
        schemas: Dict mapping source_id -> schema dict (optional)
    """
    combined_hash = hash(tuple(
        hash(str(df.values.tobytes())) 
        for df in sources_dfs.values()
    ))
    
    need_recreate = False
    if 'multi_sql_tool' not in st.session_state or st.session_state.multi_sql_tool is None:
        need_recreate = True
    elif 'multi_sql_tool_hash' not in st.session_state:
        need_recreate = True
    elif st.session_state.multi_sql_tool_hash != combined_hash:
        need_recreate = True
    
    if need_recreate:
        db_dir = Path("./agent_databases")
        db_dir.mkdir(exist_ok=True)
        db_path = db_dir / f"multi_data_{session.session_id}.db"
        
        if 'multi_sql_tool' in st.session_state and st.session_state.multi_sql_tool:
            try:
                st.session_state.multi_sql_tool.close()
            except:
                pass
        
        sql_tool = MultiTableSQLQueryTool(str(db_path))
        
        for source_id, df in sources_dfs.items():
            schema_info = {}
            if schemas and source_id in schemas:
                schema_info = {
                    col: {
                        "description": col_schema.description if hasattr(col_schema, 'description') else "",
                        "semantic_type": col_schema.semantic_type if hasattr(col_schema, 'semantic_type') else "",
                    }
                    for col, col_schema in schemas[source_id].items()
                }
            sql_tool.add_table(source_id, df, schema_info)
        
        st.session_state.multi_sql_tool = sql_tool
        st.session_state.multi_sql_tool_hash = combined_hash
        
        return sql_tool
    
    return st.session_state.multi_sql_tool

def setup_sql_tool(session, df_cleaned):

    """Setup or update SQL tool for validation"""
    current_df_hash = hash(str(df_cleaned.values.tobytes()))
    
    need_recreate = False
    if 'sql_tool' not in st.session_state or st.session_state.sql_tool is None:
        need_recreate = True
    elif 'sql_tool_df_hash' not in st.session_state:
        need_recreate = True
    elif st.session_state.sql_tool_df_hash != current_df_hash:
        need_recreate = True
    
    if need_recreate:
        from pathlib import Path
        db_dir = Path("./agent_databases")
        db_dir.mkdir(exist_ok=True)
        db_path = db_dir / f"data_{session.session_id}.db"
        
        if 'sql_tool' in st.session_state and st.session_state.sql_tool:
            try:
                st.session_state.sql_tool.close()
            except:
                pass
        
        sql_tool = SQLQueryTool(str(db_path), df_cleaned)
        st.session_state.sql_tool = sql_tool
        st.session_state.sql_tool_df_hash = current_df_hash
        
        return sql_tool, True
    
    return st.session_state.sql_tool, False
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
        api_key = os.getenv("OPENAI_API_KEY")
        st.subheader("Settings")
        if not api_key:

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
        
        safe_display_dataframe(pd.DataFrame(source_data), width='stretch')
        
        # Preview
        st.subheader("👁️ Data Preview")
        selected_source = st.selectbox(
            "Select source to preview",
            options=[s.source_id for s in st.session_state.sources]
        )
        
        if selected_source and selected_source in st.session_state.raw_dfs:
            df = st.session_state.raw_dfs[selected_source]
            safe_display_dataframe(df.head(20), width='stretch')


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
            
            structure_info, transformations, questions, is_clean,issues = analyzer.analyze_structure(df_raw)
            
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
                            try:
                                df_raw = st.session_state.raw_dfs[selected_source_id]
                                df_current = st.session_state.clean_dfs.get(
                                    selected_source_id, 
                                    st.session_state.raw_dfs[selected_source_id]
                                )
                                df_transformed = StructureAnalyzer.apply_transformation(df_raw, trans)
                                
                                # ✅ VALIDATION
                                validation_passed = True
                                error_messages = []
                                
                                if df_transformed.columns.duplicated().any():
                                    duplicates = df_transformed.columns[df_transformed.columns.duplicated()].tolist()
                                    error_messages.append(f"🔴 Duplicate columns: {duplicates}")
                                    validation_passed = False
                                
                                empty_cols = [col for col in df_transformed.columns if not str(col).strip()]
                                if empty_cols:
                                    error_messages.append(f"🔴 Empty column names detected")
                                    validation_passed = False
                                
                                if not validation_passed:
                                    st.error("❌ Transformation validation failed:")
                                    for msg in error_messages:
                                        st.write(msg)
                                else:
                                    # Save và update như cũ
                                    st.session_state.session_manager.save_checkpoint(
                                        session, df_transformed, f"transform_{trans.id}",
                                        f"After: {trans.description}"
                                    )
                                    st.session_state.clean_dfs[selected_source_id] = df_transformed
                                    trans.applied = True
                                    session.applied_transformations.append(trans.id)
                                    st.success(f"✅ Applied: {trans.description}")
                                    st.rerun()
                                    
                            except ValueError as e:
                                st.error(f"❌ Validation Error: {str(e)}")
                                if "duplicate" in str(e).lower():
                                    st.warning("💡 Try selecting a different row as header or manually rename columns")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        st.divider()
        st.subheader("🛠️ Custom Transformations")
        st.info("💡 Describe any transformation you want in natural language - filtering, aggregation, calculations, etc.")

        with st.expander("➕ Add Custom Transformation", expanded=False):
            custom_trans_text = st.text_area(
                "Describe the transformation",
                placeholder="""Examples:
        - Use row 2 as header and skip first row
        - Drop column 'Unnamed: 0'
        - Rename column 'old_name' to 'new_name'
        - Filter rows where sales > 1000
        - Group by category and calculate sum of revenue
        - Create new column 'profit' = 'revenue' - 'cost'
        - Sort by date descending
        - Keep only rows where status is 'active'""",
                height=150,
                key="custom_trans_nl"
            )

            if st.button("🔄 Apply Custom Transformation", key="apply_custom_trans_nl", type="primary"):
                if not custom_trans_text.strip():
                    st.error("Please enter a transformation description")
                else:
                    with st.spinner("Processing transformation..."):
                        try:
                            # Initialize analyzer
                            analyzer = StructureAnalyzer(
                                api_key=st.session_state.api_key,
                                model=st.session_state.model
                            )
                            
                            # Get current dataframe
                            df_current = st.session_state.clean_dfs.get(
                                selected_source_id,
                                st.session_state.raw_dfs.get(selected_source_id)
                            )
                            
                            # Save checkpoint BEFORE transformation for undo
                            if 'transformation_history' not in st.session_state:
                                st.session_state.transformation_history = {}
                            
                            if selected_source_id not in st.session_state.transformation_history:
                                st.session_state.transformation_history[selected_source_id] = []
                            
                            # Store current state
                            checkpoint = {
                                'df': df_current.copy(),
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'description': custom_trans_text[:100]  # First 100 chars
                            }
                            st.session_state.transformation_history[selected_source_id].append(checkpoint)
                            
                            # Apply custom transformation using LLM
                            df_transformed = analyzer.custom_free_transform(
                                user_input=custom_trans_text,
                                df=df_current
                            )
                            
                            # Validation
                            validation_passed = True
                            error_messages = []
                            
                            if df_transformed.columns.duplicated().any():
                                duplicates = df_transformed.columns[df_transformed.columns.duplicated()].tolist()
                                error_messages.append(f"🔴 Duplicate columns: {duplicates}")
                                validation_passed = False
                            
                            empty_cols = [col for col in df_transformed.columns if not str(col).strip()]
                            if empty_cols:
                                error_messages.append(f"🔴 Empty column names detected")
                                validation_passed = False
                            
                            if df_transformed.empty:
                                error_messages.append(f"🔴 Transformation resulted in empty DataFrame")
                                validation_passed = False
                            
                            if not validation_passed:
                                st.error("❌ Transformation validation failed:")
                                for msg in error_messages:
                                    st.write(msg)
                                # Remove the checkpoint since transformation failed
                                st.session_state.transformation_history[selected_source_id].pop()
                            else:
                                # Save successful transformation
                                trans_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                
                                st.session_state.session_manager.save_checkpoint(
                                    session, 
                                    df_transformed, 
                                    trans_id,
                                    f"Custom: {custom_trans_text[:50]}..."
                                )
                                
                                # Update clean dataframe
                                st.session_state.clean_dfs[selected_source_id] = df_transformed
                                
                                # Create transformation record
                                trans = Transformation(
                                    id=trans_id,
                                    type="custom",
                                    description=custom_trans_text,
                                    params={},
                                    confidence=1.0,
                                    applied=True
                                )
                                session.transformations.append(trans)
                                session.applied_transformations.append(trans_id)
                                
                                st.success(f"✅ Applied: {custom_trans_text[:100]}...")
                                st.info(f"📊 Result: {df_current.shape} → {df_transformed.shape}")
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Transformation failed: {str(e)}")
                            st.write("Please try rephrasing your request or check the data.")
                            # Remove the checkpoint since transformation failed
                            if selected_source_id in st.session_state.transformation_history:
                                if st.session_state.transformation_history[selected_source_id]:
                                    st.session_state.transformation_history[selected_source_id].pop()

        # Undo Section
        if 'transformation_history' in st.session_state and selected_source_id in st.session_state.transformation_history:
            history = st.session_state.transformation_history[selected_source_id]
            
            if history:
                st.divider()
                st.subheader("⏮️ Transformation History")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.info(f"📝 {len(history)} transformation(s) applied. You can undo to any previous state.")
                
                with col2:
                    if st.button("🔄 Undo Last", key="undo_last_trans", type="secondary"):
                        if history:
                            # Get the last checkpoint (before last transformation)
                            last_checkpoint = history.pop()
                            
                            # Restore the dataframe
                            st.session_state.clean_dfs[selected_source_id] = last_checkpoint['df']
                            
                            # Save this undo as a checkpoint
                            st.session_state.session_manager.save_checkpoint(
                                session,
                                last_checkpoint['df'],
                                f"undo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                f"Undo: {last_checkpoint['description']}"
                            )
                            
                            st.success(f"✅ Undone: {last_checkpoint['description']}")
                            st.rerun()
                
                # Show history
                with st.expander(f"📜 View History ({len(history)} items)", expanded=False):
                    for i, checkpoint in enumerate(reversed(history)):
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            st.write(f"**{len(history) - i}.** {checkpoint['description']}")
                        
                        with col2:
                            st.caption(f"🕒 {checkpoint['timestamp']}")
                        
                        with col3:
                            if st.button(f"⏮️ Restore", key=f"restore_{i}"):
                                # Get index in original list
                                original_idx = len(history) - 1 - i
                                
                                # Restore to this checkpoint
                                restored_df = history[original_idx]['df']
                                st.session_state.clean_dfs[selected_source_id] = restored_df
                                
                                # Remove all checkpoints after this one
                                st.session_state.transformation_history[selected_source_id] = history[:original_idx]
                                
                                # Save restore as checkpoint
                                st.session_state.session_manager.save_checkpoint(
                                    session,
                                    restored_df,
                                    f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                    f"Restored to: {checkpoint['description']}"
                                )
                                
                                st.success(f"✅ Restored to checkpoint {original_idx + 1}")
                                st.rerun()
                    
                    # Clear all history option
                    st.divider()
                    if st.button("🗑️ Clear History", key="clear_history"):
                        st.session_state.transformation_history[selected_source_id] = []
                        st.success("✅ History cleared")
                        st.rerun()

        st.divider()
        # Show before/after comparison
        if selected_source_id in st.session_state.clean_dfs:
            st.divider()
            st.subheader("📊 Before/After Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before (Raw)**")
                df_raw = st.session_state.raw_dfs[selected_source_id]
                safe_display_dataframe(df_raw.head(10), width='stretch')
                st.caption(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
            
            with col2:
                st.write("**After (Transformed)**")
                df_clean = st.session_state.clean_dfs[selected_source_id]
                safe_display_dataframe(df_clean.head(10), width='stretch')
                st.caption(f"Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")

        # Custom Transformations Section
        st.divider()


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
            # Format data issues
            if profile.data_issues:
                issues_str = ", ".join([issue['type'] for issue in profile.data_issues])
                issues_display = f"⚠️ {issues_str}"
            elif profile.has_thousand_separator:
                issues_display = "⚠️ thousand_separator"
            else:
                issues_display = "✅ Clean"

            type_data.append({
                "Column": col_name,
                "Current Type": profile.pandas_dtype,
                "Inferred Type": profile.inferred_type,
                "Data Issues": issues_display,
                "Example Value": profile.sample_raw_values[0] if profile.sample_raw_values else ""
            })

        safe_display_dataframe(pd.DataFrame(type_data), width='stretch')

        # Show detailed issues breakdown
        if any(profile.data_issues for profile in results['profiles'].values()):
            st.divider()
            st.subheader("🔍 Detailed Data Quality Issues")

            for col_name, profile in results['profiles'].items():
                if profile.data_issues:
                    with st.expander(f"**{col_name}** - {len(profile.data_issues)} issue(s) detected", expanded=False):
                        for issue in profile.data_issues:
                            st.markdown(f"**Issue Type:** `{issue['type']}`")
                            st.markdown(f"**Description:** {issue['description']}")
                            st.markdown(f"**Examples:** {issue['examples'][:3]}")
                            st.markdown(f"**Suggested Action:** `{issue['action']}`")
                            st.divider()
        
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
            
            safe_display_dataframe(df_cleaned.head(20), width='stretch')
            
            # Show dtype changes
            st.write("**Data Types After Cleaning:**")
            dtype_df = pd.DataFrame({
                "Column": df_cleaned.columns,
                "Type": [str(dtype) for dtype in df_cleaned.dtypes]
            })
            safe_display_dataframe(dtype_df, width='stretch')

    # Custom Cleaning Rules Section
    st.divider()
    st.subheader("🛠️ Custom Cleaning Rules")
    st.info("💡 Define your own data cleaning rules using natural language or JSON format")

    with st.expander("➕ Add Custom Cleaning Rule", expanded=False):
        custom_clean_method = st.radio(
            "Input Method",
            ["Natural Language", "JSON"],
            key="custom_clean_method",
            horizontal=True
        )

        if custom_clean_method == "Natural Language":
            custom_clean_text = st.text_area(
                "Describe the cleaning rule",
                placeholder="""Examples:
- Convert column 'Price' to float, remove thousand separator ','
- Convert 'Age' to integer
- Strip whitespace from 'Name' column
- Normalize 'Status' column to lowercase
- Convert 'Date' to datetime format""",
                height=150,
                key="custom_clean_nl"
            )

            if st.button("🔄 Parse & Apply", key="apply_custom_clean_nl"):
                if not custom_clean_text.strip():
                    st.error("Please enter a cleaning rule description")
                else:
                    with st.spinner("Parsing cleaning rule..."):
                        from openai import OpenAI
                        client = OpenAI(api_key=st.session_state.api_key)

                        parse_prompt = f"""Parse this data cleaning request into a structured cleaning rule.

Request: {custom_clean_text}

Provide a JSON response with this format:
{{
  "column": "column_name",
  "action": "convert_to_int" | "convert_to_float" | "convert_to_datetime" | "strip_whitespace" | "normalize_case",
  "description": "Human-readable description",
  "params": {{...params if needed...}}
}}

Examples:
- "Convert Price to float, remove comma" → {{"column": "Price", "action": "convert_to_float", "params": {{"thousand_separator": ","}}, "description": "Convert Price to float (remove comma)"}}
- "Convert Age to integer" → {{"column": "Age", "action": "convert_to_int", "params": {{}}, "description": "Convert Age to integer"}}
- "Strip whitespace from Name" → {{"column": "Name", "action": "strip_whitespace", "params": {{}}, "description": "Strip whitespace from Name"}}
- "Normalize Status to lowercase" → {{"column": "Status", "action": "normalize_case", "params": {{"case": "lower"}}, "description": "Normalize Status to lowercase"}}

Respond with ONLY the JSON object, no explanation."""

                        try:
                            response = client.chat.completions.create(
                                model=st.session_state.model,
                                messages=[
                                    {"role": "system", "content": "You are a data cleaning rule parser. Convert natural language requests into structured cleaning rule objects."},
                                    {"role": "user", "content": parse_prompt}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.3
                            )

                            parsed = json.loads(response.choices[0].message.content)

                            # Create CleaningRule object
                            rule = CleaningRule(
                                id=f"custom_clean_{datetime.now().strftime('%H%M%S')}",
                                column=parsed["column"],
                                action=parsed["action"],
                                description=parsed["description"],
                                params=parsed.get("params", {}),
                                applied=False
                            )

                            # Apply cleaning rule
                            df_current = st.session_state.cleaned_dfs.get(
                                selected_source_id,
                                st.session_state.clean_dfs.get(
                                    selected_source_id,
                                    df
                                )
                            )

                            df_cleaned = TypeInferenceEngine.apply_cleaning_rule(df_current, rule)

                            # Save
                            st.session_state.session_manager.save_checkpoint(
                                session, df_cleaned, f"custom_clean_{rule.id}",
                                f"Custom: {rule.description}"
                            )

                            st.session_state.cleaned_dfs[selected_source_id] = df_cleaned
                            rule.applied = True
                            session.cleaning_rules.append(rule)
                            session.applied_cleaning_rules.append(rule.id)

                            st.success(f"✅ Applied: {rule.description}")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Failed to parse or apply cleaning rule: {str(e)}")
                            st.write("Please try rephrasing or use JSON format instead.")

        else:  # JSON mode
            custom_clean_json = st.text_area(
                "Cleaning Rule JSON",
                placeholder="""{
  "column": "Price",
  "action": "convert_to_float",
  "params": {
    "thousand_separator": ",",
    "decimal_separator": "."
  },
  "description": "Convert Price to float (remove comma separator)"
}""",
                height=200,
                key="custom_clean_json"
            )

            if st.button("🔄 Parse & Apply", key="apply_custom_clean_json"):
                if not custom_clean_json.strip():
                    st.error("Please enter cleaning rule JSON")
                else:
                    try:
                        parsed = json.loads(custom_clean_json)

                        # Create CleaningRule object
                        rule = CleaningRule(
                            id=f"custom_clean_{datetime.now().strftime('%H%M%S')}",
                            column=parsed["column"],
                            action=parsed["action"],
                            description=parsed.get("description", "Custom cleaning rule"),
                            params=parsed.get("params", {}),
                            applied=False
                        )

                        # Apply cleaning rule
                        df_current = st.session_state.cleaned_dfs.get(
                            selected_source_id,
                            st.session_state.clean_dfs.get(
                                selected_source_id,
                                df
                            )
                        )

                        df_cleaned = TypeInferenceEngine.apply_cleaning_rule(df_current, rule)

                        # Save
                        st.session_state.session_manager.save_checkpoint(
                            session, df_cleaned, f"custom_clean_{rule.id}",
                            f"Custom: {rule.description}"
                        )

                        st.session_state.cleaned_dfs[selected_source_id] = df_cleaned
                        rule.applied = True
                        session.cleaning_rules.append(rule)
                        session.applied_cleaning_rules.append(rule.id)

                        st.success(f"✅ Applied: {rule.description}")
                        st.rerun()

                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {str(e)}")
                    except Exception as e:
                        st.error(f"Failed to apply cleaning rule: {str(e)}")


# ============================================================================
# Tab 4: Question Collection
# ============================================================================


def tab_question_collection():
    """
    Step 5: Schema Validation & Refinement
    Allows users to define questions AFTER schema generation.
    Validates if the schema can answer them, and triggers refinement if not.
    """
    st.header("✅ Schema Validation & Refinement")

    if not st.session_state.session:
        st.warning("⚠️ No active session")
        return
        
    session = st.session_state.session
    
    # Check if schema exists first
    if not session.schema:
        st.warning("⚠️ Please generate a schema first (Tab 4: Schema Generation).")
        return

    st.markdown("""
    **Validate your Schema:**
    1. Enter questions you plan to ask the data.
    2. The system will check if the current schema contains enough information.
    3. If gaps are found, you can refine the schema right here.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Define User Questions")
        
        # Load existing questions if any
        current_qs = ""
        if session.question_set and session.question_set.user_questions:
            current_qs = "\n".join([q.question for q in session.question_set.user_questions])

        questions_text = st.text_area(
            "What questions will you ask?",
            value=current_qs,
            placeholder="e.g., What is the total revenue?\nHow is profit calculated?\nFilter by specific region...",
            height=200,
            key="val_questions_input"
        )
        
        if st.button("🔍 Validate Schema against Questions", type="primary"):
            if not questions_text.strip():
                st.error("Please enter at least one question.")
                return
                
            with st.spinner("Validating schema capabilities..."):
                # 1. Update Session QuestionSet
                questions_list = [q.strip() for q in questions_text.split('\n') if q.strip()]
                
                if not session.question_set:
                    session.question_set = QuestionSet()
                
                session.question_set.user_questions = [
                    UserQuestion(id=f"uq_{i}", question=q) for i, q in enumerate(questions_list)
                ]
                
                # 2. Run Validator
                validator = SchemaValidator(
                    api_key=st.session_state.api_key,
                    model=st.session_state.model
                )
                
                # Prepare data for validation
                selected_source_id = session.current_source_id
                
                df = st.session_state.cleaned_dfs.get(
                    selected_source_id, 
                    st.session_state.raw_dfs.get(selected_source_id)
                )
                
                
                if df is None:
                    st.error(f"Could not find data for source: {selected_source_id}")
                    return

                sample_rows = ProfileGenerator.get_sample_rows(df)
                
                is_sufficient, add_questions, report = validator.validate_schema_for_questions(
                    schema=session.schema,
                    profiles=session.profiles, 
                    question_set=session.question_set,
                    sample_rows=sample_rows
                )
                
                st.session_state.validation_result = {
                    "is_sufficient": is_sufficient,
                    "additional_questions": add_questions,
                    "report": report,
                    "timestamp": datetime.now().isoformat()
                }

    with col2:
        st.subheader("2. Validation Report")
        
        if "validation_result" in st.session_state:
            res = st.session_state.validation_result
            
            # Status Banner
            if res["is_sufficient"]:
                st.success("✅ Schema is Sufficient!")
                st.write("The current schema contains enough information to answer your questions.")
            else:
                st.error("⚠️ Schema Needs Refinement")
                st.write("Some information is missing or unclear based on your questions.")
            
            # Detailed Report
            with st.expander("📄 View Detailed Report", expanded=True):
                st.markdown(res["report"])
            
            # Refinement Loop
            if not res["is_sufficient"] and res["additional_questions"]:
                st.divider()
                st.subheader("3. Refinement Required")
                st.info("Please answer the following to update the schema:")
                
                # Initialize refinement answers storage
                if "refinement_answers" not in st.session_state:
                    st.session_state.refinement_answers = {}
                
                all_answered = True
                
                for i, q in enumerate(res["additional_questions"]):
                    st.write(f"**Q{i+1}: {q.question}**")
                    st.caption(f"Targeting: `{q.target}`")
                    
                    ans_key = f"val_ans_{q.id}"
                    default_value = st.session_state.refinement_answers.get(q.id, q.suggested_answer or "")
                    user_ans = st.text_input(
                        "Answer:", 
                        key=ans_key,
                        value=default_value,
                        help=f"Suggested: {q.suggested_answer}"
                    )
                    
                    if user_ans:
                        st.session_state.refinement_answers[q.id] = user_ans
                    else:
                        all_answered = False
                
                # Button Update Schema
                if st.button("💾 Update Schema", disabled=not all_answered):
                    with st.spinner("Updating schema..."):
                        engine = RefinementEngine(session)
                        
                        count = 0
                        for q in res["additional_questions"]:
                            ans_text = st.session_state.refinement_answers.get(q.id)
                            if ans_text:
                                answer = Answer(
                                    question_id=q.id,
                                    answer=ans_text,
                                    timestamp=datetime.now().isoformat()
                                )
                                
                                q_exists = False
                                for sq in session.questions:
                                    if sq.id == q.id:
                                        q_exists = True
                                        break
                                if not q_exists:
                                    session.questions.append(q)
                                
                                if engine.apply_answer(answer):
                                    session.answers.append(answer)
                                    new_logic = f"\n- [Business Logic] {questions_text} -> Rule: {answer.answer}"
                                    session.question_set.additional_notes += new_logic
                                    count += 1
                        if count > 0:
                            st.success(f"Updated schema with {count} new details!")
                            if "validation_result" in st.session_state:
                                del st.session_state.validation_result
                            if "refinement_answers" in st.session_state:
                                del st.session_state.refinement_answers
                            
                            st.rerun()
                        else:
                            st.warning("No changes applied. Please check your answers.")

        else:
            st.info("Enter questions and click Validate to see the report.")
    # ============================================================================
    # CHAT VALIDATION FOR QUESTIONS
    # ============================================================================
    if session.question_set and session.question_set.user_questions:
        # Get cleaned DataFrame
        selected_source_id = session.current_source_id
        df_cleaned = st.session_state.cleaned_dfs.get(
            selected_source_id,
            st.session_state.clean_dfs.get(
                selected_source_id,
                st.session_state.raw_dfs.get(selected_source_id)
            )
        )
        
        if df_cleaned is not None and session.schema:
            # Setup SQL tool
            sql_tool, recreated = setup_sql_tool(session, df_cleaned)
            if recreated:
                st.success("✅ SQL database sẵn sàng cho chat!")
            
            # Add chat interface
            add_chat_validation_to_questions_tab(session, df_cleaned, sql_tool)
            
            # Optional: Add statistics and export
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                render_chat_statistics("questions_validation")
            with col2:
                render_export_chat_button("questions_validation")
        else:
            st.divider()
            st.warning("⚠️ Vui lòng hoàn thành Schema Generation (Tab 4) trước khi test")
    st.divider()
    with st.expander("👀 View Current Schema Summary"):
        if session.schema:
            schema_data = []
            for col_name, schema_col in session.schema.items():
                schema_data.append({
                    "Column": col_name,
                    "Semantic Type": schema_col.semantic_type,
                    "Description": schema_col.description,
                    "Unit": schema_col.unit or ""
                })
            safe_display_dataframe(pd.DataFrame(schema_data), width='stretch')

# ============================================================================
# Tab 5: Schema Generation
# ============================================================================

def tab_schema_generation():
    """Schema generation tab (Now Step 4)"""
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

    st.markdown("""
    **Two-Step Schema Generation Process:**

    1. **Step 1**: Analyze data and generate clarification questions (units, formats, constraints, etc.)
    2. **Step 2**: Answer questions and generate final schema
    """)

    st.divider()

    # Scenarios Section - Guide Schema Generation
    if session.scenarios:
        st.subheader("🎯 Use Scenarios to Guide Schema Generation")
        st.info("You have defined scenarios. Select which ones to use for guiding schema generation.")

        # Select scenarios
        scenario_names = [s.name for s in session.scenarios]
        selected_scenario_names = st.multiselect(
            "Select scenarios to apply",
            options=scenario_names,
            default=[],
            help="Selected scenarios will be used to guide schema generation",
            key="schema_gen_scenarios"
        )

        if selected_scenario_names:
            # Show selected scenarios summary
            selected_scenarios = [s for s in session.scenarios if s.name in selected_scenario_names]

            with st.expander(f"📋 Selected Scenarios Summary ({len(selected_scenarios)})", expanded=True):
                for scenario in selected_scenarios:
                    st.write(f"**{scenario.name}**")
                    st.write(f"  Fields: {', '.join(scenario.selected_fields[:5])}{'...' if len(scenario.selected_fields) > 5 else ''}")
                    st.write(f"  Questions: {len(scenario.questions)}")

            # Convert scenarios to QuestionSet for schema generation
            if "scenarios_question_set" not in st.session_state:
                st.session_state.scenarios_question_set = None

            if st.button("🔄 Apply Scenarios to Schema Generation", key="apply_scenarios_to_schema"):
                # Combine all questions and output fields from selected scenarios
                all_questions = []
                all_notes = []

                for scenario in selected_scenarios:
                    all_notes.append(f"\n### Scenario: {scenario.name}")
                    all_notes.append(f"Description: {scenario.description}")
                    all_notes.append(f"Required Fields: {', '.join(scenario.selected_fields)}")

                    for q in scenario.questions:
                        all_questions.append(UserQuestion(
                            id=f"scenario_{scenario.id}_{len(all_questions)}",
                            question=q,
                            description=f"From scenario: {scenario.name}"
                        ))

                    if scenario.output_format:
                        all_notes.append(f"Expected Output Format: {json.dumps(scenario.output_format, indent=2)}")

                # Create or update QuestionSet
                if not session.question_set:
                    session.question_set = QuestionSet()

                # Merge with existing questions
                existing_questions = {q.question: q for q in session.question_set.user_questions}

                for new_q in all_questions:
                    if new_q.question not in existing_questions:
                        session.question_set.user_questions.append(new_q)

                # Add scenario notes
                scenario_notes = "\n".join(all_notes)
                if scenario_notes not in session.question_set.additional_notes:
                    session.question_set.additional_notes += "\n\n## Scenarios:\n" + scenario_notes

                st.success(f"✅ Applied {len(selected_scenarios)} scenarios to schema generation context!")
                st.session_state.scenarios_question_set = session.question_set
                st.rerun()

        st.divider()
    else:
        st.info("💡 Tip: Define scenarios in the 'Scenarios' tab to guide schema generation with your use cases.")
        st.divider()

    # Initialize state for clarification flow
    if 'clarification_questions' not in st.session_state:
        st.session_state.clarification_questions = {}

    if 'clarification_answers' not in st.session_state:
        st.session_state.clarification_answers = {}

    # ============================================================================
    # STEP 1: Generate Clarification Questions
    # ============================================================================
    st.subheader("📝 Step 1: Data Analysis & Clarification Questions")

    clarif_questions = st.session_state.clarification_questions.get(selected_source_id, [])

    if not clarif_questions:
        if st.button("🔍 Analyze Data & Generate Questions", type="primary"):
            with st.spinner("Analyzing data and generating clarification questions..."):
                profiles = ProfileGenerator.generate_profiles(df)
                sample_rows = ProfileGenerator.get_sample_rows(df)
                # Store profiles
                if 'profiles' not in st.session_state:
                    st.session_state.profiles = {}
                st.session_state.profiles[selected_source_id] = {
                    'profiles': profiles,
                    'sample_rows': sample_rows
                }

                # Generate clarification questions
                schema_gen = SchemaGenerator(
                    api_key=st.session_state.api_key,
                    model=st.session_state.model
                )
                questions = schema_gen.generate_clarification_questions(
                    profiles,
                    sample_rows,
                    question_set=session.question_set
                )

                st.session_state.clarification_questions[selected_source_id] = questions
                st.success(f"✅ Generated {len(questions)} clarification questions!")
                st.rerun()

        st.info("👆 Click the button above to analyze your data and generate clarification questions")

    else:
        st.success(f"✅ Generated {len(clarif_questions)} questions. Please answer them below.")

        st.divider()

        # ============================================================================
        # Display and Answer Clarification Questions
        # ============================================================================
        st.subheader("❓ Clarification Questions")
        st.write("Please answer these questions to help generate an accurate schema:")

        # Initialize answers dict for this source
        if selected_source_id not in st.session_state.clarification_answers:
            st.session_state.clarification_answers[selected_source_id] = {}

        answers_dict = st.session_state.clarification_answers[selected_source_id]

        for i, q in enumerate(clarif_questions):
            with st.expander(f"**Question {i+1}: {q.question}**", expanded=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**Target Field:** `{q.target}`")
                    if q.suggested_answer:
                        st.write(f"**Suggested Answer:** {q.suggested_answer}")

                    # Answer input
                    answer_value = st.text_input(
                        "Your answer:",
                        value=answers_dict.get(q.id, q.suggested_answer or ""),
                        key=f"answer_{q.id}_{selected_source_id}",
                        placeholder="Enter your answer..."
                    )

                    answers_dict[q.id] = answer_value

                with col2:
                    if answer_value:
                        st.success("✓ Answered")
                    else:
                        st.warning("⚠️ Empty")

        st.divider()

        # Show progress
        answered_count = sum(1 for ans in answers_dict.values() if ans)
        total_count = len(clarif_questions)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", total_count)
        with col2:
            st.metric("Answered", answered_count)
        with col3:
            progress = answered_count / total_count if total_count > 0 else 0
            st.metric("Progress", f"{progress:.0%}")

        # ============================================================================
        # STEP 2: Generate Schema
        # ============================================================================
        st.divider()
        st.subheader("🧠 Step 2: Generate Final Schema")

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🗑️ Reset & Start Over"):
                st.session_state.clarification_questions.pop(selected_source_id, None)
                st.session_state.clarification_answers.pop(selected_source_id, None)
                st.session_state.schema_results.pop(selected_source_id, None)
                st.success("✓ Reset! Please analyze data again.")
                st.rerun()

        with col2:
            can_generate = answered_count >= total_count * 0.5  # At least 50% answered

            if not can_generate:
                st.warning(f"⚠️ Please answer at least {int(total_count * 0.5)} questions before generating schema")

            if st.button("🚀 Generate Final Schema", type="primary", disabled=not can_generate):
                with st.spinner("Generating schema with your answers..."):
                    # Get profiles
                    profiles_data = st.session_state.profiles.get(selected_source_id)
                    if not profiles_data:
                        st.error("Profiles not found. Please run Step 1 again.")
                        return

                    profiles = profiles_data['profiles']
                    sample_rows = profiles_data['sample_rows']

                    # Convert answers to Answer objects
                    clarification_answers = [
                        Answer(
                            question_id=q_id,
                            answer=ans,
                            timestamp=datetime.now().isoformat(),
                            applied=True
                        )
                        for q_id, ans in answers_dict.items() if ans
                    ]

                    # Generate schema
                    schema_gen = SchemaGenerator(
                        api_key=st.session_state.api_key,
                        model=st.session_state.model
                    )
                    schema, additional_questions = schema_gen.generate_schema(
                        profiles,
                        sample_rows,
                        question_set=session.question_set,
                        clarification_answers=clarification_answers
                    )

                    # Store results
                    if 'schema_results' not in st.session_state:
                        st.session_state.schema_results = {}

                    st.session_state.schema_results[selected_source_id] = {
                        'profiles': profiles,
                        'schema': schema,
                        'questions': additional_questions,
                        'clarification_answers': clarification_answers
                    }

                    # Update session
                    session.profiles.update(profiles)
                    session.schema.update(schema)
                    session.answers.extend(clarification_answers)

                    st.success(f"✅ Generated schema for {len(schema)} columns!")
                    st.rerun()

    if hasattr(st.session_state, 'schema_results') and selected_source_id in st.session_state.schema_results:
        results = st.session_state.schema_results[selected_source_id]
        st.divider()
        st.subheader("📊 Generated Schema")
        def normalize_key(s):
            import unicodedata
            # Chuyển về dạng Unicode chuẩn, bỏ khoảng trắng lạ, lowercase
            s = unicodedata.normalize('NFC', str(s))
            return s.replace('\xa0', ' ').replace(' ', '').lower()
        # Create tabs for each column
        column_names = list(results['schema'].keys())
        if column_names:
            tabs = st.tabs(column_names)
            for tab, col_name in zip(tabs, column_names):
                with tab:
                    profile = results['profiles'].get(col_name)
                    if profile is None:
                        target_key = normalize_key(col_name)
                        for raw_key, raw_profile in results['profiles'].items():
                            if normalize_key(raw_key) == target_key:
                                profile = raw_profile
                                break
                    
                    if profile is None:
                        profile = ColumnProfile(
                            name=col_name, pandas_dtype="unknown", inferred_type="unknown",
                            non_null_count=0, null_count=0, null_ratio=0.0, n_unique=0,
                            sample_values=[], sample_raw_values=[]
                        )
                        st.warning(f"⚠️ Không tìm thấy profile gốc cho cột: '{col_name}'. Dữ liệu hiển thị có thể không đầy đủ.")

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
        
        col1, col3 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("📤 Send", type="primary")
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
                
                try:
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in agent.query(user_input):
                        if isinstance(chunk, str):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        
                    message_placeholder.markdown(full_response)
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Quick questions
    st.divider()
    st.subheader("💡 Quick Questions")
    
    cols = st.columns(3)
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
                    safe_display_dataframe(result_df, width='stretch')


def tab_scenario_definition():
    """Scenario definition tab - define use cases with questions and output formats"""
    st.header("🎯 Scenario Definition")

    if not st.session_state.session:
        st.warning("⚠️ No active session")
        return

    if not st.session_state.session.schema:
        st.warning("⚠️ Please generate schema first (Tab 4: Schema Generation)")
        return

    session = st.session_state.session

    st.markdown("""
    **Define Scenarios (Use Cases):**
    A scenario represents a specific use case for your data.
    """)

    st.divider()

    # Create/Edit Scenario Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Create/Edit Scenario")

    with col2:
        if st.button("➕ New Scenario", key="btn_new_scenario"):
            if "current_scenario" not in st.session_state:
                st.session_state.current_scenario = None
            st.session_state.editing_scenario = True

    # Initialize session state for scenario editing
    if "editing_scenario" not in st.session_state:
        st.session_state.editing_scenario = False

    if "current_scenario" not in st.session_state:
        st.session_state.current_scenario = None

    # --- SCENARIO FORM (ĐÃ BỎ ST.FORM ĐỂ FIX LỖI UI DYNAMIC) ---
    if st.session_state.editing_scenario or st.session_state.current_scenario:
        current_scen = st.session_state.current_scenario
        
        st.info("💡 Đang chỉnh sửa kịch bản. Thay đổi sẽ cập nhật giao diện ngay lập tức.")
        
        # Container cho form để nhìn gọn gàng hơn
        with st.container():
            st.subheader("Scenario Details")

            # Basic info
            scenario_name = st.text_input(
                "Scenario Name *",
                value=current_scen.name if current_scen else "",
                placeholder="e.g., Sales Analysis",
                key="scenario_name_input"
            )

            scenario_desc = st.text_area(
                "Description",
                value=current_scen.description if current_scen else "",
                height=100,
                key="scenario_desc_input"
            )

            # Select fields
            st.subheader("📊 Select Relevant Fields")
            available_columns = list(session.schema.keys())
            default_selected = []

            selected_fields = st.multiselect(
                "Choose columns needed for this scenario (Optinal)",
                options=available_columns,
                default=default_selected,
                key="scenario_fields_input"
            )
            logger.info(f"SELECTED FIELDS {selected_fields}")
            # Questions
            st.subheader("❓ Questions")
            current_questions = "\n".join(current_scen.questions) if current_scen else ""
            questions_text = st.text_area(
                "Questions (one per line)",
                value=current_questions,
                height=150,
                key="scenario_questions_input"
            )

            # Output Format - PHẦN QUAN TRỌNG
            st.subheader("📤 Output Format")

            # Detect default method
            default_method_index = 0
            if current_scen and current_scen.output_format:
                if isinstance(current_scen.output_format, dict):
                    # Check if it looks like the text format wrapper
                    if current_scen.output_format.get("type") == "text_description" or \
                       (len(current_scen.output_format) == 1 and "description" in current_scen.output_format):
                        default_method_index = 1
            
            current_output_text = ""
            if current_scen and current_scen.output_format:
                current_output_text = current_scen.output_format.get("description", "")
            
            output_format_text = st.text_area(
                "Output Format Description",
                value=current_output_text,
                placeholder="Mô tả định dạng đầu ra mong muốn...",
                height=150,
                key="scenario_output_text"
            )

            # # Examples
            st.subheader("💡 Examples (Optional)")
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                current_ex_input = json.dumps(current_scen.example_input, indent=2) if current_scen and current_scen.example_input else ""
                example_input = st.text_area("Example Input Data", value=current_ex_input, key="scenario_ex_input", height=150)
            with col_ex2:
                current_ex_output = json.dumps(current_scen.example_output, indent=2) if current_scen and current_scen.example_output else ""
                example_output = st.text_area("Example Expected Output", value=current_ex_output, key="scenario_ex_output", height=150)

            # Buttons
            st.divider()
            col_submit, col_cancel = st.columns([1, 1])

            with col_submit:
                # Dùng st.button thường (không phải form_submit)
                submitted = st.button("💾 Save Scenario", type="primary", key="btn_save_scenario")

            with col_cancel:
                cancelled = st.button("❌ Cancel", key="btn_cancel_scenario")

            # --- LOGIC XỬ LÝ LƯU ---
            if submitted:
                if not scenario_name.strip():
                    st.error("⚠️ Vui lòng nhập tên kịch bản")

                elif not questions_text.strip():
                    st.error("⚠️ Vui lòng nhập ít nhất một câu hỏi")
                else:
                    # Parse Questions
                    questions_list = [q.strip() for q in questions_text.split('\n') if q.strip()]

                    output_format = {}
                        # Free Text
                    raw_text = st.session_state.get("scenario_output_text", "").strip()
                    if raw_text:
                        output_format = {
                            "type": "text_description",
                            "description": raw_text
                        }
                    else:
                        output_format = {"description": "No description provided"}

                    # Parse Examples
                    ex_input = None
                    if example_input.strip():
                        try: ex_input = json.loads(example_input)
                        except: pass
                    
                    ex_output = None
                    if example_output.strip():
                        try: ex_output = json.loads(example_output)
                        except: pass

                    # Save Logic
                    if current_scen:
                        scenario_id = current_scen.id
                        session.scenarios = [s for s in session.scenarios if s.id != scenario_id]
                    else:
                        scenario_id = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    new_scenario = Scenario(
                        id=scenario_id,
                        name=scenario_name,
                        description=scenario_desc,
                        selected_fields=selected_fields,
                        questions=questions_list,
                        output_format=output_format,
                        example_input=ex_input,
                        example_output=ex_output,
                        created_at=datetime.now().isoformat()
                    )

                    session.scenarios.append(new_scenario)
                    st.session_state.editing_scenario = False
                    st.session_state.current_scenario = None
                    st.success(f"✅ Đã lưu: {scenario_name}")
                    st.rerun()

            if cancelled:
                st.session_state.editing_scenario = False
                st.session_state.current_scenario = None
                st.rerun()
    if session.scenarios:
        selected_source_id = session.current_source_id
        df_cleaned = st.session_state.cleaned_dfs.get(
            selected_source_id,
            st.session_state.clean_dfs.get(
                selected_source_id,
                st.session_state.raw_dfs.get(selected_source_id)
            )
        )
        
        if df_cleaned is not None and session.schema:
            sql_tool, recreated = setup_sql_tool(session, df_cleaned)
            if recreated:
                st.success("✅ SQL database sẵn sàng cho chat!")
            
            # Add chat interface
            add_chat_validation_to_scenarios_tab(session, df_cleaned, sql_tool)
            
            st.divider()
        else:
            st.divider()
            st.warning("⚠️ Vui lòng hoàn thành Schema Generation và Data Cleaning trước khi test")
    # Display existing scenarios
    st.divider()
    st.subheader("📚 Saved Scenarios")

    if not session.scenarios:
        st.info("No scenarios defined yet.")
    else:
        for i, scenario in enumerate(session.scenarios):
            with st.expander(f"**{scenario.name}**", expanded=False):
                col_info, col_actions = st.columns([3, 1])
                with col_info:
                    st.write(f"**Description:** {scenario.description or 'N/A'}")
                    st.write(f"**Selected Fields:** `{', '.join(str(f) for f in scenario.selected_fields)}`")
                    
                    st.write("**Questions:**")
                    for q in scenario.questions:
                        st.write(f"- {q}")
                    
                    st.write("**Output Format:**")
                    # Hiển thị thông minh
                    fmt = scenario.output_format
                    if not fmt:
                        st.caption("Empty")
                    elif fmt.get("type") == "text_description" or (len(fmt)==1 and "description" in fmt):
                        st.info(fmt.get("description"))
                    else:
                        st.json(fmt)

                with col_actions:
                    if st.button("✏️ Edit", key=f"edit_scen_{i}"):
                        st.session_state.current_scenario = scenario
                        st.session_state.editing_scenario = True
                        st.rerun()
                    if st.button("🗑️ Delete", key=f"del_scen_{i}"):
                        session.scenarios = [s for s in session.scenarios if s.id != scenario.id]
                        st.success("Deleted")
                        st.rerun()

    # Export
    if session.scenarios:
        st.divider()
        scenarios_json = {"scenarios": [s.to_dict() for s in session.scenarios]}
        st.download_button("⬇️ Download Scenarios", data=json.dumps(scenarios_json, indent=2), file_name="scenarios.json", mime="application/json")


# ============================================================================
# Tab 7: Multi-Table Agent Q&A - HỖ TRỢ QUERY NHIỀU BẢNG
# ============================================================================

def tab_agent_qa_multitable():
    """
    Enhanced Agent Q&A với khả năng query nhiều bảng.
    Cho phép người dùng hỏi đáp trên nhiều bảng dữ liệu khác nhau.
    """
    st.header("🤖 Agent Q&A - Multi-Table Query")
    
    if not st.session_state.session:
        st.warning("⚠️ Chưa có session. Vui lòng vào tab 'Ingestion' trước.")
        return
    
    session = st.session_state.session
    
    # =========================================================================
    # Phần chọn bảng
    # =========================================================================
    st.subheader("📊 Chọn các bảng để Query")
    
    available_sources = [s.source_id for s in session.sources]
    
    if not available_sources:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng import dữ liệu trước.")
        return
    
    # Hiển thị thông tin các bảng có sẵn
    st.markdown("**Các nguồn dữ liệu có sẵn:**")
    cols = st.columns(min(len(available_sources), 4))
    for idx, source_id in enumerate(available_sources):
        df = st.session_state.cleaned_dfs.get(
            source_id,
            st.session_state.clean_dfs.get(
                source_id,
                st.session_state.raw_dfs.get(source_id)
            )
        )
        with cols[idx % 4]:
            if df is not None:
                st.info(f"**{source_id}**\n\n{len(df)} rows × {len(df.columns)} cols")
    
    # Multi-select để chọn bảng
    selected_tables = st.multiselect(
        "🔍 Chọn bảng để đưa vào query",
        options=available_sources,
        default=available_sources,
        help="Chọn một hoặc nhiều bảng. Agent có thể query và JOIN chúng."
    )
    
    if not selected_tables:
        st.warning("⚠️ Vui lòng chọn ít nhất một bảng.")
        return
    
    # =========================================================================
    # Thu thập DataFrames cho các bảng đã chọn
    # =========================================================================
    sources_dfs = {}
    schemas_dict = {}
    
    for source_id in selected_tables:
        df = st.session_state.cleaned_dfs.get(
            source_id,
            st.session_state.clean_dfs.get(
                source_id,
                st.session_state.raw_dfs.get(source_id)
            )
        )
        if df is not None:
            sources_dfs[source_id] = df
            if session.schema:
                schemas_dict[source_id] = session.schema
    
    if not sources_dfs:
        st.warning("⚠️ Không có dữ liệu cho các bảng đã chọn.")
        return

    with st.expander(f"📋 Chi tiết các bảng đã chọn ({len(sources_dfs)} bảng)", expanded=False):
        for source_id, df in sources_dfs.items():
            st.markdown(f"**📊 {source_id}**")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Rows", len(df))
                st.metric("Columns", len(df.columns))
            with col2:
                cols_str = [str(c) for c in df.columns[:8]]
                st.caption(f"Columns: {', '.join(cols_str)}{'...' if len(df.columns) > 8 else ''}")
                safe_display_dataframe(df.head(3), width='stretch')
            st.markdown("---")
    
    # =========================================================================
    # Setup Multi-Table SQL Tool
    # =========================================================================
    with st.spinner("Đang thiết lập SQL database cho nhiều bảng..."):
        sql_tool = setup_multi_table_sql_tool(session, sources_dfs, schemas_dict)
    
    tables_info = sql_tool.get_tables_info()
    st.success(f"✅ {tables_info['total_tables']} bảng sẵn sàng để query!")
    
    # =========================================================================
    # Hiển thị lịch sử hội thoại
    # =========================================================================
    st.divider()
    st.subheader("💬 Hội thoại")
    
    if not session.agent_conversations:
        st.caption("👋 Bắt đầu bằng cách đặt câu hỏi bên dưới!")
        st.info("""
        💡 **Gợi ý cho Multi-Table Query:**
        
        - **Một bảng**: "Hiển thị 10 dòng đầu từ [tên_bảng]"
        - **Join bảng**: "Join bảng employees và departments theo department_id"
        - **So sánh**: "So sánh tổng doanh thu giữa bảng 1 và bảng 2"
        - **Tổng hợp**: "Tổng hợp dữ liệu theo category từ tất cả các bảng"
        """)
    else:
        for msg in session.agent_conversations:
            if msg.role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>Bạn:</strong><br>{msg.content}
                </div>
                """, unsafe_allow_html=True)
            else:
                badges = ""
                if msg.context:
                    if msg.context.get('multi_table'):
                        badges += " 📊"
                    if msg.context.get('used_sql'):
                        badges += " 🔍"
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Agent{badges}:</strong><br>{msg.content}
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # Form nhập câu hỏi
    # =========================================================================
    st.divider()
    
    with st.form("multi_chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Câu hỏi của bạn:",
            placeholder="""Ví dụ:
- Hiển thị 5 dòng đầu từ mỗi bảng
- Mỗi bảng có bao nhiêu dòng?
- Các bảng có cột nào chung?
- Join bảng A và bảng B theo cột X
- So sánh tổng giữa các bảng""",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submitted = st.form_submit_button("📤 Gửi", type="primary")
        with col2:
            clear_chat = st.form_submit_button("🗑️ Xóa chat")
    
    if clear_chat:
        session.agent_conversations = []
        st.rerun()
    
    if submitted and user_input:
        with st.spinner("Agent đang phân tích câu hỏi của bạn..."):
            try:
                response = _query_multi_table_agent(user_input, sql_tool, session)
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
    
    # =========================================================================
    # Quick Queries
    # =========================================================================
    st.divider()
    st.subheader("💡 Query nhanh")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Danh sách bảng"):
            st.markdown("**Các bảng có sẵn:**")
            for table in sql_tool.tables.keys():
                info = sql_tool.tables[table]
                st.markdown(f"- **{table}**: {info['row_count']} rows")
    
    with col2:
        if st.button("📈 Số dòng mỗi bảng"):
            queries = [f"SELECT '{t}' as table_name, COUNT(*) as row_count FROM [{t}]" 
                      for t in sql_tool.tables.keys()]
            if queries:
                result_df, error = sql_tool.execute_query(" UNION ALL ".join(queries))
                if error:
                    st.error(error)
                else:
                    st.dataframe(result_df, width='stretch')
    
    with col3:
        if st.button("🔍 Xem trước dữ liệu"):
            for table_name in list(sql_tool.tables.keys())[:3]:
                result_df, error = sql_tool.execute_query(f"SELECT * FROM [{table_name}] LIMIT 5")
                if not error:
                    st.write(f"**{table_name}:**")
                    st.dataframe(result_df, width='stretch')
    
    with col4:
        if st.button("📋 Danh sách cột"):
            for table_name, info in sql_tool.tables.items():
                st.markdown(f"**{table_name}:**")
                st.code(", ".join(str(c) for c in info['columns']))
    
    # =========================================================================
    # Câu hỏi từ Question Collection và Scenarios
    # =========================================================================
    
    # Hiển thị câu hỏi từ Question Collection
    if session.question_set and session.question_set.user_questions:
        st.divider()
        st.subheader("❓ Câu hỏi từ Question Collection")
        st.caption("Click vào câu hỏi để test ngay")
        
        user_questions = [q.question for q in session.question_set.user_questions]
        cols = st.columns(min(3, len(user_questions)))
        
        for i, q in enumerate(user_questions[:9]):  # Max 9 questions
            with cols[i % 3]:
                btn_label = q[:40] + "..." if len(q) > 40 else q
                if st.button(f"💬 {btn_label}", key=f"qc_q_{i}"):
                    with st.spinner("Agent đang xử lý..."):
                        try:
                            response = _query_multi_table_agent(q, sql_tool, session)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi: {str(e)}")
    
    # Hiển thị câu hỏi từ Scenarios
    if session.scenarios:
        st.divider()
        st.subheader("🎯 Câu hỏi từ Scenarios")
        
        # Select scenario
        scenario_names = [s.name for s in session.scenarios]
        selected_scenario_name = st.selectbox(
            "Chọn scenario:",
            options=scenario_names,
            key="qa_scenario_select"
        )
        
        selected_scenario = next(
            (s for s in session.scenarios if s.name == selected_scenario_name),
            None
        )
        
        if selected_scenario:
            st.caption(f"📋 {selected_scenario.description or 'Không có mô tả'}")
            st.caption(f"📊 Các trường: {', '.join(selected_scenario.selected_fields[:5])}...")
            
            # Display scenario questions
            cols = st.columns(min(3, len(selected_scenario.questions)))
            for i, q in enumerate(selected_scenario.questions[:9]):
                with cols[i % 3]:
                    btn_label = q[:40] + "..." if len(q) > 40 else q
                    if st.button(f"🎯 {btn_label}", key=f"sc_q_{selected_scenario.id}_{i}"):
                        with st.spinner("Agent đang xử lý theo scenario..."):
                            try:
                                response = _query_multi_table_agent(q, sql_tool, session)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Lỗi: {str(e)}")
    
    # =========================================================================
    # Direct SQL Query Section
    # =========================================================================
    st.divider()
    st.subheader("🔍 SQL Query trực tiếp")
    
    with st.expander("Thực thi SQL tùy chỉnh (Nâng cao)", expanded=False):
        st.markdown("**📊 Các bảng có sẵn:**")
        for table_name, info in sql_tool.tables.items():
            cols_preview = ', '.join(str(c) for c in info['columns'][:6])
            if len(info['columns']) > 6:
                cols_preview += f" ... (+{len(info['columns']) - 6} cột nữa)"
            st.caption(f"• `{table_name}` ({info['row_count']} rows): {cols_preview}")
        
        st.markdown("---")
        
        default_query = f"SELECT * FROM [{list(sql_tool.tables.keys())[0]}] LIMIT 10" if sql_tool.tables else "SELECT 1"
        
        sql_query = st.text_area(
            "SQL Query",
            value=default_query,
            height=150,
            help="""
            Tips:
            - Dùng [table_name] cho tên bảng có ký tự đặc biệt
            - JOIN: SELECT * FROM [t1] JOIN [t2] ON t1.id = t2.ref_id
            - UNION: SELECT * FROM [t1] UNION SELECT * FROM [t2]
            """
        )
        
        if st.button("▶️ Thực thi SQL", type="primary"):
            result_df, error = sql_tool.execute_query(sql_query)
            
            if error:
                st.error(error)
            else:
                st.success(f"✅ Query trả về {len(result_df)} dòng")
                safe_display_dataframe(result_df, width='stretch')
                
                csv = result_df.to_csv(index=False)
                st.download_button("📥 Tải CSV", csv, "query_result.csv", "text/csv")


def _query_multi_table_agent(question: str, sql_tool: MultiTableSQLQueryTool, session) -> str:
    """Query agent với multi-table context, hỗ trợ Static Scenario (No-SQL)"""
    from openai import OpenAI
    
    client = OpenAI(api_key=st.session_state.api_key)
    
    # =========================================================================
    # Build comprehensive context
    # =========================================================================
    context_parts = []
    
    # 1. Scenarios (ĐƯA LÊN ĐẦU ĐỂ ƯU TIÊN)
    if session.scenarios:
        context_parts.append("\n**🎯 DEFINED SCENARIOS (USE CASES):**")
        for sc in session.scenarios:
            # Xác định loại scenario
            scenario_type = "DATA-DRIVEN (Requires SQL)" if sc.selected_fields else "STATIC/LOGIC (NO SQL NEEDED)"
            
            context_parts.append(f"\n--- Scenario: {sc.name} [{scenario_type}] ---")
            if sc.description:
                context_parts.append(f"Description: {sc.description}")
            
            if sc.selected_fields:
                context_parts.append(f"Relevant Fields: {', '.join(str(f) for f in sc.selected_fields)}")
            else:
                context_parts.append("Relevant Fields: NONE (Do NOT query database for this scenario)")
                
            context_parts.append("Trigger Questions:")
            for q in sc.questions:
                context_parts.append(f"  • {q}")
            
            # Output format
            if sc.output_format:
                if isinstance(sc.output_format, dict):
                    if sc.output_format.get("type") == "text_description" or "description" in sc.output_format:
                        context_parts.append(f"REQUIRED ANSWER: {sc.output_format.get('description', '')}")
                    else:
                        context_parts.append(f"REQUIRED FORMAT (JSON): {json.dumps(sc.output_format, ensure_ascii=False)}")
            
            # Example if available
            if sc.example_output:
                context_parts.append(f"Example Answer: {json.dumps(sc.example_output, ensure_ascii=False)}")
    
    # 2. Tables info
    context_parts.append("\n**📊 DATABASE TABLES:**")
    for table_name, info in sql_tool.tables.items():
        cols_preview = ', '.join(str(c) for c in info['columns'][:10])
        if len(info['columns']) > 10:
            cols_preview += f", ... (+{len(info['columns']) - 10} columns)"
        context_parts.append(f"- **{table_name}** ({info['row_count']} rows): {cols_preview}")
    
    # 3. Schema details & Business Rules (Giữ nguyên như cũ...)
    if session.schema:
        context_parts.append("\n**📋 SCHEMA DETAILS:**")
        for col_name, col_schema in list(session.schema.items())[:20]:
            if hasattr(col_schema, 'semantic_type'):
                desc = col_schema.description[:80] if col_schema.description else ''
                context_parts.append(f"- {col_name}: {col_schema.semantic_type} - {desc}")
    
    if session.question_set and session.question_set.additional_notes:
        context_parts.append(f"\n**⚠️ BUSINESS RULES:**\n{session.question_set.additional_notes}")

    context = "\n".join(context_parts)
    
    # =========================================================================
    # Build System Prompt (Cập nhật logic cấm dùng SQL khi không cần thiết)
    # =========================================================================
    system_prompt = """You are an intelligent Data Analyst Agent.

🚨 **DECISION PROTOCOL (EXECUTE IN ORDER):** 🚨

1. **CHECK SCENARIO MATCH (TOP PRIORITY)**:
   - Does the user's question match a defined Scenario?
   - **IF MATCHED AND SCENARIO HAS NO RELEVANT FIELDS (Static/Logic Type):**
     - 🛑 **STOP! DO NOT USE SQL.**
     - Answer DIRECTLY using the "REQUIRED ANSWER" or "Example Answer" provided in the scenario.
     - Ignore the database completely for this turn.
   
   - **IF MATCHED AND SCENARIO HAS FIELDS (Data-Driven Type):**
     - Use SQL to query the specified fields.
     - Format the output according to the scenario definition.

2. **GENERAL QUERY (If no scenario matches)**:
   - Use SQL to query the database tables to answer the question.
   - Always reply in VIETNAMESE.

**SQL RULES:**
- Only SELECT queries allowed.
- Use [table_name] syntax.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"**Context:**\n{context}\n\n**Question:**\n{question}"}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": "Execute SQL SELECT query. ONLY use this if real data retrieval is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query"}
                },
                "required": ["query"]
            }
        }
    }]
    
    # Gọi LLM
    response = client.chat.completions.create(
        model=st.session_state.get('model', 'gpt-4o-mini'),
        messages=messages,
        tools=tools,
        tool_choice="auto", 
        temperature=0.3, 
        max_tokens=2000
    )
    
    assistant_msg = response.choices[0].message
    
    if assistant_msg.tool_calls:

        sql_results = []
        for tool_call in assistant_msg.tool_calls:
            if tool_call.function.name == "execute_sql_query":
                args = json.loads(tool_call.function.arguments)
                query = args.get("query", "")
                result_df, error = sql_tool.execute_query(query)
                if error: sql_results.append(f"❌ SQL Error: {error}")
                else: sql_results.append(f"Result:\n{result_df.head(20).to_string()}")

        result_summary = "\n".join(sql_results)
        
        # Second call to summarize
        final_response = client.chat.completions.create(
            model=st.session_state.get('model', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": "Analyze SQL results and answer in Vietnamese."},
                {"role": "user", "content": f"Question: {question}\nSQL Result: {result_summary}"}
            ]
        )
        final_text = final_response.choices[0].message.content
    else:
        # Trường hợp Static Scenario (LLM quyết định không dùng tool)
        final_text = assistant_msg.content or ""
    
    # Logging chat history
    session.agent_conversations.append(AgentMessage(
        role="user", content=question, timestamp=datetime.now().isoformat()
    ))
    session.agent_conversations.append(AgentMessage(
        role="assistant", content=final_text, timestamp=datetime.now().isoformat(),
        context={"used_sql": bool(assistant_msg.tool_calls), "multi_table": True}
    ))
    
    return final_text

def main():
    initialize_session_state()
    tabs = st.tabs([
        "📁 Ingestion",
        "🔍 Structure Analysis",
        "🧹 Type Cleaning",
        "📋 Schema Generation",
        "❓ Question Collection",
        "🎯 Scenarios",
        "🤖 Agent Q&A (Multi-Table)"
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
        tab_question_collection()

    with tabs[5]:
        tab_scenario_definition()

    with tabs[6]:
        tab_agent_qa_multitable()

if __name__ == "__main__":
    main()