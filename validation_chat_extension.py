"""
Validation Extension with Interactive Chat Interface
Cho phép người dùng chat với agent ngay trong tab Questions và Scenarios
HỖ TRỢ QUERY NHIỀU BẢNG (Multi-Table) VÀ STREAMING
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime
from pathlib import Path
import sqlite3
import re
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# SQL Query Tool (Single Table - GIỮ NGUYÊN ĐỂ BACKWARD COMPATIBLE)
# ============================================================================

class SQLQueryTool:
    """Tool to execute SQL queries - thread-safe for Streamlit"""

    def __init__(self, db_path: str, df: pd.DataFrame, table_name: str = "data"):
        self.db_path = db_path
        self.table_name = table_name
        self.df_hash = hash(str(df.values.tobytes()))
        self._create_database(df)

    def _create_database(self, df: pd.DataFrame):
        """Create/recreate database with data"""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            df.to_sql(self.table_name, self.conn, if_exists='replace', index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to create database: {str(e)}")

    def _ensure_connection(self):
        """Ensure connection is valid, recreate if needed"""
        try:
            if not self.conn:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("SELECT 1")
        except:
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
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Lấy thông tin schema cho tất cả các bảng (tương thích với SQLQueryTool)"""
        return self.get_tables_info()
    
    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None


# ============================================================================
# Setup Helper Functions
# ============================================================================

def setup_sql_tool(session, df_cleaned):
    """Setup or update SQL tool for chat (Single table - backward compatible)"""
    current_df_hash = hash(str(df_cleaned.values.tobytes()))
    
    need_recreate = False
    if 'sql_tool' not in st.session_state or st.session_state.sql_tool is None:
        need_recreate = True
    elif 'sql_tool_df_hash' not in st.session_state:
        need_recreate = True
    elif st.session_state.sql_tool_df_hash != current_df_hash:
        need_recreate = True
    
    if need_recreate:
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


def setup_multi_table_sql_tool(session, sources_dfs: Dict[str, pd.DataFrame], 
                                schemas: Dict[str, Dict] = None) -> MultiTableSQLQueryTool:
    """
    Setup hoặc cập nhật multi-table SQL tool.
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


def get_all_available_dataframes(session) -> Dict[str, pd.DataFrame]:
    """Lấy tất cả các DataFrame có sẵn từ session state."""
    sources_dfs = {}
    
    for source in session.sources:
        source_id = source.source_id
        df = st.session_state.cleaned_dfs.get(
            source_id,
            st.session_state.clean_dfs.get(
                source_id,
                st.session_state.raw_dfs.get(source_id)
            )
        )
        if df is not None:
            sources_dfs[source_id] = df
    
    return sources_dfs


# ============================================================================
# Chat State Management
# ============================================================================

def initialize_chat_state(context_key: str):
    chat_key = f"chat_history_{context_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []


def get_chat_history(context_key: str) -> List[Dict[str, str]]:
    chat_key = f"chat_history_{context_key}"
    return st.session_state.get(chat_key, [])


def add_to_chat_history(context_key: str, role: str, content: str):
    chat_key = f"chat_history_{context_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    st.session_state[chat_key].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })


def clear_chat_history(context_key: str):
    chat_key = f"chat_history_{context_key}"
    st.session_state[chat_key] = []


# ============================================================================
# Chat Interface Components
# ============================================================================

def render_chat_message(role: str, content: str, timestamp: str = None):
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>Bạn:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Agent:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)


def stream_agent_response(agent, question: str, placeholder) -> Generator[str, None, None]:
    """
    Stream response từ agent (Single Table) dưới dạng Generator.
    Giúp UI hiển thị mượt mà từng từ.
    """
    full_response = ""
    try:
        for chunk in agent.query(question):
            if isinstance(chunk, dict):
                continue
            full_response += chunk
            placeholder.markdown(full_response + "▌")
            yield chunk
        
        # Kết thúc stream, hiển thị bản clean (bỏ con trỏ)
        placeholder.markdown(full_response)
        
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        placeholder.markdown(error_msg)
        yield error_msg


# ============================================================================
# Multi-Table Query with Streaming Support
# ============================================================================

def _build_multi_table_context(sql_tool: MultiTableSQLQueryTool, session) -> str:
    """Build context string for multi-table query"""
    context_parts = []
    
    context_parts.append("**📊 CÁC BẢNG DỮ LIỆU:**")
    for table_name, info in sql_tool.tables.items():
        cols_preview = ', '.join(str(c) for c in info['columns'][:10])
        if len(info['columns']) > 10:
            cols_preview += f", ... (+{len(info['columns']) - 10} cột)"
        context_parts.append(f"- **{table_name}** ({info['row_count']} rows): {cols_preview}")
    
    if session.schema:
        context_parts.append("\n**📋 CHI TIẾT SCHEMA:**")
        for col_name, col_schema in list(session.schema.items()):
            if hasattr(col_schema, 'semantic_type') and hasattr(col_schema, 'description'):
                desc = col_schema.description[:60] if col_schema.description else ''
                context_parts.append(f"- {col_name}: {col_schema.semantic_type} - {desc}")
    
    if session.question_set and session.question_set.additional_notes:
        context_parts.append(f"\n**⚠️ QUY TẮC NGHIỆP VỤ:**\n{session.question_set.additional_notes[:500]}")
    
    if session.scenarios:
        context_parts.append("\n**🎯 SCENARIOS:**")
        for sc in session.scenarios:
            context_parts.append(f"- {sc.name}: {sc.description if sc.description else 'N/A'}")
    
    return "\n".join(context_parts)


def _stream_multi_table_query(question: str, sql_tool: MultiTableSQLQueryTool, 
                               session, placeholder) -> Generator[str, None, None]:
    """
    Query agent với multi-table context VỚI STREAMING.
    Hàm này yield từng chunk text để UI cập nhật.
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=st.session_state.api_key)
    context = _build_multi_table_context(sql_tool, session)
    
    system_prompt = """Bạn là một data analyst thông minh với khả năng query SQL trên NHIỀU BẢNG.

**NGUYÊN TẮC:**
1. Dùng [table_name] cho tên bảng
2. Dùng table.column khi cần phân biệt
3. Có thể JOIN, UNION nhiều bảng
4. LUÔN trả lời bằng TIẾNG VIỆT
5. Nếu có business rules → tuân theo

Khi cần dữ liệu, dùng tool execute_sql_query."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"**Context:**\n{context}\n\n**Câu hỏi:**\n{question}"}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": f"Thực thi SQL SELECT query trên các bảng: {', '.join(sql_tool.tables.keys())}. Dùng [table_name] cho tên bảng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query"}
                },
                "required": ["query"]
            }
        }
    }]
    
    full_response = ""
    
    try:
        # Bước 1: Agent suy nghĩ và gọi Tool (không stream phần này, chỉ hiện status)
        placeholder.markdown("🔍 Agent đang phân tích...")
        
        response = client.chat.completions.create(
            model=st.session_state.get('model', 'gpt-4o-mini'),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2000
        )
        
        assistant_msg = response.choices[0].message
        
        # Bước 2: Xử lý Tool Call (SQL)
        if assistant_msg.tool_calls:
            sql_results = []
            
            for tool_call in assistant_msg.tool_calls:
                if tool_call.function.name == "execute_sql_query":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", "")
                    
                    placeholder.markdown(f"🔍 Đang thực thi SQL:\n```sql\n{query}\n```")
                    
                    result_df, error = sql_tool.execute_query(query)
                    
                    if error:
                        sql_results.append(f"❌ SQL Error: {error}")
                    else:
                        if len(result_df) == 0:
                            sql_results.append("Query không trả về kết quả")
                        else:
                            # Format kết quả gọn gàng
                            result_text = f"Kết quả ({len(result_df)} dòng):\n```\n{result_df.head(15).to_string(index=False)}\n```"
                            if len(result_df) > 15:
                                result_text += f"\n(Hiển thị 15/{len(result_df)} dòng)"
                            sql_results.append(result_text)
            
            result_summary = "\n\n".join(sql_results)
            
            # Bước 3: Agent phân tích kết quả cuối cùng (STREAMING)
            placeholder.markdown("📝 Đang phân tích kết quả...")
            
            final_stream = client.chat.completions.create(
                model=st.session_state.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": "Phân tích kết quả SQL và trả lời bằng tiếng Việt. Trả lời ngắn gọn và rõ ràng. Format markdown đẹp."},
                    {"role": "user", "content": f"Câu hỏi: {question}\n\nKết quả SQL:\n{result_summary}\n\nPhân tích và trả lời:"}
                ],
                temperature=0.7,
                max_tokens=1500,
                stream=True
            )
            
            for chunk in final_stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    # Cập nhật UI ngay lập tức
                    placeholder.markdown(full_response + "▌")
                    # Yield text để caller có thể xử lý nếu cần
                    yield text

            placeholder.markdown(full_response)
            
        else:
            # Trường hợp không gọi tool, trả lời trực tiếp
            if assistant_msg.content:
                full_response = assistant_msg.content
                placeholder.markdown(full_response)
                yield full_response
            else:
                full_response = "Không có câu trả lời từ Agent."
                placeholder.markdown(full_response)
                yield full_response
            
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        placeholder.markdown(error_msg)
        yield error_msg

def _query_multi_table_for_chat(question: str, sql_tool: MultiTableSQLQueryTool, session) -> str:
    """Query agent với multi-table context (KHÔNG STREAMING - backward compatible)"""
    from openai import OpenAI
    
    client = OpenAI(api_key=st.session_state.api_key)
    context = _build_multi_table_context(sql_tool, session)
    
    system_prompt = """Bạn là một data analyst thông minh với khả năng query SQL trên NHIỀU BẢNG.

**NGUYÊN TẮC:**
1. Dùng [table_name] cho tên bảng
2. Dùng table.column khi cần phân biệt
3. Có thể JOIN, UNION nhiều bảng
4. LUÔN trả lời bằng TIẾNG VIỆT
5. Nếu có business rules → tuân theo

Khi cần dữ liệu, dùng tool execute_sql_query."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"**Context:**\n{context}\n\n**Câu hỏi:**\n{question}"}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_sql_query",
            "description": f"Thực thi SQL SELECT query trên các bảng: {', '.join(sql_tool.tables.keys())}. Dùng [table_name] cho tên bảng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query"}
                },
                "required": ["query"]
            }
        }
    }]
    
    try:
        response = client.chat.completions.create(
            model=st.session_state.get('model', 'gpt-4o-mini'),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
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
                    
                    if error:
                        sql_results.append(f"❌ SQL Error: {error}")
                    else:
                        if len(result_df) == 0:
                            sql_results.append("Query không trả về kết quả")
                        else:
                            result_text = f"Kết quả ({len(result_df)} dòng):\n```\n{result_df.head(15).to_string(index=False)}\n```"
                            if len(result_df) > 15:
                                result_text += f"\n(Hiển thị 15/{len(result_df)} dòng)"
                            sql_results.append(result_text)
            
            result_summary = "\n\n".join(sql_results)
            
            final_response = client.chat.completions.create(
                model=st.session_state.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": "Phân tích kết quả SQL và trả lời bằng tiếng Việt."},
                    {"role": "user", "content": f"Câu hỏi: {question}\n\nKết quả SQL:\n{result_summary}\n\nPhân tích và trả lời:"}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return final_response.choices[0].message.content
        else:
            return assistant_msg.content or "Không có câu trả lời"
            
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"



def render_chat_interface(
    session,
    agent,
    context_key: str,
    placeholder_text: str = "Hỏi một câu để test...",
    quick_questions: List[str] = None
):
    """Render chat interface for validation with STREAMING support (Generator Safe)"""
    initialize_chat_state(context_key)
    
    chat_history = get_chat_history(context_key)
    
    # Hiển thị lịch sử
    if not chat_history:
        st.info("💬 Bắt đầu chat để test câu hỏi của bạn! Agent sẽ trả lời dựa trên dữ liệu thực.")
    else:
        chat_container = st.container()
        with chat_container:
            for msg in chat_history:
                # Safety check khi render
                content = msg["content"]
                if not isinstance(content, str):
                    content = str(content)
                render_chat_message(msg["role"], content, msg.get("timestamp"))
    
    # Xử lý Quick Questions
    if quick_questions:
        st.markdown("**💡 Câu hỏi gợi ý:**")
        cols = st.columns(min(len(quick_questions), 3))
        for i, q in enumerate(quick_questions[:6]):
            with cols[i % 3]:
                btn_label = q[:30] + "..." if len(q) > 30 else q
                if st.button(f"💬 {btn_label}", key=f"{context_key}_quick_{i}"):
                    # 1. Add User Question
                    add_to_chat_history(context_key, "user", q)
                    render_chat_message("user", q)
                    # 2. Stream & Accumulate
                    response_placeholder = st.empty()
                    gen = stream_agent_response(agent, q, response_placeholder)
                    
                    full_response = ""
                    logger.info(f"GENERATOR 1: {gen}")
                    for chunk in gen:
                        full_response += chunk
                        logger.info(f"CHUNK 1: {chunk}")
                    # 3. Save String Only
                    add_to_chat_history(context_key, "assistant", full_response)
                    logger.info(f"FULL RESPONSE 1: {full_response}")

                    st.rerun()
    
    st.markdown("---")
    with st.form(f"chat_form_{context_key}", clear_on_submit=True):
        user_input = st.text_input(
            "Nhập câu hỏi của bạn:",
            placeholder=placeholder_text,
            key=f"chat_input_{context_key}"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submitted = st.form_submit_button("📤 Gửi", type="primary")
        with col2:
            clear = st.form_submit_button("🗑️ Xóa chat")
    
    if clear:
        clear_chat_history(context_key)
        st.rerun()
    
    if submitted and user_input:
        add_to_chat_history(context_key, "user", user_input)
        render_chat_message("user", user_input)
        response_placeholder = st.empty()
        
        # FIX: Loop generator to get full string
        gen = stream_agent_response(agent, user_input, response_placeholder)
        full_response = ""
        logger.info(f"GENERATOR 2: {gen}")

        for chunk in gen:
            full_response += chunk
            logger.info(f"CHUNK 2: {chunk}")
        add_to_chat_history(context_key, "assistant", full_response)
        logger.info(f"FULL RESPONSE 3: {full_response}")
        st.rerun()


def render_multi_table_chat_interface(
    session,
    sql_tool,
    context_key: str,
    placeholder_text: str = "Hỏi về dữ liệu trong các bảng...",
    quick_questions: List[str] = None
):
    """Render chat interface với hỗ trợ nhiều bảng VÀ STREAMING (Generator Safe)"""
    initialize_chat_state(context_key)
    
    tables_info = sql_tool.get_tables_info()
    with st.expander(f"📊 Các bảng có sẵn ({tables_info['total_tables']} bảng)", expanded=False):
        for table in tables_info["tables"]:
            cols_preview = ', '.join(str(c) for c in table['columns'][:6])
            if len(table['columns']) > 6:
                cols_preview += f" ... (+{len(table['columns']) - 6} cột)"
            st.markdown(f"• **{table['name']}** ({table['row_count']} rows): {cols_preview}")
    
    chat_history = get_chat_history(context_key)
    
    if not chat_history:
        st.info("💬 Bắt đầu chat để test câu hỏi! Agent có thể query trên nhiều bảng.")
    else:
        chat_container = st.container()
        with chat_container:
            for msg in chat_history:
                # Safety check
                content = msg["content"]
                if not isinstance(content, str):
                    content = str(content)
                render_chat_message(msg["role"], content, msg.get("timestamp"))
    
    # Xử lý Quick Questions
    if quick_questions:
        st.markdown("**💡 Câu hỏi gợi ý:**")
        cols = st.columns(min(len(quick_questions), 3))
        for i, q in enumerate(quick_questions[:6]):
            with cols[i % 3]:
                btn_label = q[:30] + "..." if len(q) > 30 else q
                if st.button(f"💬 {btn_label}", key=f"{context_key}_quick_{i}"):
                    # 1. Add User Question
                    add_to_chat_history(context_key, "user", q)
                    render_chat_message("user", q)
                    
                    # 2. Stream & Accumulate
                    response_placeholder = st.empty()
                    gen = _stream_multi_table_query(q, sql_tool, session, response_placeholder)
                    
                    full_response = ""
                    for chunk in gen:
                        full_response += chunk
                    
                    # 3. Save String Only
                    add_to_chat_history(context_key, "assistant", full_response)
                    logger.info(f"FULL RESPONSE 4: {full_response}")
                    st.rerun()
    
    st.markdown("---")
    with st.form(f"chat_form_{context_key}", clear_on_submit=True):
        user_input = st.text_input(
            "Nhập câu hỏi của bạn:",
            placeholder=placeholder_text,
            key=f"chat_input_{context_key}"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submitted = st.form_submit_button("📤 Gửi", type="primary")
        with col2:
            clear = st.form_submit_button("🗑️ Xóa chat")
    
    if clear:
        clear_chat_history(context_key)
        st.rerun()
    
    if submitted and user_input:
        add_to_chat_history(context_key, "user", user_input)
        render_chat_message("user", user_input)
        
        response_placeholder = st.empty()
        
        # FIX: Loop generator to get full string
        gen = _stream_multi_table_query(user_input, sql_tool, session, response_placeholder)
        full_response = ""
        for chunk in gen:
            full_response += chunk
            
        add_to_chat_history(context_key, "assistant", full_response)
        logger.info(f"FULL RESPONSE 5: {full_response}")
        st.rerun()

# ============================================================================
# Add Chat Validation Functions
# ============================================================================

def add_chat_validation_to_questions_tab(session, df_cleaned, sql_tool, 
                                          use_multi_table: bool = False,
                                          sources_dfs: Dict[str, pd.DataFrame] = None):
    """Add interactive chat validation to Questions tab"""
    st.divider()
    st.subheader("🔍 Test Câu hỏi với Agent")
    
    if not session.question_set or not session.question_set.user_questions:
        st.info("📝 Tạo câu hỏi ở phần trên, sau đó quay lại đây để test!")
        return
    
    user_questions = [q.question for q in session.question_set.user_questions]
    
    # --- MULTI TABLE MODE ---
    if use_multi_table and sources_dfs and len(sources_dfs) > 1:
        st.info(f"📊 **Chế độ Multi-Table**: Query trên {len(sources_dfs)} bảng")
        
        multi_sql_tool = setup_multi_table_sql_tool(session, sources_dfs, 
                                                     {sid: session.schema for sid in sources_dfs})
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"💡 Bạn đã tạo **{len(user_questions)}** câu hỏi. Test với **{len(sources_dfs)} bảng**!")
        with col2:
            if st.button("📋 Copy câu hỏi", key="copy_questions_mt"):
                questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(user_questions)])
                st.code(questions_text)
        
        render_multi_table_chat_interface(
            session=session,
            sql_tool=multi_sql_tool,
            context_key="questions_validation_multi",
            placeholder_text="Ví dụ: So sánh dữ liệu giữa các bảng, JOIN bảng A và B...",
            quick_questions=user_questions
        )
        
    # --- SINGLE TABLE MODE ---
    else:
        if 'validation_agent' not in st.session_state or st.session_state.validation_agent is None:
            from hst_agent import DataSchemaAgent
            agent = DataSchemaAgent(
                session,
                st.session_state.api_key,
                st.session_state.model,
                df_cleaned=df_cleaned,
                sql_tool=sql_tool
            )
            st.session_state.validation_agent = agent
        else:
            agent = st.session_state.validation_agent
            if sql_tool and agent.sql_tool != sql_tool:
                agent.sql_tool = sql_tool
                agent.db_path = sql_tool.db_path
                agent.df_cleaned = df_cleaned
        
        agent = st.session_state.validation_agent
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"💡 Bạn đã tạo **{len(user_questions)}** câu hỏi. Test với agent!")
        with col2:
            if st.button("📋 Copy câu hỏi", key="copy_questions"):
                questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(user_questions)])
                st.code(questions_text)
        
        render_chat_interface(
            session=session,
            agent=agent,
            context_key="questions_validation",
            placeholder_text="Ví dụ: What is the average price?",
            quick_questions=user_questions
        )


def add_chat_validation_to_scenarios_tab(session, df_cleaned, sql_tool,
                                          use_multi_table: bool = False,
                                          sources_dfs: Dict[str, pd.DataFrame] = None):
    """Add interactive chat validation to Scenarios tab VỚI STREAMING"""
    st.divider()
    st.subheader("🔍 Test Scenario với Agent")
    
    if not session.scenarios:
        st.info("📝 Tạo scenario ở phần trên, sau đó quay lại đây để test!")
        return
    
    scenario_names = [s.name for s in session.scenarios]
    selected_scenario_name = st.selectbox(
        "Chọn scenario để test:",
        options=scenario_names,
        key="test_scenario_select"
    )
    
    selected_scenario = next(
        (s for s in session.scenarios if s.name == selected_scenario_name),
        None
    )
    
    if not selected_scenario:
        return
    
    with st.expander("ℹ️ Thông tin Scenario", expanded=False):
        st.write(f"**Tên:** {selected_scenario.name}")
        st.write(f"**Mô tả:** {selected_scenario.description or 'N/A'}")
        st.write(f"**Selected Fields:** `{', '.join(selected_scenario.selected_fields)}`")
        st.write(f"**Số câu hỏi:** {len(selected_scenario.questions)}")
    
    is_multi = use_multi_table and sources_dfs and len(sources_dfs) > 1
    context_key = f"scenario_{selected_scenario.id}_multi" if is_multi else f"scenario_{selected_scenario.id}"
    
    # --- MULTI TABLE SCENARIO ---
    if is_multi:
        st.info(f"📊 **Chế độ Multi-Table**: Query trên {len(sources_dfs)} bảng")
        
        multi_sql_tool = setup_multi_table_sql_tool(session, sources_dfs,
                                                     {sid: session.schema for sid in sources_dfs})
        
        render_multi_table_chat_interface(
            session=session,
            sql_tool=multi_sql_tool,
            context_key=context_key,
            placeholder_text="Hỏi về scenario hoặc query trên nhiều bảng...",
            quick_questions=selected_scenario.questions
        )
        
        # === FIX: TEST ALL QUESTIONS (Multi-Table) ===
        st.divider()
        st.markdown("### 🧪 Test Tự Động (Streaming)")
        
        if st.button("▶️ Test All Questions", type="primary", key="auto_test_scenario_multi"):
            clear_chat_history(context_key)
            
            for i, question in enumerate(selected_scenario.questions):
                st.markdown(f"---\n**[Q{i+1}] {question}**")
                add_to_chat_history(context_key, "user", f"[Q{i+1}] {question}")
                
                response_placeholder = st.empty()
                
                # Gọi generator
                gen = _stream_multi_table_query(question, multi_sql_tool, session, response_placeholder)
                
                # Gom toàn bộ text từ generator
                full_response = ""
                for chunk in gen:
                    full_response += chunk
                
                # Lưu text đầy đủ vào history (KHÔNG lưu generator object)
                add_to_chat_history(context_key, "assistant", full_response)
            
            st.success(f"✅ Đã test {len(selected_scenario.questions)} câu hỏi!")
            # Không rerun ở đây để người dùng thấy kết quả streaming cuối cùng
            # Nếu rerun, history sẽ hiển thị đúng text đầy đủ

    # --- SINGLE TABLE SCENARIO ---
    else:
        if 'validation_agent' not in st.session_state or st.session_state.validation_agent is None:
            from hst_agent import DataSchemaAgent
            agent = DataSchemaAgent(
                session,
                st.session_state.api_key,
                st.session_state.model,
                df_cleaned=df_cleaned,
                sql_tool=sql_tool
            )
            st.session_state.validation_agent = agent
        else:
            agent = st.session_state.validation_agent
            if sql_tool and agent.sql_tool != sql_tool:
                agent.sql_tool = sql_tool
                agent.db_path = sql_tool.db_path
                agent.df_cleaned = df_cleaned
        
        agent = st.session_state.validation_agent
        
        render_chat_interface(
            session=session,
            agent=agent,
            context_key=context_key,
            placeholder_text="Hỏi một câu từ scenario hoặc câu hỏi mới...",
            quick_questions=selected_scenario.questions
        )
        
        # === FIX: TEST ALL QUESTIONS (Single Table) ===
        st.divider()
        st.markdown("### 🧪 Test Tự Động (Streaming)")
        
        if st.button("▶️ Test All Questions", type="primary", key="auto_test_scenario"):
            clear_chat_history(context_key)
            
            for i, question in enumerate(selected_scenario.questions):
                st.markdown(f"---\n**[Q{i+1}] {question}**")
                add_to_chat_history(context_key, "user", f"[Q{i+1}] {question}")
                
                response_placeholder = st.empty()
                
                # Gọi generator
                gen = stream_agent_response(agent, question, response_placeholder)
                
                # Gom toàn bộ text
                full_response = ""
                for chunk in gen:
                    full_response += chunk
                
                # Lưu text đầy đủ vào history
                add_to_chat_history(context_key, "assistant", full_response)
            
            st.success(f"✅ Đã test {len(selected_scenario.questions)} câu hỏi!")

# ============================================================================
# Export chat history
# ============================================================================

# ============================================================================
# FIXED: Export chat history (Safety Version)
# ============================================================================

def export_chat_history_to_json(context_key: str, filename: str = None):
    """Export chat history to JSON file with safety checks for generators"""
    chat_history = get_chat_history(context_key)
    
    if not chat_history:
        return None
    
    if filename is None:
        filename = f"chat_history_{context_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Sanitize history: Convert generators or non-serializable objects to string
    safe_history = []
    for msg in chat_history:
        safe_msg = msg.copy()
        content = safe_msg.get("content")
        
        # Kiểm tra nếu content không phải string (ví dụ là generator), convert sang string
        if not isinstance(content, str):
            if content is None:
                safe_msg["content"] = ""
            else:
                # Nếu là generator, ta không cứu được nội dung đã mất, nhưng tránh được crash
                safe_msg["content"] = str(content) 
        
        safe_history.append(safe_msg)
    
    export_data = {
        "context": context_key,
        "exported_at": datetime.now().isoformat(),
        "message_count": len(safe_history),
        "messages": safe_history
    }
    
    # Dùng default=str để force convert mọi thứ còn sót lại thành string
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
    return json_str

def render_export_chat_button(context_key: str):
    """Render button to export chat history"""
    chat_history = get_chat_history(context_key)
    
    if chat_history:
        json_str = export_chat_history_to_json(context_key)
        if json_str:
            st.download_button(
                label="⬇️ Export Chat History",
                data=json_str,
                file_name=f"chat_{context_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key=f"export_chat_{context_key}"
            )


# ============================================================================
# Statistics and Analysis
# ============================================================================

def get_chat_statistics(context_key: str) -> Dict[str, Any]:
    """Get statistics about chat history (Safe version)"""
    chat_history = get_chat_history(context_key)
    
    if not chat_history:
        return {
            "total_messages": 0,
            "user_messages": 0,
            "agent_messages": 0,
            "avg_response_length": 0
        }
    
    user_msgs = [msg for msg in chat_history if msg["role"] == "user"]
    agent_msgs = [msg for msg in chat_history if msg["role"] == "assistant"]
    
    # Tính tổng độ dài an toàn (kiểm tra xem content có phải string không)
    total_length = 0
    for msg in agent_msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_length += len(content)
        # Nếu là generator hoặc object khác, ta bỏ qua hoặc tính là 0 để không gây lỗi
    
    avg_length = total_length / len(agent_msgs) if agent_msgs else 0
    
    return {
        "total_messages": len(chat_history),
        "user_messages": len(user_msgs),
        "agent_messages": len(agent_msgs),
        "avg_response_length": avg_length
    }

def render_chat_statistics(context_key: str):
    """Render chat statistics"""
    stats = get_chat_statistics(context_key)
    
    if stats["total_messages"] > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng tin nhắn", stats["total_messages"])
        with col2:
            st.metric("Câu hỏi", stats["user_messages"])
        with col3:
            st.metric("Câu trả lời", stats["agent_messages"])
        with col4:
            st.metric("Độ dài TB", f"{stats['avg_response_length']:.0f} chars")