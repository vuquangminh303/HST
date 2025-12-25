"""
Validation Extension with Interactive Chat Interface
Cho phép người dùng chat với agent ngay trong tab Questions và Scenarios
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sqlite3


# ============================================================================
# SQL Query Tool (Helper)
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
# Setup Helper Functions
# ============================================================================

def setup_sql_tool(session, df_cleaned):
    """Setup or update SQL tool for chat"""
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


def initialize_chat_state(context_key: str):
    """Initialize chat state for a specific context (questions/scenarios)"""
    chat_key = f"chat_history_{context_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []


def get_chat_history(context_key: str) -> List[Dict[str, str]]:
    """Get chat history for a specific context"""
    chat_key = f"chat_history_{context_key}"
    return st.session_state.get(chat_key, [])


def add_to_chat_history(context_key: str, role: str, content: str):
    """Add message to chat history"""
    chat_key = f"chat_history_{context_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    st.session_state[chat_key].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })


def clear_chat_history(context_key: str):
    """Clear chat history for a specific context"""
    chat_key = f"chat_history_{context_key}"
    st.session_state[chat_key] = []


# ============================================================================
# Chat Interface Components
# ============================================================================

def render_chat_message(role: str, content: str, timestamp: str = None):
    """Render a single chat message"""
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


def stream_agent_response(agent, question: str, placeholder):
    """
    Stream response từ agent với st.write_stream style
    
    Args:
        agent: DataSchemaAgent instance
        question: User question
        placeholder: Streamlit placeholder để hiển thị streaming
    
    Returns:
        Full response text
    """
    full_response = ""
    try:
        for chunk in agent.query(question):
            if isinstance(chunk, dict):
                continue
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        
        # Hiển thị kết quả cuối cùng không có cursor
        placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        placeholder.markdown(error_msg)
        return error_msg


def render_chat_interface(
    session,
    agent,
    context_key: str,
    placeholder_text: str = "Hỏi một câu để test...",
    quick_questions: List[str] = None
):
    """
    Render chat interface for validation with STREAMING support
    
    Args:
        session: Session object
        agent: DataSchemaAgent instance
        context_key: Unique key for this chat context (e.g., "questions_tab", "scenario_1")
        placeholder_text: Placeholder for input
        quick_questions: List of suggested questions
    """
    # Initialize chat state
    initialize_chat_state(context_key)
    
    # Display chat history
    chat_history = get_chat_history(context_key)
    
    if not chat_history:
        st.info("💬 Bắt đầu chat để test câu hỏi của bạn! Agent sẽ trả lời dựa trên dữ liệu thực.")
    else:
        st.markdown("### 💬 Lịch sử Chat")
        for msg in chat_history:
            render_chat_message(msg["role"], msg["content"], msg.get("timestamp"))
    
    st.divider()
    
    # Chat input - KHÔNG dùng form để có thể streaming
    user_input = st.text_area(
        "Câu hỏi của bạn:",
        placeholder=placeholder_text,
        height=100,
        key=f"chat_input_{context_key}"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submitted = st.button("📤 Gửi", type="primary", key=f"btn_send_{context_key}")
    with col2:
        clear = st.button("🗑️ Xóa lịch sử", key=f"btn_clear_{context_key}")
    
    if clear:
        clear_chat_history(context_key)
        st.rerun()
    
    if submitted and user_input.strip():
        # Add user message to history
        add_to_chat_history(context_key, "user", user_input.strip())
        
        # Hiển thị user message
        render_chat_message("user", user_input.strip())
        
        # Streaming response
        st.markdown("**Agent:**")
        response_placeholder = st.empty()
        
        response = stream_agent_response(agent, user_input.strip(), response_placeholder)
        add_to_chat_history(context_key, "assistant", response)
        
        # Rerun để clear input
        st.rerun()
    
    # Quick questions (if provided)
    if quick_questions:
        st.divider()
        st.markdown("### 💡 Câu hỏi gợi ý")
        st.caption("Click vào câu hỏi để thử ngay")
        
        # Lưu câu hỏi được chọn vào session state
        quick_q_key = f"selected_quick_q_{context_key}"
        
        cols = st.columns(min(3, len(quick_questions)))
        for i, q in enumerate(quick_questions[:6]):  # Max 6 quick questions
            with cols[i % 3]:
                btn_label = q[:35] + "..." if len(q) > 35 else q
                if st.button(f"💬 {btn_label}", key=f"quick_{context_key}_{i}"):
                    st.session_state[quick_q_key] = q
        
        # Xử lý câu hỏi được chọn
        if quick_q_key in st.session_state and st.session_state[quick_q_key]:
            selected_q = st.session_state[quick_q_key]
            st.session_state[quick_q_key] = None  # Clear
            
            # Add to history
            add_to_chat_history(context_key, "user", selected_q)
            
            # Hiển thị
            render_chat_message("user", selected_q)
            
            # Streaming response
            st.markdown("**Agent:**")
            response_placeholder = st.empty()
            
            response = stream_agent_response(agent, selected_q, response_placeholder)
            add_to_chat_history(context_key, "assistant", response)
            
            st.rerun()


# ============================================================================
# Tab Extension Functions
# ============================================================================

def add_chat_validation_to_questions_tab(session, df_cleaned, sql_tool):
    """
    Add interactive chat validation to Questions tab
    
    Args:
        session: Session object
        df_cleaned: Cleaned DataFrame
        sql_tool: SQLQueryTool instance
    """
    st.divider()
    st.subheader("🔍 Test Questions với Agent")
    
    if not session.question_set or not session.question_set.user_questions:
        st.info("📝 Thêm câu hỏi ở phần trên, sau đó quay lại đây để test!")
        return
    
    # Initialize agent with df_cleaned and sql_tool
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
        # Update existing agent with current sql_tool if needed
        agent = st.session_state.validation_agent
        if sql_tool and agent.sql_tool != sql_tool:
            agent.sql_tool = sql_tool
            agent.db_path = sql_tool.db_path
            agent.df_cleaned = df_cleaned
    
    agent = st.session_state.validation_agent
    
    user_questions = [q.question for q in session.question_set.user_questions]
    
    # Show info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"💡 Bạn đã tạo **{len(user_questions)}** câu hỏi. Test chúng bằng cách chat với agent bên dưới!")
    with col2:
        if st.button("📋 Copy câu hỏi", key="copy_questions"):
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(user_questions)])
            st.code(questions_text)
    
    # Render chat interface
    render_chat_interface(
        session=session,
        agent=agent,
        context_key="questions_validation",
        placeholder_text="Ví dụ: What is the average price?\nHoặc chọn từ danh sách câu hỏi bạn đã tạo...",
        quick_questions=user_questions
    )


def add_chat_validation_to_scenarios_tab(session, df_cleaned, sql_tool):
    """
    Add interactive chat validation to Scenarios tab
    
    Args:
        session: Session object
        df_cleaned: Cleaned DataFrame
        sql_tool: SQLQueryTool instance
    """
    st.divider()
    st.subheader("🔍 Test Scenario với Agent")
    
    if not session.scenarios:
        st.info("📝 Tạo scenario ở phần trên, sau đó quay lại đây để test!")
        return
    
    # Initialize agent with df_cleaned and sql_tool
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
        # Update existing agent with current sql_tool if needed
        agent = st.session_state.validation_agent
        if sql_tool and agent.sql_tool != sql_tool:
            agent.sql_tool = sql_tool
            agent.db_path = sql_tool.db_path
            agent.df_cleaned = df_cleaned
    
    agent = st.session_state.validation_agent
    
    # Select scenario to test
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
    
    if selected_scenario:
        # Show scenario info
        with st.expander("ℹ️ Thông tin Scenario", expanded=False):
            st.write(f"**Tên:** {selected_scenario.name}")
            st.write(f"**Mô tả:** {selected_scenario.description or 'N/A'}")
            st.write(f"**Selected Fields:** `{', '.join(selected_scenario.selected_fields)}`")
            st.write(f"**Số câu hỏi:** {len(selected_scenario.questions)}")
        
        # Show scenario questions
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"💡 Scenario **{selected_scenario.name}** có **{len(selected_scenario.questions)}** câu hỏi. Click vào câu hỏi bên dưới để test!")
        with col2:
            if st.button("📋 Copy câu hỏi", key="copy_scenario_questions"):
                questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(selected_scenario.questions)])
                st.code(questions_text)
        
        # Context key unique per scenario
        context_key = f"scenario_{selected_scenario.id}"
        
        # Render chat interface with scenario questions as quick questions
        render_chat_interface(
            session=session,
            agent=agent,
            context_key=context_key,
            placeholder_text="Hỏi một câu từ scenario hoặc câu hỏi mới...",
            quick_questions=selected_scenario.questions
        )
        
        # Additional features for scenario testing
        st.divider()
        st.markdown("### 🧪 Test Tự Động")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption("Test tất cả câu hỏi trong scenario một lần")
        with col2:
            if st.button("▶️ Test All Questions", type="primary", key="auto_test_scenario"):
                # Clear current chat
                clear_chat_history(context_key)
                
                # Test each question với streaming
                for i, question in enumerate(selected_scenario.questions):
                    st.markdown(f"**[Q{i+1}] {question}**")
                    add_to_chat_history(context_key, "user", f"[Q{i+1}] {question}")
                    
                    # Streaming response
                    response_placeholder = st.empty()
                    response = stream_agent_response(agent, question, response_placeholder)
                    add_to_chat_history(context_key, "assistant", response)
                    
                    st.markdown("---")
                
                st.success(f"✅ Đã test {len(selected_scenario.questions)} câu hỏi!")
                st.rerun()


# ============================================================================
# Export chat history
# ============================================================================

def export_chat_history_to_json(context_key: str, filename: str = None):
    """Export chat history to JSON file"""
    import json
    
    chat_history = get_chat_history(context_key)
    
    if not chat_history:
        return None
    
    if filename is None:
        filename = f"chat_history_{context_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    export_data = {
        "context": context_key,
        "exported_at": datetime.now().isoformat(),
        "message_count": len(chat_history),
        "messages": chat_history
    }
    
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    return json_str


def render_export_chat_button(context_key: str):
    """Render button to export chat history"""
    import json
    
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
    """Get statistics about chat history"""
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
    
    avg_length = sum(len(msg["content"]) for msg in agent_msgs) / len(agent_msgs) if agent_msgs else 0
    
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