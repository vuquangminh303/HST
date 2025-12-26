"""
Schema Refinement Module
Provides functions for generating, reviewing, and refining schemas with AI assistance
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from openai import OpenAI

from hst_types import ColumnProfile, ColumnSchema, SchemaColumn
from schema_generation import ProfileGenerator, SchemaGenerator


# ============================================================================
# Initial Schema Generation (Fast - No Questions)
# ============================================================================

def generate_initial_schema_fast(
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    question_set=None
) -> Tuple[Dict[str, ColumnSchema], List[ColumnProfile]]:
    """
    Generate initial schema directly from data profiles without clarification questions

    Returns:
        (schema_dict, profiles_list)
    """
    # Generate profiles
    profiles = ProfileGenerator.generate_profiles(df)
    sample_rows = ProfileGenerator.get_sample_rows(df)

    # Generate schema directly
    schema_gen = SchemaGenerator(api_key=api_key, model=model)
    schema, _ = schema_gen.generate_schema(
        profiles=profiles,
        sample_rows=sample_rows,
        question_set=question_set,
        clarification_answers=[]  # No clarification answers
    )

    return schema, profiles


# ============================================================================
# AI Schema Improvement Suggestions
# ============================================================================

def generate_schema_improvement_suggestions(
    schema: Dict[str, ColumnSchema],
    profiles: List[ColumnProfile],
    api_key: str,
    model: str = "gpt-4o-mini",
    question_set=None
) -> Dict[str, List[str]]:
    """
    Generate AI suggestions for improving each column's schema

    Returns:
        {column_name: [suggestion1, suggestion2, ...]}
    """
    client = OpenAI(api_key=api_key)

    suggestions_dict = {}

    # Build context
    context_parts = ["**Current Schema:**"]
    for col_name, col_schema in schema.items():
        context_parts.append(f"\n**{col_name}:**")
        context_parts.append(f"- Type: {col_schema.semantic_type}")
        context_parts.append(f"- Description: {col_schema.description}")
        context_parts.append(f"- Unit: {col_schema.unit or 'N/A'}")
        context_parts.append(f"- Required: {col_schema.is_required}")

    context = "\n".join(context_parts)

    # Add question context if available
    question_context = ""
    if question_set and question_set.user_questions:
        questions_preview = [q.question for q in question_set.user_questions[:5]]
        question_context = f"\n\n**User Questions (for context):**\n" + "\n".join(f"- {q}" for q in questions_preview)

    system_prompt = """Bạn là một data expert. Phân tích schema hiện tại và đưa ra các gợi ý cải thiện.

Cho mỗi column, hãy đưa ra 2-3 gợi ý ngắn gọn về:
- Semantic type có phù hợp không?
- Description có rõ ràng không?
- Có thiếu constraints quan trọng không?
- Unit có cần cụ thể hơn không?

Format: JSON với key là column name, value là array các gợi ý."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}{question_context}\n\nĐưa ra gợi ý cải thiện cho từng column (JSON format)."}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        content = response.choices[0].message.content

        # Try to parse JSON
        try:
            # Extract JSON from markdown if present
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            suggestions_dict = json.loads(json_str)
        except:
            # Fallback: parse as text
            suggestions_dict = {col: ["AI suggestion generation failed - please review manually"] for col in schema.keys()}

    except Exception as e:
        # Return empty suggestions on error
        suggestions_dict = {col: [f"Error generating suggestions: {str(e)}"] for col in schema.keys()}

    return suggestions_dict


# ============================================================================
# Natural Language Schema Refinement Chat
# ============================================================================

def apply_schema_refinement_chat(
    user_request: str,
    current_schema: Dict[str, ColumnSchema],
    profiles: List[ColumnProfile],
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Tuple[Dict[str, ColumnSchema], str]:
    """
    Use LLM to refine schema based on natural language request

    Args:
        user_request: "Make description for column X more clear", "Add unit to price column", etc.
        current_schema: Current schema dict
        profiles: Column profiles
        api_key: OpenAI API key
        model: Model name

    Returns:
        (updated_schema, explanation)
    """
    client = OpenAI(api_key=api_key)

    # Serialize current schema for LLM
    schema_json = {}
    for col_name, col_schema in current_schema.items():
        schema_json[col_name] = {
            "description": col_schema.description,
            "semantic_type": col_schema.semantic_type,
            "physical_type": col_schema.physical_type,
            "unit": col_schema.unit,
            "is_required": col_schema.is_required,
            "constraints": col_schema.constraints
        }

    system_prompt = """Bạn là data schema expert. User sẽ yêu cầu sửa đổi schema.

Nhiệm vụ:
1. Hiểu yêu cầu của user
2. Sửa đổi schema theo yêu cầu
3. Trả về JSON với format:
{
  "updated_schema": {column_name: {description, semantic_type, physical_type, unit, is_required, constraints}, ...},
  "explanation": "Giải thích những gì đã thay đổi"
}

Chỉ sửa những phần user yêu cầu, giữ nguyên phần còn lại."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"**Schema hiện tại:**\n```json\n{json.dumps(schema_json, indent=2, ensure_ascii=False)}\n```\n\n**Yêu cầu:** {user_request}"}
            ],
            temperature=0.7,
            max_tokens=2500
        )

        content = response.choices[0].message.content

        # Parse JSON response
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            json_str = content.strip()

        result = json.loads(json_str)

        # Reconstruct ColumnSchema objects
        updated_schema = {}
        for col_name, col_data in result["updated_schema"].items():
            updated_schema[col_name] = ColumnSchema(
                description=col_data.get("description", ""),
                semantic_type=col_data.get("semantic_type", "unknown"),
                physical_type=col_data.get("physical_type", "string"),
                original_type=current_schema[col_name].original_type if col_name in current_schema else "unknown",
                is_required=col_data.get("is_required", False),
                unit=col_data.get("unit"),
                constraints=col_data.get("constraints")
            )

        explanation = result.get("explanation", "Schema updated successfully")

        return updated_schema, explanation

    except Exception as e:
        return current_schema, f"❌ Error: {str(e)}"


# ============================================================================
# Schema Editor UI Component
# ============================================================================

def render_schema_editor(
    column_name: str,
    schema_col: ColumnSchema,
    profile: ColumnProfile,
    suggestions: List[str] = None,
    key_prefix: str = ""
) -> ColumnSchema:
    """
    Render editable schema fields for a single column

    Returns:
        Updated ColumnSchema object
    """
    st.subheader(f"📝 Edit: {column_name}")

    # Show suggestions if available
    if suggestions:
        with st.expander("💡 AI Suggestions", expanded=False):
            for i, suggestion in enumerate(suggestions):
                st.info(f"{i+1}. {suggestion}")

    # Show data profile summary
    with st.expander("📊 Data Profile", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Non-Null", profile.non_null_count)
        with col2:
            st.metric("Null %", f"{profile.null_ratio:.1%}")
        with col3:
            st.metric("Unique", profile.n_unique)

        st.write("**Sample values:**")
        st.code(", ".join(str(v) for v in profile.sample_values[:5]))

    st.divider()

    # Editable fields
    col_left, col_right = st.columns(2)

    with col_left:
        new_description = st.text_area(
            "Description",
            value=schema_col.description,
            height=80,
            key=f"{key_prefix}_desc_{column_name}"
        )

        new_semantic_type = st.text_input(
            "Semantic Type",
            value=schema_col.semantic_type,
            help="e.g., email, phone_number, currency, date, etc.",
            key=f"{key_prefix}_sem_{column_name}"
        )

        new_unit = st.text_input(
            "Unit (optional)",
            value=schema_col.unit or "",
            help="e.g., USD, meters, seconds, etc.",
            key=f"{key_prefix}_unit_{column_name}"
        )

    with col_right:
        new_physical_type = st.selectbox(
            "Physical Type",
            options=["string", "integer", "float", "boolean", "date", "datetime", "json"],
            index=["string", "integer", "float", "boolean", "date", "datetime", "json"].index(schema_col.physical_type) if schema_col.physical_type in ["string", "integer", "float", "boolean", "date", "datetime", "json"] else 0,
            key=f"{key_prefix}_phys_{column_name}"
        )

        new_is_required = st.checkbox(
            "Required field",
            value=schema_col.is_required,
            key=f"{key_prefix}_req_{column_name}"
        )

        # Constraints (JSON editor)
        constraints_str = json.dumps(schema_col.constraints, indent=2) if schema_col.constraints else "{}"
        new_constraints_str = st.text_area(
            "Constraints (JSON)",
            value=constraints_str,
            height=80,
            help="e.g., {\"min\": 0, \"max\": 100}",
            key=f"{key_prefix}_const_{column_name}"
        )

        try:
            new_constraints = json.loads(new_constraints_str) if new_constraints_str.strip() else None
        except:
            st.error("Invalid JSON in constraints")
            new_constraints = schema_col.constraints

    # Return updated schema
    return ColumnSchema(
        description=new_description,
        semantic_type=new_semantic_type,
        physical_type=new_physical_type,
        original_type=schema_col.original_type,
        is_required=new_is_required,
        unit=new_unit if new_unit else None,
        constraints=new_constraints
    )


# ============================================================================
# Schema Refinement Chat Interface
# ============================================================================

def render_schema_refinement_chat(
    current_schema: Dict[str, ColumnSchema],
    profiles: List[ColumnProfile],
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Optional[Tuple[Dict[str, ColumnSchema], str]]:
    """
    Render chat interface for schema refinement with natural language

    Returns:
        (updated_schema, explanation) if user submitted request, else None
    """
    st.subheader("💬 Chat với AI để Refine Schema")

    st.info("Ví dụ: 'Làm rõ description cho column price', 'Thêm unit USD cho price', 'Đặt email là required'")

    with st.form("schema_refinement_chat_form"):
        user_request = st.text_area(
            "Yêu cầu sửa đổi schema:",
            height=100,
            placeholder="Ví dụ: Thêm constraints cho age column (min: 0, max: 120)"
        )

        submitted = st.form_submit_button("🚀 Apply Changes", type="primary")

    if submitted and user_request:
        with st.spinner("🧠 AI đang xử lý yêu cầu..."):
            updated_schema, explanation = apply_schema_refinement_chat(
                user_request=user_request,
                current_schema=current_schema,
                profiles=profiles,
                api_key=api_key,
                model=model
            )

            st.success(f"✅ {explanation}")
            return updated_schema, explanation

    return None
