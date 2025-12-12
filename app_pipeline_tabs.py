"""
New Pipeline Tab Functions
Following strict order: Structure → Type → Insights → Transform → Clean → Schema → Questions
"""

import streamlit as st
import pandas as pd
import json
from schema_pipeline import (
    StructureAnalyzer, TypeInferenceEngine, ProfileGenerator,
    DataInsightsAnalyzer, TransformationOption, CleaningRule
)


# ============================================================================
# Helper: Pipeline State Validation
# ============================================================================

def check_pipeline_step_allowed(step_name: str) -> bool:
    """Check if a pipeline step can be accessed based on previous steps"""

    if not st.session_state.session:
        return False

    session = st.session_state.session

    # Define prerequisites for each step
    prerequisites = {
        "structure_analysis": ["ingestion"],
        "type_inference": ["ingestion", "structure_analysis"],
        "data_insights": ["ingestion", "structure_analysis", "type_inference"],
        "transformations": ["ingestion", "structure_analysis", "type_inference", "data_insights"],
        "data_cleaning": ["ingestion", "structure_analysis", "type_inference", "transformations"],
        "schema_generation": ["ingestion", "structure_analysis", "type_inference", "data_cleaning"],
        "question_validation": ["schema_generation"],
        "agent_qa": ["schema_generation"]
    }

    if step_name not in prerequisites:
        return True  # No prerequisites

    # Check if all prerequisites are completed
    required_steps = prerequisites[step_name]

    # Initialize pipeline_state if needed
    if not hasattr(session, 'pipeline_state'):
        session.pipeline_state = {}

    for req in required_steps:
        if not session.pipeline_state.get(req):
            return False

    return True


def show_pipeline_warning(step_name: str, required_step: str):
    """Show warning when trying to access step without completing prerequisites"""
    st.warning(f"⚠️ Please complete **{required_step}** first before proceeding to {step_name}")
    st.info("💡 Follow the pipeline order to ensure correct data processing")


# ============================================================================
# Tab 2: Structure Analysis (MUST BE FIRST!)
# ============================================================================

def tab_structure_analysis_new():
    """Step 1: Detect and fix structural issues"""
    st.header("🏗️ Step 2: Structure Analysis")

    st.markdown("""
    **CRITICAL FIRST STEP:** Fix structure issues before any other analysis.

    Common issues:
    - First row should be header
    - Empty rows at top
    - Missing column names
    - Merged cells
    """)

    if not st.session_state.session:
        st.warning("⚠️ Please ingest data first (Tab 1)")
        return

    session = st.session_state.session

    # Select source
    selected_source_id = st.selectbox(
        "Select Source",
        options=[s.source_id for s in session.sources],
        key="struct_source_select"
    )

    df_raw = st.session_state.raw_dfs[selected_source_id]

    # Analyze button
    if st.button("🔍 Analyze Structure", type="primary"):
        with st.spinner("Analyzing structure..."):
            analyzer = StructureAnalyzer(
                api_key=st.session_state.api_key,
                model=st.session_state.model
            )

            # Just detect issues, don't infer types yet
            is_clean, issues = analyzer._heuristic_structure_check(df_raw)

            # Store results
            if 'structure_analysis' not in st.session_state:
                st.session_state.structure_analysis = {}

            st.session_state.structure_analysis[selected_source_id] = {
                'is_clean': is_clean,
                'issues': issues
            }

            # Mark step as completed
            if not hasattr(session, 'pipeline_state'):
                session.pipeline_state = {}
            session.pipeline_state['ingestion'] = True  # Assume ingestion done if we have data
            session.pipeline_state['structure_analysis'] = is_clean  # Only mark complete if clean

            st.rerun()

    # Display results
    if hasattr(st.session_state, 'structure_analysis') and selected_source_id in st.session_state.structure_analysis:
        results = st.session_state.structure_analysis[selected_source_id]

        if results['is_clean']:
            st.success("✅ Structure is clean! Ready for next step.")
            st.info("➡️ **Next:** Go to Tab 3 (Type Inference)")
        else:
            st.warning(f"⚠️ Found {len(results['issues'])} structural issues:")
            for issue in results['issues']:
                st.error(f"• {issue}")

            st.divider()
            st.subheader("🔧 Fix Required")
            st.info("Please fix structural issues before proceeding. Use transformations in Tab 5.")


# ============================================================================
# Tab 3: Type Inference
# ============================================================================

def tab_type_inference():
    """Step 2: Infer correct data types"""
    st.header("🔢 Step 3: Type Inference")

    # Check prerequisites
    if not check_pipeline_step_allowed("type_inference"):
        show_pipeline_warning("Type Inference", "Structure Analysis")
        return

    st.markdown("**Detect type mismatches** (e.g., numbers stored as strings)")

    session = st.session_state.session

    # Select source
    selected_source_id = st.selectbox(
        "Select Source",
        options=[s.source_id for s in session.sources],
        key="type_source_select"
    )

    # Get structurally-clean df (use transformed if available, else raw)
    df = st.session_state.clean_dfs.get(selected_source_id, st.session_state.raw_dfs[selected_source_id])

    if st.button("🔍 Analyze Types", type="primary"):
        with st.spinner("Analyzing types..."):
            # Generate profiles with type inference
            profiles = ProfileGenerator.generate_profiles(df)

            # Store results
            if 'type_inference' not in st.session_state:
                st.session_state.type_inference = {}

            st.session_state.type_inference[selected_source_id] = {
                'profiles': profiles
            }

            # Mark step complete
            session.pipeline_state['type_inference'] = True

            st.success("✅ Type inference complete!")
            st.rerun()

    # Display results
    if hasattr(st.session_state, 'type_inference') and selected_source_id in st.session_state.type_inference:
        results = st.session_state.type_inference[selected_source_id]

        # Show type mismatches
        type_data = []
        for col_name, profile in results['profiles'].items():
            type_data.append({
                "Column": col_name,
                "Stored As": profile.pandas_dtype,
                "Should Be": profile.inferred_type,
                "Mismatch": "⚠️" if profile.pandas_dtype != profile.inferred_type and profile.pandas_dtype == 'object' else "✅"
            })

        st.dataframe(pd.DataFrame(type_data), use_container_width=True)

        st.info("➡️ **Next:** Go to Tab 4 (Data Insights)")


# ============================================================================
# Tab 4: Data Insights
# ============================================================================

def tab_data_insights():
    """Step 3: Analyze patterns and anomalies"""
    st.header("🔍 Step 4: Data Insights Analysis")

    # Check prerequisites
    if not check_pipeline_step_allowed("data_insights"):
        show_pipeline_warning("Data Insights", "Type Inference")
        return

    st.markdown("**Discover patterns, anomalies, and correlations** in your data")

    session = st.session_state.session

    # Select source
    selected_source_id = st.selectbox(
        "Select Source",
        options=[s.source_id for s in session.sources],
        key="insights_source_select"
    )

    # Get df and profiles
    df = st.session_state.clean_dfs.get(selected_source_id, st.session_state.raw_dfs[selected_source_id])

    if 'type_inference' not in st.session_state or selected_source_id not in st.session_state.type_inference:
        st.error("Please complete Type Inference first (Tab 3)")
        return

    profiles = st.session_state.type_inference[selected_source_id]['profiles']

    if st.button("🔍 Analyze Insights", type="primary"):
        with st.spinner("Analyzing data patterns..."):
            analyzer = DataInsightsAnalyzer(
                api_key=st.session_state.api_key,
                model=st.session_state.model
            )

            insights = analyzer.analyze_insights(df, profiles)

            # Store results
            if 'data_insights' not in st.session_state:
                st.session_state.data_insights = {}

            st.session_state.data_insights[selected_source_id] = insights

            # Mark step complete
            session.pipeline_state['data_insights'] = True

            st.success("✅ Insights analysis complete!")
            st.rerun()

    # Display results
    if hasattr(st.session_state, 'data_insights') and selected_source_id in st.session_state.data_insights:
        insights = st.session_state.data_insights[selected_source_id]

        # Summary
        st.subheader("📊 Summary")
        st.write(insights.summary)

        st.divider()

        # Patterns
        if insights.patterns:
            st.subheader(f"🔍 Detected Patterns ({len(insights.patterns)})")
            for p in insights.patterns:
                with st.expander(f"**{p.column}**: {p.pattern_type} (Confidence: {p.confidence:.0%})"):
                    st.write(p.description)
                    if p.details:
                        st.json(p.details)

        # Anomalies
        if insights.anomalies:
            st.divider()
            st.subheader(f"⚠️ Detected Anomalies ({len(insights.anomalies)})")
            for a in insights.anomalies:
                severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}
                with st.expander(f"{severity_emoji[a.severity]} **{a.column}**: {a.anomaly_type}"):
                    st.write(a.description)
                    st.write(f"Affected rows: {a.affected_rows}")
                    if a.examples:
                        st.write("Examples:", a.examples[:5])

        st.info("➡️ **Next:** Go to Tab 5 (Transformations)")


# ============================================================================
# Placeholder tabs (to be implemented)
# ============================================================================

def tab_transformations():
    st.header("🔧 Step 5: Transformation Suggestions")
    st.info("Based on insights, suggest transformations (to be implemented)")

def tab_data_cleaning():
    st.header("🧹 Step 6: Data Cleaning")
    st.info("Apply type conversions (to be implemented)")
