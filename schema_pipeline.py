"""
Enhanced Data Schema Analysis Pipeline V5
- Smart type inference (handles 500.000 → int, not str)
- Data cleaning and type conversion
- Question-driven schema generation
- Schema validation against user questions
- Interactive Q&A agent after schema refinement
- Full Streamlit integration
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import argparse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from openai import OpenAI
import logging
import sys
import sqlite3


logging.basicConfig(level=os.getenv('LOGLEVEL','INFO').upper())
logger = logging.getLogger(__name__)

# ============================================================================
# Domain Models
# ============================================================================

class TransformationType(str, Enum):
    """Types of structural transformations"""
    SKIP_ROWS = "skip_rows"
    USE_ROW_AS_HEADER = "use_row_as_header"
    DROP_COLUMNS = "drop_columns"
    DROP_ROWS = "drop_rows"
    RENAME_COLUMNS = "rename_columns"
    SPLIT_COLUMN = "split_column"
    MERGE_COLUMNS = "merge_columns"
    CUSTOM = "custom"


class DataCleaningAction(str, Enum):
    """Types of data cleaning actions"""
    REMOVE_THOUSAND_SEPARATOR = "remove_thousand_separator"
    CONVERT_TO_INT = "convert_to_int"
    CONVERT_TO_FLOAT = "convert_to_float"
    CONVERT_TO_DATETIME = "convert_to_datetime"
    STRIP_WHITESPACE = "strip_whitespace"
    NORMALIZE_CASE = "normalize_case"
    REPLACE_VALUES = "replace_values"


class CleaningRule(BaseModel):
    """A data cleaning rule"""
    id: str
    column: str
    action: DataCleaningAction
    description: str
    params: Dict[str, Any] = Field(default_factory=dict)
    applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class Transformation(BaseModel):
    """A structural transformation to apply to the DataFrame"""
    id: str
    type: TransformationType
    description: str
    params: Dict[str, Any]
    confidence: float
    applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TransformationOption(BaseModel):
    """A set of transformations representing one approach to clean data"""
    id: str
    name: str  # e.g., "Auto-detect and fix structure", "Custom user transform"
    description: str
    transformations: List[Transformation]
    confidence: float
    is_recommended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ColumnProfile(BaseModel):
    """Statistical profile of a single column"""
    name: str
    pandas_dtype: str
    inferred_type: str  # After smart inference
    non_null_count: int
    null_count: int
    null_ratio: float
    n_unique: int
    sample_values: List[Any]
    sample_raw_values: List[str]  # Original string values
    has_thousand_separator: bool = False
    decimal_separator: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ColumnSchema(BaseModel):
    """Schema definition for a single column"""
    name: str
    description: str
    semantic_type: str
    physical_type: str
    original_type: str  # Before cleaning
    unit: Optional[str]
    is_required: bool
    constraints: Optional[Dict[str, Any]] = None
    cleaning_applied: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class Question(BaseModel):
    """A question requiring human clarification"""
    id: str
    question: str
    suggested_answer: Optional[str]
    target: str
    question_type: str = "semantic"
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class Answer(BaseModel):
    """Human answer to a question"""
    question_id: str
    answer: str
    timestamp: str
    applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class AgentMessage(BaseModel):
    """Message in agent conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DataFrameCheckpoint(BaseModel):
    """Checkpoint of DataFrame state"""
    checkpoint_id: str
    stage: str
    timestamp: str
    shape: Tuple[int, int]
    description: str
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class HistoryEntry(BaseModel):
    """Single entry in refinement history"""
    timestamp: str
    action: str
    details: Dict[str, Any]
    schema_version: int
    checkpoint_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DataSource(BaseModel):
    """Information about a data source"""
    source_id: str
    file_path: str
    sheet_name: Optional[str] = None
    source_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class UserQuestion(BaseModel):
    """User-defined question that will be frequently asked"""
    id: str
    question: str
    description: str = ""  # Optional description of what the question is about
    priority: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class OutputField(BaseModel):
    """Expected output field definition"""
    field_name: str
    description: str
    data_type: str  # Expected data type (string, number, date, etc.)
    required: bool = True
    example_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class QuestionSet(BaseModel):
    """Collection of user questions and expected output format"""
    user_questions: List[UserQuestion] = Field(default_factory=list)
    output_fields: List[OutputField] = Field(default_factory=list)
    additional_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_questions": [q.to_dict() for q in self.user_questions],
            "output_fields": [f.to_dict() for f in self.output_fields],
            "additional_notes": self.additional_notes
        }


class Session(BaseModel):
    """Complete session state"""
    session_id: str
    created_at: str

    sources: List[DataSource]
    current_source_id: str

    raw_structure_info: Dict[str, Any]
    transformations: List[Transformation]
    applied_transformations: List[str]

    # Data cleaning
    cleaning_rules: List[CleaningRule] = Field(default_factory=list)
    applied_cleaning_rules: List[str] = Field(default_factory=list)

    # User questions and output format
    question_set: Optional[QuestionSet] = None
    
    schema: Dict[str, ColumnSchema]
    profiles: Dict[str, ColumnProfile]
    
    questions: List[Question]
    answers: List[Answer]
    
    # Agent Q&A
    agent_conversations: List[AgentMessage] = Field(default_factory=list)
    agent_enabled: bool = True
    
    history: List[HistoryEntry]
    checkpoints: List[DataFrameCheckpoint]
    schema_version: int = 1
    
    is_clean_structure: bool = False
    structure_issues: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "sources": [s.to_dict() for s in self.sources],
            "current_source_id": self.current_source_id,
            "raw_structure_info": self.raw_structure_info,
            "transformations": [t.to_dict() for t in self.transformations],
            "applied_transformations": self.applied_transformations,
            "cleaning_rules": [r.to_dict() for r in self.cleaning_rules],
            "applied_cleaning_rules": self.applied_cleaning_rules,
            "question_set": self.question_set.to_dict() if self.question_set else None,
            "schema": {k: v.to_dict() for k, v in self.schema.items()},
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "questions": [q.to_dict() for q in self.questions],
            "answers": [a.to_dict() for a in self.answers],
            "agent_conversations": [m.to_dict() for m in self.agent_conversations],
            "agent_enabled": self.agent_enabled,
            "history": [h.to_dict() for h in self.history],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "schema_version": self.schema_version,
            "is_clean_structure": self.is_clean_structure,
            "structure_issues": self.structure_issues
        }




class SQLQueryTool:
    """Tool to execute SQL queries on the cleaned data"""
    
    def __init__(self, db_path: str, df: pd.DataFrame, table_name: str = "data"):
        self.db_path = db_path
        self.table_name = table_name
        self.conn = None
        self._create_database(df)
    
    def _create_database(self, df: pd.DataFrame):
        """Create SQLite database from DataFrame"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            df.to_sql(self.table_name, self.conn, if_exists='replace', index=False)
            
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns = cursor.fetchall()
            
            logger.info(f"✓ Created SQL database: {self.db_path}")
            logger.info(f"  Table: {self.table_name}, Columns: {len(columns)}, Rows: {len(df)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create database: {str(e)}")
    
    def execute_query(self, query: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """Execute SQL query and return results"""
        try:
            query_upper = query.upper().strip()
            if not query_upper.startswith('SELECT'):
                return pd.DataFrame(), "Error: Only SELECT queries allowed"
            
            dangerous = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
            for keyword in dangerous:
                if keyword in query_upper:
                    return pd.DataFrame(), f"Error: {keyword} not allowed"
            
            result_df = pd.read_sql_query(query, self.conn)
            return result_df, None
            
        except Exception as e:
            return pd.DataFrame(), f"SQL Error: {str(e)}"
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get table schema information"""
        try:
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
        if self.conn:
            self.conn.close()

# ============================================================================
# Type Inference & Data Cleaning
# ============================================================================

class TypeInferenceEngine:
    """Smart type inference that handles formatted numbers"""
    
    @staticmethod
    def detect_thousand_separator(series: pd.Series) -> Optional[str]:
        """Detect thousand separator (. or ,)"""
        sample = series.dropna().astype(str).head(100)
        
        # Check for patterns like 1.000, 500.000.000
        dot_pattern = r'^\d{1,3}(\.\d{3})+$'
        comma_pattern = r'^\d{1,3}(,\d{3})+$'
        
        dot_matches = sum(1 for val in sample if re.match(dot_pattern, val.strip()))
        comma_matches = sum(1 for val in sample if re.match(comma_pattern, val.strip()))
        
        if dot_matches > len(sample) * 0.5:
            return '.'
        if comma_matches > len(sample) * 0.5:
            return ','
        
        return None
    
    @staticmethod
    def detect_decimal_separator(series: pd.Series, thousand_sep: Optional[str]) -> Optional[str]:
        """Detect decimal separator"""
        sample = series.dropna().astype(str).head(100)
        
        # If thousand sep is '.', decimal is ','
        if thousand_sep == '.':
            # Check for pattern like 1.000,50
            pattern = r'\d+\.\d{3},\d{1,2}$'
            if sum(1 for val in sample if re.match(pattern, val.strip())) > 0:
                return ','
        
        # If thousand sep is ',', decimal is '.'
        elif thousand_sep == ',':
            # Check for pattern like 1,000.50
            pattern = r'\d+,\d{3}\.\d{1,2}$'
            if sum(1 for val in sample if re.match(pattern, val.strip())) > 0:
                return '.'
        
        # Default: check for single decimal point
        if any('.' in str(val) and str(val).count('.') == 1 for val in sample):
            return '.'
        
        return None
    
    @staticmethod
    def infer_type(series: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """
        Infer actual data type from series.
        Returns: (inferred_type, metadata)
        """
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return 'int', {}
            else:
                return 'float', {}
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime', {}
        
        if series.dtype == 'object':
            sample = series.dropna()
            if len(sample) == 0:
                return 'string', {}
            
            thousand_sep = TypeInferenceEngine.detect_thousand_separator(sample)
            decimal_sep = TypeInferenceEngine.detect_decimal_separator(sample, thousand_sep)
            
            metadata = {
                'thousand_separator': thousand_sep,
                'decimal_separator': decimal_sep
            }
            
            try:
                if thousand_sep:
                    cleaned = sample.astype(str).str.replace(thousand_sep, '', regex=False)
                    
                    if decimal_sep and decimal_sep != '.':
                        cleaned = cleaned.str.replace(decimal_sep, '.', regex=False)
                    
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                    
                    if numeric.notna().sum() / len(numeric) > 0.8:
                        if (numeric.dropna() % 1 == 0).all():
                            return 'int', metadata
                        else:
                            return 'float', metadata
                
                # Try direct numeric conversion
                numeric = pd.to_numeric(sample, errors='coerce')
                if numeric.notna().sum() / len(numeric) > 0.8:
                    if (numeric.dropna() % 1 == 0).all():
                        return 'int', {}
                    else:
                        return 'float', {}
                
                # Try datetime
                dt = pd.to_datetime(sample, errors='coerce')
                if dt.notna().sum() / len(dt) > 0.8:
                    return 'datetime', {}
                
            except:
                pass
        
        return 'string', {}
    
    @staticmethod
    def generate_cleaning_rules(df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> List[CleaningRule]:
        """Generate cleaning rules based on inferred types"""
        rules = []
        
        for col_name, profile in profiles.items():
            # Skip if inferred type matches pandas type
            if profile.inferred_type == 'int' and df[col_name].dtype in ['int64', 'int32']:
                continue
            if profile.inferred_type == 'float' and df[col_name].dtype in ['float64', 'float32']:
                continue
            
            # Generate cleaning rule for number with thousand separator
            if profile.has_thousand_separator:
                if profile.decimal_separator:
                    # Has both thousand and decimal
                    rules.append(CleaningRule(
                        id=f"clean_{col_name}_numeric",
                        column=col_name,
                        action=DataCleaningAction.CONVERT_TO_FLOAT,
                        description=f"Convert '{col_name}' from formatted number to float (remove {profile.sample_raw_values[0]!r} → float)",
                        params={
                            'thousand_separator': '.',
                            'decimal_separator': profile.decimal_separator
                        }
                    ))
                else:
                    # Only thousand separator (integer)
                    rules.append(CleaningRule(
                        id=f"clean_{col_name}_int",
                        column=col_name,
                        action=DataCleaningAction.CONVERT_TO_INT,
                        description=f"Convert '{col_name}' from formatted number to integer (remove dots: {profile.sample_raw_values[0]!r} → int)",
                        params={
                            'thousand_separator': '.'
                        }
                    ))
            
            # Generate rule for plain numeric conversion
            elif profile.inferred_type == 'int' and df[col_name].dtype == 'object':
                rules.append(CleaningRule(
                    id=f"clean_{col_name}_to_int",
                    column=col_name,
                    action=DataCleaningAction.CONVERT_TO_INT,
                    description=f"Convert '{col_name}' to integer",
                    params={}
                ))
            
            elif profile.inferred_type == 'float' and df[col_name].dtype == 'object':
                rules.append(CleaningRule(
                    id=f"clean_{col_name}_to_float",
                    column=col_name,
                    action=DataCleaningAction.CONVERT_TO_FLOAT,
                    description=f"Convert '{col_name}' to float",
                    params={}
                ))
            
            elif profile.inferred_type == 'datetime' and df[col_name].dtype == 'object':
                rules.append(CleaningRule(
                    id=f"clean_{col_name}_to_datetime",
                    column=col_name,
                    action=DataCleaningAction.CONVERT_TO_DATETIME,
                    description=f"Convert '{col_name}' to datetime",
                    params={}
                ))
        
        return rules
    
    @staticmethod
    def apply_cleaning_rule(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
        """Apply a single cleaning rule"""
        df_result = df.copy()
        col = rule.column
        
        try:
            if rule.action == DataCleaningAction.CONVERT_TO_INT:
                # Remove thousand separator if specified
                if 'thousand_separator' in rule.params:
                    df_result[col] = df_result[col].astype(str).str.replace(
                        rule.params['thousand_separator'], '', regex=False
                    )
                
                # Convert to int
                df_result[col] = pd.to_numeric(df_result[col], errors='coerce').astype('Int64')
                logger.info(f"✓ Converted '{col}' to integer")
            
            elif rule.action == DataCleaningAction.CONVERT_TO_FLOAT:
                series = df_result[col].astype(str)
                
                # Remove thousand separator
                if 'thousand_separator' in rule.params:
                    series = series.str.replace(rule.params['thousand_separator'], '', regex=False)
                
                # Replace decimal separator
                if 'decimal_separator' in rule.params and rule.params['decimal_separator'] != '.':
                    series = series.str.replace(rule.params['decimal_separator'], '.', regex=False)
                
                df_result[col] = pd.to_numeric(series, errors='coerce')
                logger.info(f"✓ Converted '{col}' to float")
            
            elif rule.action == DataCleaningAction.CONVERT_TO_DATETIME:
                df_result[col] = pd.to_datetime(df_result[col], errors='coerce')
                logger.info(f"✓ Converted '{col}' to datetime")
            
            elif rule.action == DataCleaningAction.STRIP_WHITESPACE:
                df_result[col] = df_result[col].astype(str).str.strip()
                logger.info(f"✓ Stripped whitespace from '{col}'")
            
            elif rule.action == DataCleaningAction.NORMALIZE_CASE:
                case_type = rule.params.get('case', 'lower')
                if case_type == 'lower':
                    df_result[col] = df_result[col].astype(str).str.lower()
                elif case_type == 'upper':
                    df_result[col] = df_result[col].astype(str).str.upper()
                logger.info(f"✓ Normalized case for '{col}'")
            
        except Exception as e:
            logger.error(f"✗ Failed to apply cleaning rule {rule.id}: {str(e)}")
            raise
        
        return df_result


# ============================================================================
# Data Ingestion (same as V3)
# ============================================================================

class DataIngestor:
    """Handles file loading and validation with multi-file/multi-sheet support"""
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    
    @classmethod
    def discover_sources(cls, file_paths: List[str]) -> List[DataSource]:
        """Discover all data sources from file paths"""
        sources = []
        
        for file_path in file_paths:
            path = Path(file_path)
            
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
                logger.warning(f"Unsupported file: {file_path}")
                continue
            
            if path.suffix.lower() == '.csv':
                source = DataSource(
                    source_id=f"{path.stem}_csv",
                    file_path=file_path,
                    source_type="csv"
                )
                sources.append(source)
            else:
                try:
                    xls = pd.ExcelFile(file_path)
                    for sheet_name in xls.sheet_names:
                        source = DataSource(
                            source_id=f"{path.stem}_{sheet_name}",
                            file_path=file_path,
                            sheet_name=sheet_name,
                            source_type="xlsx_sheet"
                        )
                        sources.append(source)
                except Exception as e:
                    logger.warning(f"Failed to read Excel sheets from {file_path}: {e}")
        
        logger.info(f"✓ Discovered {len(sources)} data source(s)")
        return sources
    
    @classmethod
    def load_source(cls, source: DataSource, header: Optional[int] = None) -> pd.DataFrame:
        """Load a specific data source"""
        try:
            if source.source_type == "csv":
                df = pd.read_csv(source.file_path, header=header)
            else:
                df = pd.read_excel(
                    source.file_path, 
                    sheet_name=source.sheet_name,
                    header=header
                )
            
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            logger.info(f"✓ Loaded {source.source_id}: {len(df)} rows × {len(df.columns)} columns")
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {source.source_id}: {str(e)}")
    
    @classmethod
    def load_raw(cls, source: DataSource) -> pd.DataFrame:
        """Load raw data without headers"""
        return cls.load_source(source, header=None)


# ============================================================================
# Structure Analysis (from V3)
# ============================================================================

class StructureAnalyzer:
    """Analyzes raw DataFrame structure and types, proposes multiple transformation options"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze_structure_and_types(
        self,
        df: pd.DataFrame,
        max_preview_rows: int = 10
    ) -> Tuple[Dict[str, Any], List[TransformationOption], Dict[str, ColumnProfile], List[CleaningRule], List[Question], bool]:
        """
        Comprehensive analysis: structure + types, propose multiple transformation options.
        Returns: (structure_info, transformation_options, column_profiles, cleaning_rules, questions, is_clean)
        """

        # First, perform heuristic structure check
        is_clean_structure, structure_issues = self._heuristic_structure_check(df)

        # Extract structure info
        structure_info = self._extract_structure_info(df, max_preview_rows)

        # Analyze types (regardless of structure cleanliness)
        column_profiles = {}
        for col in df.columns:
            inferred_type, metadata = TypeInferenceEngine.infer_type(df[col])
            profile = ColumnProfile(
                name=str(col),
                pandas_dtype=str(df[col].dtype),
                inferred_type=inferred_type,
                non_null_count=int(df[col].notna().sum()),
                null_count=int(df[col].isna().sum()),
                null_ratio=float(df[col].isna().sum() / len(df)) if len(df) > 0 else 0.0,
                n_unique=int(df[col].nunique()),
                sample_values=df[col].dropna().head(5).tolist(),
                sample_raw_values=df[col].dropna().head(5).astype(str).tolist(),
                has_thousand_separator=bool(metadata.get('thousand_separator')),
                decimal_separator=metadata.get('decimal_separator')
            )
            column_profiles[str(col)] = profile

        # Generate type cleaning rules
        cleaning_rules = TypeInferenceEngine.generate_cleaning_rules(df, column_profiles)

        # If structure is clean, return with only type cleaning
        if is_clean_structure:
            logger.info("✓ Data structure appears clean")
            logger.info(f"✓ Generated {len(cleaning_rules)} type cleaning rules")

            # Create a "no structure change" option
            options = [
                TransformationOption(
                    id="no_structure_change",
                    name="Keep Current Structure",
                    description="Data structure is clean, only type cleaning needed",
                    transformations=[],
                    confidence=1.0,
                    is_recommended=True
                )
            ]

            return structure_info, options, column_profiles, cleaning_rules, [], True

        logger.info(f"⚠ Detected structure issues: {', '.join(structure_issues)}")

        # Perform LLM analysis to get multiple transformation options
        prompt = self._build_enhanced_structure_prompt(structure_info, structure_issues, column_profiles)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_enhanced_structure_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5  # Higher temp for more diverse options
            )

            result = json.loads(response.choices[0].message.content)

            # Parse transformation options
            transformation_options = []
            for opt in result.get("transformation_options", []):
                transformations = [Transformation(**t) for t in opt.get("transformations", [])]
                option = TransformationOption(
                    id=opt["id"],
                    name=opt["name"],
                    description=opt["description"],
                    transformations=transformations,
                    confidence=opt.get("confidence", 0.5),
                    is_recommended=opt.get("is_recommended", False)
                )
                transformation_options.append(option)

            # Add a "custom" option for user-defined transforms
            transformation_options.append(
                TransformationOption(
                    id="custom_user_transform",
                    name="Custom Transformation",
                    description="Define your own transformation steps",
                    transformations=[],
                    confidence=0.0,
                    is_recommended=False
                )
            )

            questions = [Question(**q) for q in result.get("questions", [])]

            logger.info(f"✓ Analyzed structure: {len(transformation_options)} options proposed, {len(questions)} questions")
            return structure_info, transformation_options, column_profiles, cleaning_rules, questions, False

        except Exception as e:
            raise RuntimeError(f"Structure analysis failed: {str(e)}")
    
    def _heuristic_structure_check(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Fast heuristic check for common structure issues.
        Returns: (is_clean, list_of_issues)
        """
        issues = []
        
        first_row = df.iloc[0]
        potential_headers = first_row.astype(str).tolist()
        
        numeric_headers = sum(1 for val in potential_headers if str(val).replace('.', '').replace('-', '').isdigit())
        if numeric_headers > len(potential_headers) * 0.5:
            issues.append("First row contains mostly numeric values (should be headers)")
        
        if all(str(col).isdigit() for col in df.columns):
            issues.append("Column names are numeric indices (no header row detected)")
        
        empty_cols = sum(1 for col in df.columns if pd.isna(col) or str(col).strip() == '')
        if empty_cols > 0:
            issues.append(f"{empty_cols} empty column name(s)")
        
        # Check 4: Duplicate column names
        if len(df.columns) != len(set(df.columns)):
            issues.append("Duplicate column names detected")
        
        # Check 5: Excessive nulls in first few rows (might need to skip)
        if len(df) > 3:
            first_rows_null_ratio = df.head(3).isna().sum().sum() / (3 * len(df.columns))
            if first_rows_null_ratio > 0.7:
                issues.append("First 3 rows are mostly empty (might need to skip)")
        
        # Check 6: Are column names actually descriptive?
        if not all(str(col).isdigit() for col in df.columns):
            # We have string column names - check if they're descriptive
            avg_length = sum(len(str(col)) for col in df.columns) / len(df.columns)
            if avg_length < 3:
                issues.append("Column names are too short (less than 3 characters on average)")
        
        is_clean = len(issues) == 0
        
        return is_clean, issues
    
    def _extract_structure_info(self, df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
        """Extract structure information for LLM analysis"""
        preview_rows = min(max_rows, len(df))
        
        return {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "preview_data": df.head(preview_rows).to_dict(orient='records'),
            "null_counts": df.isna().sum().to_dict(),
            "first_row_values": df.iloc[0].tolist() if len(df) > 0 else []
        }
    
    def _build_enhanced_structure_prompt(
        self,
        info: Dict[str, Any],
        detected_issues: List[str],
        column_profiles: Dict[str, ColumnProfile]
    ) -> str:
        """Build enhanced prompt for structure analysis with multiple options"""

        # Summarize type issues
        type_issues = []
        for col_name, profile in column_profiles.items():
            if profile.has_thousand_separator:
                type_issues.append(f"Column '{col_name}': has thousand separator (e.g., {profile.sample_raw_values[0]})")
            if profile.inferred_type != profile.pandas_dtype and profile.pandas_dtype == 'object':
                type_issues.append(f"Column '{col_name}': stored as string but inferred as {profile.inferred_type}")

        return f"""Analyze this DataFrame and propose MULTIPLE transformation options.

**Detected Structure Issues:**
{chr(10).join('- ' + issue for issue in detected_issues)}

**Detected Type Issues:**
{chr(10).join('- ' + issue for issue in type_issues[:5])}
(Total {len(type_issues)} type issues detected)

**DataFrame Info:**
- Shape: {info['shape']['rows']} rows × {info['shape']['columns']} columns
- Current column names: {info['column_names']}
- First row values: {info['first_row_values'][:10]}

**Preview (first few rows):**
```json
{json.dumps(info['preview_data'][:5], indent=2, default=str)}
```

**Column Type Summary:**
{json.dumps({col: f"{profile.pandas_dtype} → {profile.inferred_type}" for col, profile in list(column_profiles.items())[:10]}, indent=2)}

**Instructions:**
1. Propose 2-3 DIFFERENT approaches to fix the structure issues
2. Each approach should be a complete transformation pipeline
3. Consider conservative vs aggressive fixes
4. Examples of different approaches:
   - Option 1: Minimal fix (only critical issues)
   - Option 2: Recommended fix (balance safety & completeness)
   - Option 3: Aggressive fix (maximum cleanup)

**Response Format:**
{{
  "transformation_options": [
    {{
      "id": "minimal_fix",
      "name": "Minimal Fix",
      "description": "Only fix critical structural issues",
      "transformations": [
        {{
          "id": "unique_id",
          "type": "use_row_as_header" | "skip_rows" | "drop_columns" | "rename_columns" | ...,
          "description": "Human-readable description",
          "params": {{"row_index": 0}} or {{"rows_to_skip": 2}} etc,
          "confidence": 0.95
        }}
      ],
      "confidence": 0.9,
      "is_recommended": false
    }},
    {{
      "id": "recommended_fix",
      "name": "Recommended Fix",
      "description": "Balanced approach fixing structure + preparing for type cleaning",
      "transformations": [...],
      "confidence": 0.95,
      "is_recommended": true
    }}
  ],
  "questions": [
    {{
      "id": "question_id",
      "question": "Should we...?",
      "suggested_answer": "yes" or null,
      "target": "transform.unique_id",
      "question_type": "structural"
    }}
  ]
}}

**IMPORTANT:**
- Propose at least 2 different options
- Mark one as is_recommended=true
- Type cleaning will be handled separately - focus on structural transformations here
"""

    def _build_structure_prompt(self, info: Dict[str, Any], detected_issues: List[str]) -> str:
        """Build prompt for structure analysis (legacy, kept for compatibility)"""
        return f"""Analyze this raw DataFrame structure and propose transformations.

**Detected Issues (from heuristics):**
{chr(10).join('- ' + issue for issue in detected_issues)}

**DataFrame Info:**
- Shape: {info['shape']['rows']} rows × {info['shape']['columns']} columns
- Current column names: {info['column_names']}
- First row values: {info['first_row_values'][:10]}

**Preview (first few rows):**
```json
{json.dumps(info['preview_data'][:5], indent=2, default=str)}
```

**Instructions:**
1. Determine if this data needs structural transformation
2. Common issues to look for:
   - First row should be used as header
   - Empty rows at the top that should be skipped
   - Unnamed or poorly named columns
   - Empty columns that should be dropped
   - Merged header rows that need special handling

3. For each transformation, propose:
   - A unique ID (e.g., "use_row_0_as_header")
   - Type of transformation
   - Description
   - Parameters
   - Confidence score (0.0-1.0)

4. If you're uncertain about any transformation, create a question for the user

**Response Format:**
{{
  "transformations": [
    {{
      "id": "unique_id",
      "type": "use_row_as_header" | "skip_rows" | "drop_columns" | "rename_columns" | ...,
      "description": "Human-readable description",
      "params": {{"row_index": 0}} or {{"rows_to_skip": 2}} etc,
      "confidence": 0.95
    }}
  ],
  "questions": [
    {{
      "id": "question_id",
      "question": "Should we...?",
      "suggested_answer": "yes" or null,
      "target": "transform.unique_id",
      "question_type": "structural"
    }}
  ]
}}

**IMPORTANT:** If the data already has proper headers and clean structure, return empty transformations list!
"""
    
    def _get_enhanced_structure_system_prompt(self) -> str:
        """Enhanced system prompt for structure analysis with multiple options"""
        return """You are an expert data analyst specializing in detecting and fixing structural issues in raw tabular data.

Your job is to:
1. Identify structural problems (wrong headers, empty rows, malformed columns)
2. Propose MULTIPLE different transformation approaches (2-3 options)
3. Each option should represent a different strategy (minimal, recommended, aggressive)
4. Ask clarifying questions when uncertain

Guidelines:
- Minimal Fix: Only fix critical issues that would break data processing
- Recommended Fix: Balanced approach, fix structure issues + prepare for type cleaning
- Aggressive Fix: Maximum cleanup, drop suspicious columns/rows, rename everything

Always respond in valid JSON format with "transformation_options" and "questions" arrays.
Each option must have: id, name, description, transformations array, confidence, is_recommended."""

    def _get_structure_system_prompt(self) -> str:
        """System prompt for structure analysis (legacy)"""
        return """You are an expert data analyst specializing in detecting and fixing structural issues in raw tabular data.

Your job is to:
1. Identify structural problems (wrong headers, empty rows, malformed columns)
2. Propose specific transformations to fix them
3. Ask clarifying questions when uncertain

Be conservative: if the data looks clean, don't propose unnecessary transformations.

Always respond in valid JSON format with "transformations" and "questions" arrays."""
    
    @staticmethod
    def apply_transformation(df: pd.DataFrame, trans: Transformation) -> pd.DataFrame:
        """Apply a single transformation to DataFrame"""
        df_result = df.copy()
        
        try:
            if trans.type == TransformationType.USE_ROW_AS_HEADER:
                row_idx = trans.params["row_index"]
                df_result.columns = df_result.iloc[row_idx].astype(str).tolist()
                df_result = df_result.iloc[row_idx + 1:].reset_index(drop=True)
                
            elif trans.type == TransformationType.SKIP_ROWS:
                rows_to_skip = trans.params["rows_to_skip"]
                df_result = df_result.iloc[rows_to_skip:].reset_index(drop=True)
                
            elif trans.type == TransformationType.DROP_COLUMNS:
                cols = trans.params["columns"]
                df_result = df_result.drop(columns=cols, errors='ignore')
                
            elif trans.type == TransformationType.DROP_ROWS:
                indices = trans.params["indices"]
                df_result = df_result.drop(index=indices, errors='ignore').reset_index(drop=True)
                
            elif trans.type == TransformationType.RENAME_COLUMNS:
                mapping = trans.params["mapping"]
                df_result = df_result.rename(columns=mapping)
            
            logger.info(f"✓ Applied: {trans.description}")

        except Exception as e:
            logger.error(f"✗ Failed to apply {trans.id}: {str(e)}")
            raise

        return df_result

    @staticmethod
    def apply_transformation_option(df: pd.DataFrame, option: TransformationOption) -> pd.DataFrame:
        """Apply all transformations in an option sequentially"""
        df_result = df.copy()

        logger.info(f"Applying transformation option: {option.name}")
        for trans in option.transformations:
            df_result = StructureAnalyzer.apply_transformation(df_result, trans)

        logger.info(f"✓ Completed transformation option: {option.name}")
        return df_result


# ============================================================================
# Profile Generation (Enhanced with Type Inference)
# ============================================================================

class ProfileGenerator:
    """Generate statistical profiles with smart type inference"""
    
    @staticmethod
    def generate_profiles(
        df: pd.DataFrame, 
        max_samples: int = 10
    ) -> Dict[str, ColumnProfile]:
        """Generate profile for each column with type inference"""
        profiles = {}
        
        for col in df.columns:
            series = df[col]
            non_null = series.notna().sum()
            null = series.isna().sum()
            total = len(series)
            
            # Sample values
            samples = series.dropna().head(max_samples).tolist()
            raw_samples = series.dropna().astype(str).head(max_samples).tolist()
            
            # Type inference
            inferred_type, metadata = TypeInferenceEngine.infer_type(series)
            
            profile = ColumnProfile(
                name=str(col),
                pandas_dtype=str(series.dtype),
                inferred_type=inferred_type,
                non_null_count=int(non_null),
                null_count=int(null),
                null_ratio=float(null / total) if total > 0 else 0.0,
                n_unique=int(series.nunique()),
                sample_values=samples,
                sample_raw_values=raw_samples,
                has_thousand_separator=metadata.get('thousand_separator') is not None,
                decimal_separator=metadata.get('decimal_separator')
            )
            profiles[str(col)] = profile
        
        return profiles
    
    @staticmethod
    def get_sample_rows(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows"""
        return df.head(n).to_dict(orient='records')


class SchemaGenerator:
    """Generate semantic schema using LLM with type awareness"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_clarification_questions(
        self,
        profiles: Dict[str, ColumnProfile],
        sample_rows: List[Dict[str, Any]],
        question_set: Optional[QuestionSet] = None
    ) -> List[Question]:
        """
        Analyze data and generate clarification questions before schema generation.

        Examples: unit of measurement, data format, constraints, relationships, etc.
        """
        prompt = self._build_clarification_prompt(profiles, sample_rows, question_set)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_clarification_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)
            questions = [Question(**q) for q in result.get("questions", [])]

            logger.info(f"✓ Generated {len(questions)} clarification questions")
            return questions

        except Exception as e:
            raise RuntimeError(f"Clarification question generation failed: {str(e)}")

    def _build_clarification_prompt(
        self,
        profiles: Dict[str, ColumnProfile],
        sample_rows: List[Dict[str, Any]],
        question_set: Optional[QuestionSet] = None
    ) -> str:
        """Build prompt for clarification questions"""

        profiles_json = {col: prof.to_dict() for col, prof in profiles.items()}

        user_context = ""
        if question_set and (question_set.user_questions or question_set.output_fields):
            user_context = f"""

**User's Expected Usage:**
- Questions: {json.dumps([q.question for q in question_set.user_questions], indent=2)}
- Output Fields: {json.dumps([f.field_name for f in question_set.output_fields], indent=2)}
- Notes: {question_set.additional_notes}
"""

        return f"""Analyze this data and generate clarification questions to better understand the schema.{user_context}

**Column Profiles:**
```json
{json.dumps(profiles_json, indent=2, default=str)}
```

**Sample Rows:**
```json
{json.dumps(sample_rows[:5], indent=2, default=str)}
```

Generate clarification questions about:
1. **Units**: For numeric columns (price, area, salary, etc.) - what is the unit? (VND, USD, m², etc.)
2. **Format**: For date/text columns - what format is expected?
3. **Constraints**: Any min/max values, allowed ranges?
4. **Relationships**: Foreign keys, categorical hierarchies?
5. **Business Context**: What does this column represent in business terms?

Focus on questions that will help generate an accurate, useful schema.

**Response Format:**
{{
  "questions": [
    {{
      "id": "salary_unit",
      "question": "What is the currency unit for the 'Salary' column?",
      "suggested_answer": "VND",
      "target": "Salary.unit",
      "question_type": "semantic"
    }},
    {{
      "id": "area_unit",
      "question": "What is the unit of measurement for 'Area'?",
      "suggested_answer": "m2",
      "target": "Area.unit",
      "question_type": "semantic"
    }}
  ]
}}

Generate 3-10 most important clarification questions.
"""

    def _get_clarification_system_prompt(self) -> str:
        """System prompt for clarification questions"""
        return """You are a data analyst expert who asks clarifying questions about data schemas.

Your job is to analyze column profiles and sample data, then ask questions that will help:
1. Understand units of measurement for numeric columns
2. Clarify date/time formats
3. Identify constraints and validation rules
4. Understand business context and relationships

Ask specific, actionable questions. Each question should help define a specific schema attribute.
Be concise but thorough. Prioritize questions that impact data interpretation.

Always respond in valid JSON format."""

    def generate_schema(
        self,
        profiles: Dict[str, ColumnProfile],
        sample_rows: List[Dict[str, Any]],
        question_set: Optional[QuestionSet] = None,
        clarification_answers: Optional[List[Answer]] = None
    ) -> Tuple[Dict[str, ColumnSchema], List[Question]]:
        """Generate schema with semantic types, guided by user questions and clarification answers"""

        prompt = self._build_schema_prompt(profiles, sample_rows, question_set, clarification_answers)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_schema_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            schema = {}
            for col, spec in result["schema"].items():
                # Add original_type from profile
                original_type = profiles[col].pandas_dtype
                spec['original_type'] = original_type
                schema[col] = ColumnSchema(**spec)
            
            questions = [Question(**q) for q in result.get("questions", [])]
            
            logger.info(f"✓ Generated schema for {len(schema)} columns")
            return schema, questions
            
        except Exception as e:
            raise RuntimeError(f"Schema generation failed: {str(e)}")
    
    def _build_schema_prompt(
        self,
        profiles: Dict[str, ColumnProfile],
        sample_rows: List[Dict[str, Any]],
        question_set: Optional[QuestionSet] = None,
        clarification_answers: Optional[List[Answer]] = None
    ) -> str:
        """Build prompt with type inference info, user questions, and clarification answers"""

        profiles_json = {}
        for col, prof in profiles.items():
            prof_dict = prof.to_dict()
            # Add inference hints
            if prof.has_thousand_separator:
                prof_dict['note'] = f"Has thousand separator (e.g., {prof.sample_raw_values[0]!r}), should be {prof.inferred_type}"
            profiles_json[col] = prof_dict

        # Build clarification answers context
        clarification_context = ""
        if clarification_answers:
            clarification_context = "\n\n**Clarification Answers:**\n"
            for answer in clarification_answers:
                clarification_context += f"- {answer.question_id}: {answer.answer}\n"

        # Build user questions context if provided
        user_context = ""
        if question_set and (question_set.user_questions or question_set.output_fields):
            user_context = f"""

**IMPORTANT: User's Expected Usage**

The user has specified how they intend to use this data:

**Questions the user will ask:**
{json.dumps([q.to_dict() for q in question_set.user_questions], indent=2)}

**Expected output fields needed:**
{json.dumps([f.to_dict() for f in question_set.output_fields], indent=2)}

**Additional notes:**
{question_set.additional_notes}

Please ensure the schema you generate has enough semantic information to:
1. Answer the user's questions accurately
2. Map to the expected output fields
3. Provide clear descriptions that help answer these questions
"""

        return f"""Analyze column profiles and generate semantic schema.{user_context}{clarification_context}

**Column Profiles (with type inference):**
```json
{json.dumps(profiles_json, indent=2, default=str)}
```

**Sample Rows:**
```json
{json.dumps(sample_rows, indent=2, default=str)}
```

Generate schema for each column:

**Response Format:**
{{
  "schema": {{
    "column_name": {{
      "name": "column_name",
      "description": "Brief description",
      "semantic_type": "categorical" | "numeric" | "boolean" | "datetime" | "text" | "identifier",
      "physical_type": "int64" | "float64" | "string" | "datetime64[ns]",
      "unit": "m" | "m2" | "VND" | "USD" | null,
      "is_required": true | false,
      "constraints": {{"min": 0}} or null
    }}
  }},
  "questions": [
    {{
      "id": "price_unit",
      "question": "What is the unit of the 'price' column?",
      "suggested_answer": "VND",
      "target": "price.unit",
      "question_type": "semantic"
    }}
  ]
}}

IMPORTANT: Use the inferred_type to set physical_type correctly (not pandas_dtype if it's object but should be int/float).
"""
    
    def _get_schema_system_prompt(self) -> str:
        """System prompt"""
        return """You are an expert data analyst specializing in semantic schema inference.

Analyze profiles and determine:
1. Semantic type (what it represents)
2. Physical type (actual data type - use inferred_type from profile)
3. Units if applicable
4. Required status (based on null_ratio)
5. Constraints

Pay attention to type inference hints - if a column has "inferred_type": "int" but "pandas_dtype": "object",
it means the data has formatting (like 500.000) and should be treated as int after cleaning.

Always respond in valid JSON."""


class SchemaValidator:
    """Validate if schema can answer user questions and suggest refinements"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def validate_schema_for_questions(
        self,
        schema: Dict[str, ColumnSchema],
        profiles: Dict[str, ColumnProfile],
        question_set: QuestionSet,
        sample_rows: List[Dict[str, Any]]
    ) -> Tuple[bool, List[Question], str]:
        """
        Validate if current schema can answer user questions.

        Returns:
            - is_sufficient: Whether schema is sufficient
            - additional_questions: Questions to ask if schema is insufficient
            - validation_report: Detailed report of the validation
        """
        prompt = self._build_validation_prompt(schema, profiles, question_set, sample_rows)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_validation_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)

            is_sufficient = result.get("is_sufficient", False)
            additional_questions = [Question(**q) for q in result.get("additional_questions", [])]
            validation_report = result.get("validation_report", "")

            logger.info(f"✓ Schema validation: {'Sufficient' if is_sufficient else 'Needs refinement'}")
            if not is_sufficient:
                logger.info(f"  Additional questions needed: {len(additional_questions)}")

            return is_sufficient, additional_questions, validation_report

        except Exception as e:
            raise RuntimeError(f"Schema validation failed: {str(e)}")

    def _build_validation_prompt(
        self,
        schema: Dict[str, ColumnSchema],
        profiles: Dict[str, ColumnProfile],
        question_set: QuestionSet,
        sample_rows: List[Dict[str, Any]]
    ) -> str:
        """Build validation prompt"""

        schema_json = {k: v.to_dict() for k, v in schema.items()}
        profiles_json = {k: v.to_dict() for k, v in profiles.items()}

        return f"""Validate if the current schema can adequately answer the user's questions.

**User Questions to Answer:**
{json.dumps([q.to_dict() for q in question_set.user_questions], indent=2)}

**Expected Output Fields:**
{json.dumps([f.to_dict() for f in question_set.output_fields], indent=2)}

**Additional Notes:**
{question_set.additional_notes}

**Current Schema:**
```json
{json.dumps(schema_json, indent=2, default=str)}
```

**Column Profiles:**
```json
{json.dumps(profiles_json, indent=2, default=str)}
```

**Sample Data Rows:**
```json
{json.dumps(sample_rows[:5], indent=2, default=str)}
```

Analyze whether the current schema provides enough information to:
1. Answer each user question accurately
2. Generate the expected output fields
3. Handle the data transformations required

**Response Format:**
{{
  "is_sufficient": true | false,
  "validation_report": "Detailed explanation of what can/cannot be answered",
  "additional_questions": [
    {{
      "id": "unique_id",
      "question": "What specific information is needed?",
      "suggested_answer": "Possible answer",
      "target": "schema_field_path",
      "question_type": "semantic"
    }}
  ],
  "missing_fields": ["field1", "field2"],
  "unclear_mappings": {{
    "user_question_id": "explanation of why it's unclear"
  }}
}}
"""

    def _get_validation_system_prompt(self) -> str:
        """System prompt for validation"""
        return """You are an expert data schema validator.

Your task is to determine if a data schema is sufficient to answer specific user questions.

Analyze:
1. Can each user question be answered with the current schema?
2. Are the expected output fields mappable to schema columns?
3. Are there ambiguities or missing information?
4. What additional clarifications are needed?

Be thorough but practical. Only request clarifications that are truly necessary.

Always respond in valid JSON format."""


# ============================================================================
# Agent Q&A System
# ============================================================================

# ============================================================================
# Data Schema Agent - Flexible OpenAI Agent
# ============================================================================

# ============================================================================
# Data Schema Agent - OpenAI Agent Standard
# ============================================================================

class DataSchemaAgent:
    """
    Data Schema Agent following OpenAI Agent standard

    Features:
    - Streaming query interface (Generator pattern)
    - SQL query execution with dynamic schema
    - Helper methods: get_sample_rows, get_distinct_values, execute_query
    - Can be instantiated and used from any code
    - Fully flexible - adapts to any schema
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        session: 'Session',
        api_key: str,
        model: str = None,
        df_cleaned: Optional[pd.DataFrame] = None,
        sql_tool: Optional['SQLQueryTool'] = None,
        system_prompt: str = None
    ):
        """
        Initialize Data Schema Agent

        Args:
            session: Session with schema info
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            df_cleaned: Cleaned DataFrame (optional)
            sql_tool: SQL query tool (optional, will create if df_cleaned provided)
            system_prompt: Custom system prompt (optional)
        """
        self.session = session
        self.df_cleaned = df_cleaned
        self.client = OpenAI(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        self.sql_tool = sql_tool

        # System prompt
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Create SQL database if cleaned data provided and no sql_tool
        if df_cleaned is not None and sql_tool is None:
            db_dir = Path("./agent_databases")
            db_dir.mkdir(exist_ok=True)

            self.db_path = db_dir / f"data_{session.session_id}.db"
            self.sql_tool = SQLQueryTool(str(self.db_path), df_cleaned)
            logger.info("✓ Agent initialized with SQL capability")

        logger.info(f"DataSchemaAgent initialized with model: {self.model}")

    def _default_system_prompt(self) -> str:
        """Default system prompt for data schema agent"""
        return """You are an intelligent data analyst assistant with SQL query capabilities.

**Your Capabilities:**
- Understand and explain data schemas
- Execute SQL queries to analyze data
- Provide insights and recommendations
- Answer questions about data with evidence

**SQL Guidelines:**
- You can only execute SELECT queries (no INSERT, UPDATE, DELETE)
- Table name: 'data'
- Use column names exactly as shown in the schema
- Use LIMIT for previews (e.g., LIMIT 10)
- Use aggregations: COUNT(), SUM(), AVG(), MIN(), MAX()
- Use GROUP BY for category analysis
- Use WHERE for filtering
- Use ORDER BY for sorting

**Query Examples:**
- Preview data: `SELECT * FROM data LIMIT 5`
- Calculate average: `SELECT AVG(price) FROM data WHERE category='A'`
- Count by category: `SELECT category, COUNT(*) as count FROM data GROUP BY category ORDER BY count DESC`
- Find top records: `SELECT * FROM data ORDER BY price DESC LIMIT 10`

**Response Guidelines:**
- Always explain your reasoning
- Use specific numbers and facts from query results
- Provide actionable insights
- Be concise and helpful

When you need to query data, use the execute_sql_query tool."""

    def query(
        self,
        question: str,
        conversation_history: List[Dict[str, Any]] = None,
        model: str = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Query with streaming response

        Args:
            question: User question
            conversation_history: Previous conversation (optional)
            model: Model to override default (optional)

        Yields:
            Response text chunks

        Returns:
            Metadata dict with usage info
        """
        import time
        start_time = time.time()

        try:
            # Build context
            context = self._build_context()

            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"**Context:**\n{context}\n\n**Question:**\n{question}"}
            ]

            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages
                    messages.append(msg)

            # Determine if SQL tools should be available
            tools = self._get_tools() if self.sql_tool else None

            # Call OpenAI
            use_model = model or self.model

            if tools:
                # With SQL capability - use tools
                response = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=1000
                )

                assistant_msg = response.choices[0].message

                # Handle tool calls
                if assistant_msg.tool_calls:
                    # Execute tools and get results
                    tool_results = self._handle_tool_calls(assistant_msg, question)

                    # Stream tool results
                    for char in tool_results:
                        yield char

                    final_text = tool_results
                else:
                    # No tools called, stream response
                    final_text = assistant_msg.content or ""
                    for char in final_text:
                        yield char

                # Calculate usage
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            else:
                # Without SQL - simple streaming response
                response = self.client.chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500,
                    stream=True
                )

                final_text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        final_text += text
                        yield text

                usage = {"estimated_tokens": len(final_text) // 4}  # Rough estimate

            # Record in conversation
            if hasattr(self, 'session'):
                self.session.agent_conversations.append(AgentMessage(
                    role="user",
                    content=question,
                    timestamp=datetime.now().isoformat()
                ))
                self.session.agent_conversations.append(AgentMessage(
                    role="assistant",
                    content=final_text,
                    timestamp=datetime.now().isoformat(),
                    context={"used_sql": bool(tools)}
                ))

            # Return metadata
            return {
                "usage": usage,
                "duration": time.time() - start_time,
                "model": use_model,
                "used_sql": bool(tools)
            }

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            error_msg = f"\n\n⚠️ Error: {str(e)}"
            yield error_msg

            return {
                "error": str(e),
                "duration": time.time() - start_time,
                "model": model or self.model
            }

    def chat(self, user_message: str) -> str:
        """
        Simple chat interface (non-streaming)

        Args:
            user_message: User message

        Returns:
            Assistant response text
        """
        result = []
        metadata = None

        for chunk in self.query(user_message):
            if isinstance(chunk, dict):
                metadata = chunk
            else:
                result.append(chunk)

        return "".join(result)

    def _handle_tool_calls(self, assistant_message, original_question: str) -> str:
        """
        Handle SQL query execution from tool calls

        Args:
            assistant_message: OpenAI assistant message with tool calls
            original_question: Original user question

        Returns:
            Response text with query results and analysis
        """
        responses = []

        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == "execute_sql_query":
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    query = arguments.get("query", "")

                    logger.info(f"🔍 Executing SQL: {query}")

                    # Execute query
                    result_df, error = self._execute_sql(query)

                    if error:
                        responses.append(f"❌ SQL Error: {error}")
                        logger.warning(f"SQL execution failed: {error}")
                    else:
                        if len(result_df) == 0:
                            responses.append("✅ Query executed successfully but returned no results")
                        else:
                            # Format results
                            display_df = result_df.head(10)
                            result_text = f"✅ Query Results ({len(result_df)} rows returned):\n\n{display_df.to_string(index=False)}"
                            if len(result_df) > 10:
                                result_text += f"\n\n(Showing first 10 of {len(result_df)} rows)"
                            responses.append(result_text)
                            logger.info(f"SQL execution successful: {len(result_df)} rows")

                except Exception as e:
                    logger.error(f"Tool execution error: {str(e)}")
                    responses.append(f"❌ Error executing query: {str(e)}")

        # Combine all tool responses
        if responses:
            result_summary = "\n\n".join(responses)

            # Get LLM interpretation of results
            try:
                interp_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a data analyst. Interpret SQL query results and provide insights to answer the user's question."},
                        {"role": "user", "content": f"**Original Question:** {original_question}\n\n**Query Results:**\n{result_summary}\n\nPlease interpret these results and answer the question:"}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                interpretation = interp_response.choices[0].message.content
                return f"{result_summary}\n\n**Analysis:**\n{interpretation}"
            except Exception as e:
                logger.error(f"Interpretation error: {str(e)}")
                return result_summary

        return "No results from query execution."

    def _execute_sql(self, query: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Execute SQL query

        Args:
            query: SQL query string

        Returns:
            Tuple of (result DataFrame, error message)
        """
        if not self.sql_tool:
            return pd.DataFrame(), "SQL capability not enabled"

        try:
            result_df, error = self.sql_tool.execute_query(query)
            return result_df, error
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return pd.DataFrame(), str(e)

    def _get_sample_rows(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample rows from data

        Args:
            limit: Number of rows to return

        Returns:
            List of row dicts
        """
        if self.df_cleaned is None:
            return []

        try:
            sample_df = self.df_cleaned.head(limit)
            return sample_df.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Failed to get sample rows: {str(e)}")
            return []

    def _get_distinct_values(self, column_name: str, limit: int = 50) -> List[Any]:
        """
        Get distinct values for a column

        Args:
            column_name: Column name
            limit: Maximum number of values to return

        Returns:
            List of distinct values
        """
        if self.df_cleaned is None:
            return []

        try:
            if column_name not in self.df_cleaned.columns:
                logger.warning(f"Column '{column_name}' not found in data")
                return []

            distinct_values = self.df_cleaned[column_name].dropna().unique()[:limit]
            return distinct_values.tolist()
        except Exception as e:
            logger.error(f"Failed to get distinct values: {str(e)}")
            return []

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information

        Returns:
            Dict with schema information
        """
        schema_info = {
            "table_name": "data",
            "columns": [],
            "row_count": 0
        }

        if self.session.schema:
            for col_name, col_schema in self.session.schema.items():
                schema_info["columns"].append({
                    "name": col_name,
                    "semantic_type": col_schema.semantic_type,
                    "physical_type": col_schema.physical_type,
                    "unit": col_schema.unit,
                    "description": col_schema.description,
                    "required": col_schema.is_required
                })

        if self.df_cleaned is not None:
            schema_info["row_count"] = len(self.df_cleaned)

        return schema_info

    def _get_tools(self) -> List[Dict[str, Any]]:
        """
        Define SQL query tool dynamically based on actual schema

        Returns:
            List of tool definitions for OpenAI function calling
        """
        if not self.sql_tool:
            return []

        # Get schema info
        schema_info = self.get_schema_info()

        # Build column descriptions from session schema
        column_descriptions = []
        if self.session.schema:
            for col_name, col_schema in list(self.session.schema.items())[:20]:
                desc = f"{col_name} ({col_schema.semantic_type}, {col_schema.physical_type})"
                if col_schema.unit:
                    desc += f" - Unit: {col_schema.unit}"
                if col_schema.description:
                    desc += f" - {col_schema.description}"
                column_descriptions.append(desc)

        columns_desc = "\n".join(column_descriptions) if column_descriptions else "Use PRAGMA table_info(data) to see columns"

        return [{
            "type": "function",
            "function": {
                "name": "execute_sql_query",
                "description": f"""Execute SQL SELECT query on the data table.

**Table:** {schema_info['table_name']}
**Rows:** {schema_info['row_count']}

**Available Columns:**
{columns_desc}

Use this tool to:
- Analyze data (aggregations, statistics)
- Filter and search records
- Count, sum, average values
- Group by categories
- Find top/bottom records

Only SELECT queries are allowed.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT query. Use standard SQL syntax. Examples: 'SELECT * FROM data LIMIT 5', 'SELECT AVG(column) FROM data', 'SELECT category, COUNT(*) FROM data GROUP BY category'"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]

    def _build_context(self) -> str:
        """Build context from session state"""
        parts = []

        # Data source info
        if self.session.sources:
            source = self.session.sources[0]
            parts.append(f"**Data Source:** {source.file_path}")
            if source.sheet_name:
                parts.append(f"**Sheet:** {source.sheet_name}")

        # Data shape
        if self.df_cleaned is not None:
            parts.append(f"**Data Shape:** {self.df_cleaned.shape[0]} rows × {self.df_cleaned.shape[1]} columns")

        # Schema summary (first 10 columns)
        if self.session.schema:
            schema_lines = []
            for col, col_schema in list(self.session.schema.items())[:10]:
                line = f"- **{col}**: {col_schema.semantic_type} ({col_schema.physical_type})"
                if col_schema.unit:
                    line += f", Unit: {col_schema.unit}"
                schema_lines.append(line)

            if len(self.session.schema) > 10:
                schema_lines.append(f"... and {len(self.session.schema) - 10} more columns")

            parts.append("**Schema:**\n" + "\n".join(schema_lines))

        # SQL capability status
        if self.sql_tool:
            parts.append(f"**SQL Database:** Ready for queries")

        return "\n\n".join(parts)

    def enable_sql(self, df_cleaned: pd.DataFrame, sql_tool: Optional['SQLQueryTool'] = None):
        """
        Enable SQL capability

        Args:
            df_cleaned: Cleaned DataFrame
            sql_tool: Optional SQL tool to use (will create if not provided)
        """
        if sql_tool:
            self.sql_tool = sql_tool
            self.df_cleaned = df_cleaned
            logger.info("✓ SQL capability enabled with provided tool")
        elif self.sql_tool is None:
            db_dir = Path("./agent_databases")
            db_dir.mkdir(exist_ok=True)

            self.db_path = db_dir / f"data_{self.session.session_id}.db"
            self.sql_tool = SQLQueryTool(str(self.db_path), df_cleaned)
            self.df_cleaned = df_cleaned
            logger.info("✓ SQL capability enabled for agent")

    def close(self):
        """Close database connection"""
        if self.sql_tool:
            try:
                self.sql_tool.close()
                logger.info("SQL tool closed")
            except Exception as e:
                logger.error(f"Error closing SQL tool: {str(e)}")


class RefinementEngine:
    """Handle schema refinement based on user answers"""
    
    def __init__(self, session: 'Session'):
        self.session = session
    
    def apply_answer(self, answer: Answer) -> bool:
        """Apply user answer to schema"""
        try:
            question = next(q for q in self.session.questions if q.id == answer.question_id)
            
            parts = question.target.split(".")
            if len(parts) != 2:
                logger.warning(f"Invalid target: {question.target}")
                return False
            
            col_name, field = parts
            
            if col_name not in self.session.schema:
                logger.warning(f"Column not found: {col_name}")
                return False
            
            schema_col = self.session.schema[col_name]
            
            if field == "unit":
                schema_col.unit = answer.answer
            elif field == "description":
                schema_col.description = answer.answer
            elif field == "semantic_type":
                schema_col.semantic_type = answer.answer
            elif field == "physical_type":
                schema_col.physical_type = answer.answer
            elif field == "is_required":
                schema_col.is_required = answer.answer.lower() in ('true', 'yes', '1')
            else:
                logger.warning(f"Unknown field: {field}")
                return False
            
            answer.applied = True
            
            self.session.history.append(HistoryEntry(
                timestamp=datetime.now().isoformat(),
                action="question_answered",
                details={
                    "question_id": answer.question_id,
                    "question": question.question,
                    "answer": answer.answer,
                    "target": question.target
                },
                schema_version=self.session.schema_version
            ))
            
            self.session.schema_version += 1
            
            logger.info(f"✓ Applied answer to {question.target}: {answer.answer}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply answer: {str(e)}")
            return False


# ============================================================================
# Session Management (Enhanced with Cleaning)
# ============================================================================

class SessionManager:
    """Manage session state and checkpoints"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
    
    def create_session(
        self,
        sources: List[DataSource],
        current_source_id: str
    ) -> Session:
        """Create new session"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session = Session(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            sources=sources,
            current_source_id=current_source_id,
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
        
        logger.info(f"✓ Created session: {session_id}")
        return session
    
    def save_checkpoint(
        self,
        session: Session,
        df: pd.DataFrame,
        stage: str,
        description: str
    ) -> DataFrameCheckpoint:
        """Save DataFrame checkpoint"""
        checkpoint_id = f"{session.session_id}_{stage}_{len(session.checkpoints)}"
        timestamp = datetime.now().isoformat()
        
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.parquet"
        df.to_parquet(checkpoint_file, index=False)
        
        checkpoint = DataFrameCheckpoint(
            checkpoint_id=checkpoint_id,
            stage=stage,
            timestamp=timestamp,
            shape=(len(df), len(df.columns)),
            description=description,
            file_path=str(checkpoint_file)
        )
        
        session.checkpoints.append(checkpoint)
        
        logger.info(f"✓ Checkpoint: {stage} ({checkpoint.shape[0]}×{checkpoint.shape[1]})")
        return checkpoint
    
    def load_checkpoint(self, checkpoint: DataFrameCheckpoint) -> pd.DataFrame:
        """Load DataFrame from checkpoint"""
        if not checkpoint.file_path:
            raise ValueError("No file path")
        
        df = pd.read_parquet(checkpoint.file_path)
        logger.info(f"✓ Loaded: {checkpoint.stage}")
        return df
    
    def save_session(self, session: Session):
        """Save session to JSON"""
        session_file = self.output_dir / f"session_{session.session_id}.json"
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✓ Session saved: {session_file}")


# ============================================================================
# CLI Interface (Enhanced)
# ============================================================================

class CLI:
    """Command-line interface"""
    
    def __init__(self, args):
        self.args = args
        self.session_manager = SessionManager(args.output_dir)
    
    def run(self):
        """Execute pipeline"""
        
        logger.info("="*70)
        logger.info("Enhanced Data Schema Analysis Pipeline V4")
        logger.info("="*70)
        
        # Discover sources
        logger.info("\nStep 1: Discovering data sources...")
        file_paths = self.args.file_paths if isinstance(self.args.file_paths, list) else [self.args.file_paths]
        sources = DataIngestor.discover_sources(file_paths)
        
        if not sources:
            raise ValueError("No valid sources found")
        
        # Create session
        session = self.session_manager.create_session(
            sources=sources,
            current_source_id=sources[0].source_id
        )
        
        # Process each source
        for source in sources:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing: {source.source_id}")
            logger.info(f"{'='*70}")
            
            self._process_source(session, source)
        
        # Agent Q&A
        if not self.args.skip_agent and session.schema:
            logger.info("\n" + "="*70)
            logger.info("Agent Q&A: Ask questions about your data schema")
            logger.info("="*70)
            self._agent_chat(session)
        
        # Save
        self.session_manager.save_session(session)
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ Complete! Session: {session.session_id}")
        logger.info(f"✓ Output: {self.args.output_dir}")
        logger.info("="*70 + "\n")
    
    def _process_source(self, session: Session, source: DataSource):
        """Process single source"""
        
        # Load raw
        logger.info(f"\nLoading raw data...")
        df_raw = DataIngestor.load_raw(source)
        
        checkpoint_raw = self.session_manager.save_checkpoint(
            session, df_raw, "raw", f"Raw: {source.source_id}"
        )
        
        # Structure analysis
        logger.info("\nAnalyzing structure...")
        analyzer = StructureAnalyzer(api_key=self.args.api_key, model=self.args.model)
        
        structure_info, transformations, structural_qs, is_clean = analyzer.analyze_structure(df_raw)
        
        session.raw_structure_info[source.source_id] = structure_info
        session.is_clean_structure = is_clean
        
        df_clean = df_raw.copy()
        applied_trans = []
        
        # Handle transformations
        if transformations and not self.args.skip_interactive:
            df_clean, applied_trans = self._handle_transformations(df_clean, transformations, session)
        elif transformations and self.args.auto_transform:
            for t in transformations:
                if t.confidence >= 0.9:
                    logger.info(f"Auto-applying: {t.description}")
                    df_clean = StructureAnalyzer.apply_transformation(df_clean, t)
                    applied_trans.append(t.id)
        
        session.transformations.extend(transformations)
        session.applied_transformations.extend(applied_trans)
        
        checkpoint_clean = self.session_manager.save_checkpoint(
            session, df_clean, "after_structure", "After structure transforms"
        )
        
        # Generate profiles with type inference
        logger.info("\nGenerating profiles with type inference...")
        profiles = ProfileGenerator.generate_profiles(df_clean)
        
        # Show type inference results
        for col_name, profile in profiles.items():
            if profile.inferred_type != 'string' and df_clean[col_name].dtype == 'object':
                logger.info(f"  Column '{col_name}': {profile.pandas_dtype} → {profile.inferred_type}")
                if profile.has_thousand_separator:
                    logger.info(f"    Example: {profile.sample_raw_values[0]!r}")
        
        # Generate cleaning rules
        logger.info("\nGenerating data cleaning rules...")
        cleaning_rules = TypeInferenceEngine.generate_cleaning_rules(df_clean, profiles)
        session.cleaning_rules.extend(cleaning_rules)
        
        if cleaning_rules:
            logger.info(f"Found {len(cleaning_rules)} cleaning rule(s)")
            
            # Apply cleaning
            df_cleaned = df_clean.copy()
            applied_cleaning = []
            
            if not self.args.skip_interactive:
                df_cleaned, applied_cleaning = self._handle_cleaning(df_cleaned, cleaning_rules, session)
            elif self.args.auto_clean:
                for rule in cleaning_rules:
                    df_cleaned = TypeInferenceEngine.apply_cleaning_rule(df_cleaned, rule)
                    applied_cleaning.append(rule.id)
            
            session.applied_cleaning_rules.extend(applied_cleaning)
            
            # Save checkpoint after cleaning
            checkpoint_cleaned = self.session_manager.save_checkpoint(
                session, df_cleaned, "cleaned", "After data cleaning"
            )
            
            # Regenerate profiles after cleaning
            profiles = ProfileGenerator.generate_profiles(df_cleaned)
            
            df_final = df_cleaned
            
        else:
            logger.info("No cleaning needed")
            df_final = df_clean
        self._final_df = df_final  # Save for agent
        sample_rows = ProfileGenerator.get_sample_rows(df_final)
        
        # Generate schema
        logger.info("\nGenerating semantic schema...")
        schema_gen = SchemaGenerator(api_key=self.args.api_key, model=self.args.model)
        schema, semantic_qs = schema_gen.generate_schema(profiles, sample_rows)
        
        session.profiles.update(profiles)
        session.schema.update(schema)
        session.questions.extend(structural_qs + semantic_qs)
        
        # Interactive refinement
        if semantic_qs and not self.args.skip_interactive:
            logger.info("\nInteractive refinement...")
            self._interactive_refinement(session, semantic_qs)
    
    def _handle_transformations(
        self,
        df: pd.DataFrame,
        transformations: List[Transformation],
        session: Session
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Handle transformations interactively"""
        df_result = df.copy()
        applied = []
        
        logger.info(f"\nFound {len(transformations)} transformation(s):\n")
        
        for i, trans in enumerate(transformations, 1):
            logger.info(f"[{i}/{len(transformations)}] {trans.description}")
            logger.info(f"    Confidence: {trans.confidence:.2%}")
            
            user_input = input(f"    Apply? [y/N]: ").strip().lower()
            
            if user_input in ('y', 'yes'):
                df_result = StructureAnalyzer.apply_transformation(df_result, trans)
                applied.append(trans.id)
                
                self.session_manager.save_checkpoint(
                    session, df_result, f"trans_{trans.id}", trans.description
                )
            
            print()
        
        return df_result, applied
    
    def _handle_cleaning(
        self,
        df: pd.DataFrame,
        rules: List[CleaningRule],
        session: Session
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Handle cleaning rules interactively"""
        df_result = df.copy()
        applied = []
        
        logger.info(f"\nFound {len(rules)} cleaning rule(s):\n")
        
        for i, rule in enumerate(rules, 1):
            logger.info(f"[{i}/{len(rules)}] {rule.description}")
            logger.info(f"    Column: {rule.column}")
            logger.info(f"    Action: {rule.action}")
            
            user_input = input(f"    Apply? [y/N]: ").strip().lower()
            
            if user_input in ('y', 'yes'):
                df_result = TypeInferenceEngine.apply_cleaning_rule(df_result, rule)
                applied.append(rule.id)
                
                self.session_manager.save_checkpoint(
                    session, df_result, f"clean_{rule.id}", rule.description
                )
            
            print()
        
        return df_result, applied
    
    def _interactive_refinement(self, session: Session, questions: List[Question]):
        """Handle Q&A"""
        engine = RefinementEngine(session)
        
        semantic_qs = [q for q in questions if q.question_type == "semantic"]
        
        if not semantic_qs:
            return
        
        logger.info(f"\nFound {len(semantic_qs)} question(s):\n")
        
        for i, question in enumerate(semantic_qs, 1):
            logger.info(f"[{i}/{len(semantic_qs)}] {question.question}")
            if question.suggested_answer:
                logger.info(f"    Suggested: {question.suggested_answer}")
            
            user_input = input(f"    Answer (or Enter for suggestion): ").strip()
            
            if not user_input and question.suggested_answer:
                user_input = question.suggested_answer
            
            if user_input:
                answer = Answer(
                    question_id=question.id,
                    answer=user_input,
                    timestamp=datetime.now().isoformat()
                )
                session.answers.append(answer)
                engine.apply_answer(answer)
            
            print()
    
    def _agent_chat(self, session: Session):
        """Interactive agent Q&A with SQL capability"""
        
        # Get the final cleaned DataFrame
        df_final = None
        if hasattr(self, '_final_df'):
            df_final = self._final_df
        else:
            # Try to load from latest checkpoint
            if session.checkpoints:
                latest_cp = session.checkpoints[-1]
                try:
                    df_final = self.session_manager.load_checkpoint(latest_cp)
                except:
                    pass
        
        # Initialize or enable SQL for agent
        agent = DataSchemaAgent(session, api_key=self.args.api_key, model=self.args.model)
        
        if df_final is not None:
            agent.enable_sql(df_final)
            logger.info("\n✓ Agent ready with SQL query capability!")
            logger.info("  You can ask questions that require data analysis")
            logger.info("  Examples:")
            logger.info("    - 'What is the average price?'")
            logger.info("    - 'How many items per category?'")
            logger.info("    - 'Show me the top 5 most expensive items'")
        else:
            logger.info("\n✓ Agent ready (schema questions only)")
        
        logger.info("\nAsk me anything! (Type 'exit' to quit)\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ('exit', 'quit', 'q'):
                    logger.info("\nAgent session ended.\n")
                    break
                
                response = agent.chat(user_input)
                print(f"\nAgent: {response}\n")
                
            except KeyboardInterrupt:
                logger.info("\n\nAgent session ended.\n")
                break
            except Exception as e:
                logger.error(f"\nError: {str(e)}\n")
        
        agent.close()

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Data Schema Analysis Pipeline V4"
    )
    parser.add_argument(
        "file_paths",
        nargs='+',
        type=str,
        help="CSV or XLSX file(s)"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--skip-interactive", action="store_true")
    parser.add_argument("--auto-transform", action="store_true")
    parser.add_argument("--auto-clean", action="store_true", help="Auto-apply cleaning rules")
    parser.add_argument("--skip-agent", action="store_true", help="Skip agent Q&A")
    
    args = parser.parse_args()
    
    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY")
    
    if not args.api_key:
        logger.error("Error: OpenAI API key required")
        sys.exit(1)
    
    try:
        cli = CLI(args)
        cli.run()
    except Exception as e:
        logger.error(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()