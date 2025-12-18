import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union,Generator
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
    # Advanced numeric cleaning
    REMOVE_CURRENCY_SYMBOL = "remove_currency_symbol"
    CONVERT_PERCENT_TO_FLOAT = "convert_percent_to_float"
    CONVERT_PARENTHESES_TO_NEGATIVE = "convert_parentheses_to_negative"
    EXTRACT_NUMBER_FROM_STRING = "extract_number_from_string"
    # Advanced datetime cleaning
    CONVERT_EXCEL_SERIAL_DATE = "convert_excel_serial_date"
    PARSE_TEXT_DATE = "parse_text_date"
    # Boolean cleaning
    MAP_TO_BOOLEAN = "map_to_boolean"
    # Null handling
    REPLACE_PLACEHOLDERS_WITH_NAN = "replace_placeholders_with_nan"


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
    data_issues: List[Dict[str, Any]] = Field(default_factory=list)  # Detected data quality issues

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


class Scenario(BaseModel):
    """A scenario defining use case, questions, and expected output format"""
    id: str
    name: str
    description: str = ""
    selected_fields: List[str] = Field(default_factory=list)  # List of column names
    questions: List[str] = Field(default_factory=list)  # Questions for this scenario
    output_format: Dict[str, Any] = Field(default_factory=dict)  # JSON schema for output
    example_input: Optional[Dict[str, Any]] = None  # Example input data
    example_output: Optional[Dict[str, Any]] = None  # Example expected output
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DataPattern(BaseModel):

    """Detected pattern in data"""

    pattern_type: str  # "trend", "seasonality", "categorical", "unique_id", etc.
    column: str
    description: str
    confidence: float
    details: Dict[str, Any] = Field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
class DataAnomaly(BaseModel):
    """Detected anomaly in data"""
    anomaly_type: str  # "outlier", "missing_pattern", "inconsistent_format", etc.
    column: str
    description: str
    severity: str  # "low", "medium", "high"
    affected_rows: int
    examples: List[Any] = Field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
class DataInsights(BaseModel):
    """Complete insights about a dataset"""
    patterns: List[DataPattern] = Field(default_factory=list)
    anomalies: List[DataAnomaly] = Field(default_factory=list)
    distributions: Dict[str, str] = Field(default_factory=dict)  # column -> distribution type
    correlations: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
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

    # Scenarios
    scenarios: List[Scenario] = Field(default_factory=list)

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
            "scenarios": [s.to_dict() for s in self.scenarios],
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
    """Smart type inference that handles formatted numbers and various data issues"""

    # Common null placeholders
    NULL_PLACEHOLDERS = {'N/A', 'n/a', 'NA', 'null', 'NULL', 'None', 'NONE', '-', '?', 'Unknown', 'unknown', '', ' '}

    # Currency symbols
    CURRENCY_SYMBOLS = {'$', '€', '£', '¥', 'USD', 'EUR', 'GBP', 'JPY', 'VND', 'VNĐ', '₫'}

    # Boolean mappings
    BOOLEAN_MAPPINGS = {
        'yes': True, 'no': False,
        'y': True, 'n': False,
        'true': True, 'false': False,
        'có': True, 'không': False,
        'enabled': True, 'disabled': False,
        '1': True, '0': False,
        'on': True, 'off': False
    }

    @staticmethod
    def detect_thousand_separator(series: pd.Series) -> Optional[str]:
        """Detect thousand separator (. or ,)"""
        sample = series.dropna().astype(str).head(100)
        
        # Check for patterns like 1.000, 500.000.000
        dot_pattern = r'^\d{1,3}(\.\d{3})+$'
        comma_pattern = r'^\d{1,3}(,\d{3})+$'
        
        dot_matches = sum(1 for val in sample if re.match(dot_pattern, val.strip()))
        comma_matches = sum(1 for val in sample if re.match(comma_pattern, val.strip()))
        
        if dot_matches > 0:
            return '.'
        if comma_matches > 0:
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
    def detect_data_issues(series: pd.Series) -> List[Dict[str, Any]]:
        """
        Detect various data quality issues in a series.
        Returns list of issues with type and details.
        """
        issues = []
        sample = series.dropna().astype(str).head(100)

        if len(sample) == 0:
            return issues

        # 1. Currency symbols
        currency_pattern = r'[$€£¥₫]|USD|EUR|GBP|JPY|VND|VNĐ'
        has_currency = sample.str.contains(currency_pattern, regex=True, na=False).any()
        if has_currency:
            issues.append({
                'type': 'currency',
                'action': DataCleaningAction.REMOVE_CURRENCY_SYMBOL,
                'description': 'Contains currency symbols',
                'examples': sample[sample.str.contains(currency_pattern, regex=True, na=False)].head(3).tolist()
            })

        # 2. Percentages
        percent_pattern = r'^\s*-?\d+(?:[.,]\d+)?\s*%\s*$'
        has_percent = sample.str.match(percent_pattern, na=False).any()
        if has_percent:
            issues.append({
                'type': 'percentage',
                'action': DataCleaningAction.CONVERT_PERCENT_TO_FLOAT,
                'description': 'Contains percentage values',
                'examples': sample[sample.str.match(percent_pattern, na=False)].head(3).tolist()
            })

        # 3. Accounting negatives (parentheses)
        paren_pattern = r'^\(\s*\d+(?:[.,]\d+)?\s*\)$'
        has_parentheses = sample.str.match(paren_pattern, na=False).any()
        if has_parentheses:
            issues.append({
                'type': 'accounting_negative',
                'action': DataCleaningAction.CONVERT_PARENTHESES_TO_NEGATIVE,
                'description': 'Contains accounting-style negative numbers (parentheses)',
                'examples': sample[sample.str.match(paren_pattern, na=False)].head(3).tolist()
            })

        # 4. Units (numbers with text suffix)
        unit_pattern = r'^\s*\d+(?:[.,]\d+)?\s+[a-zA-Z]+\d*\s*$'
        has_units = sample.str.match(unit_pattern, na=False).any()
        if has_units:
            issues.append({
                'type': 'units_suffix',
                'action': DataCleaningAction.EXTRACT_NUMBER_FROM_STRING,
                'description': 'Contains numbers with unit suffixes (kg, cm, m2, etc.)',
                'examples': sample[sample.str.match(unit_pattern, na=False)].head(3).tolist()
            })


        # 6. Boolean-like values
        unique_lower = set(sample.str.lower().str.strip())
        boolean_keys = set(TypeInferenceEngine.BOOLEAN_MAPPINGS.keys())
        if unique_lower.issubset(boolean_keys | {'nan', 'none'}):
            issues.append({
                'type': 'boolean_text',
                'action': DataCleaningAction.MAP_TO_BOOLEAN,
                'description': 'Contains boolean-like text values',
                'examples': list(unique_lower)[:5]
            })

        # 7. Null placeholders
        null_placeholders_found = unique_lower & TypeInferenceEngine.NULL_PLACEHOLDERS
        if null_placeholders_found:
            issues.append({
                'type': 'null_placeholders',
                'action': DataCleaningAction.REPLACE_PLACEHOLDERS_WITH_NAN,
                'description': 'Contains text representing null values',
                'examples': list(null_placeholders_found)
            })

        return issues

    @staticmethod
    def generate_cleaning_rules(df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> List[CleaningRule]:
        """Generate cleaning rules based on inferred types and detected issues"""
        rules = []

        for col_name, profile in profiles.items():
            # First, generate rules from detected data issues
            for issue in profile.data_issues:
                rule_id = f"clean_{col_name}_{issue['type']}"
                rules.append(CleaningRule(
                    id=rule_id,
                    column=col_name,
                    action=issue['action'],
                    description=f"{col_name}: {issue['description']} (e.g., {issue['examples'][:2]})",
                    params=issue.get('params', {}),
                    applied=False
                ))

            # Then, existing logic for type conversions
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

            # Advanced numeric cleaning
            elif rule.action == DataCleaningAction.REMOVE_CURRENCY_SYMBOL:
                # Remove currency symbols and convert to numeric
                currency_pattern = r'[$€£¥₫]|USD|EUR|GBP|JPY|VND|VNĐ'
                series = df_result[col].astype(str).str.replace(currency_pattern, '', regex=True)
                series = series.str.replace(',', '').str.strip()
                df_result[col] = pd.to_numeric(series, errors='coerce')
                logger.info(f"✓ Removed currency symbols from '{col}'")

            elif rule.action == DataCleaningAction.CONVERT_PERCENT_TO_FLOAT:
                # Remove % and convert to float (optionally divide by 100)
                series = df_result[col].astype(str).str.replace('%', '').str.strip()
                df_result[col] = pd.to_numeric(series, errors='coerce') / 100
                logger.info(f"✓ Converted percentages to floats in '{col}'")

            elif rule.action == DataCleaningAction.CONVERT_PARENTHESES_TO_NEGATIVE:
                # Convert (500) to -500
                def convert_paren(x):
                    s = str(x).strip()
                    if s.startswith('(') and s.endswith(')'):
                        return '-' + s[1:-1]
                    return s
                series = df_result[col].apply(convert_paren)
                df_result[col] = pd.to_numeric(series, errors='coerce')
                logger.info(f"✓ Converted parentheses to negatives in '{col}'")

            elif rule.action == DataCleaningAction.EXTRACT_NUMBER_FROM_STRING:
                # Extract numeric part from strings like "50 kg", "175 cm"
                series = df_result[col].astype(str).str.extract(r'([-+]?\d+(?:[.,]\d+)?)', expand=False)
                df_result[col] = pd.to_numeric(series, errors='coerce')
                logger.info(f"✓ Extracted numbers from strings in '{col}'")

            # Advanced datetime cleaning
            elif rule.action == DataCleaningAction.PARSE_TEXT_DATE:
                # Parse textual dates with flexible format
                df_result[col] = pd.to_datetime(df_result[col], errors='coerce', infer_datetime_format=True)
                logger.info(f"✓ Parsed text dates in '{col}'")

            # Boolean mapping
            elif rule.action == DataCleaningAction.MAP_TO_BOOLEAN:
                # Map various text values to boolean
                df_result[col] = df_result[col].astype(str).str.lower().str.strip().map(
                    TypeInferenceEngine.BOOLEAN_MAPPINGS
                )
                logger.info(f"✓ Mapped to boolean in '{col}'")

            # Null handling
            elif rule.action == DataCleaningAction.REPLACE_PLACEHOLDERS_WITH_NAN:
                # Replace null placeholders with actual NaN
                df_result[col] = df_result[col].replace(list(TypeInferenceEngine.NULL_PLACEHOLDERS), pd.NA)
                logger.info(f"✓ Replaced null placeholders in '{col}'")

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
    def load_source(cls, source: DataSource, header: Optional[int] = 0) -> pd.DataFrame:
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
        """
        Load raw data strictly as strings with NO header assumptions.
        This prevents PyArrow errors with mixed types and allows correct structure analysis.
        """
        try:
            # Luôn đọc tất cả là string để tránh lỗi mixed types của PyArrow
            # Luôn để header=None để LLM nhìn thấy dòng đầu tiên thực tế
            if source.source_type == "csv":
                df = pd.read_csv(source.file_path, header=None, dtype=str)
            else:
                df = pd.read_excel(
                    source.file_path, 
                    sheet_name=source.sheet_name,
                    header=None,
                    dtype=str
                )
            
            # Thay thế NaN bằng chuỗi rỗng để clean hơn khi hiển thị
            df = df.fillna("")
            
            logger.info(f"✓ Loaded RAW {source.source_id}: {len(df)} rows (treated as string grid)")
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load raw source {source.source_id}: {str(e)}")


# ============================================================================
# Structure Analysis (from V3)
# ============================================================================

class StructureAnalyzer:
    """Analyzes raw DataFrame structure using LLM for clean data detection"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def analyze_structure(
        self, 
        df: pd.DataFrame, 
        max_preview_rows: int = 10
    ) -> Tuple[Dict[str, Any], List[Transformation], List[Question], bool]:
        """
        Analyze raw structure using LLM to check if data is clean.
        Returns: (structure_info, transformations, questions, is_clean)
        """
        
        # Extract structure info for LLM analysis
        structure_info = self._extract_structure_info(df, max_preview_rows)
        
        # Use LLM to check if data is clean and propose transformations
        prompt = self._build_structure_prompt(structure_info)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_structure_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            is_clean = result.get("is_clean", False)
            transformations = [Transformation(**t) for t in result.get("transformations", [])]
            questions = [Question(**q) for q in result.get("questions", [])]
            
            if is_clean:
                logger.info("✓ Data structure is clean - no transformations needed")
                return structure_info, [], [], True, []
            else:
                issues = result.get("detected_issues", [])
                logger.info(f"⚠ Detected structure issues: {', '.join(issues)}")
                logger.info(f"✓ Analyzed structure: {len(transformations)} transformations proposed, {len(questions)} questions")
                return structure_info, transformations, questions, False,issues
            
        except Exception as e:
            raise RuntimeError(f"Structure analysis failed: {str(e)}")
    def custom_free_transform(self, user_input: str, 
                              df: pd.DataFrame, 
                              max_preview_rows: int = 10) -> pd.DataFrame:
        """
        Execute custom transformation based on free-form user input.
        Uses LLM to generate and execute Python code for transformations outside predefined cases.
        
        :param user_input: Natural language description of desired transformation
        :param df: Input DataFrame to transform
        :return: Transformed DataFrame
        """
        structure_info = self._extract_structure_info(df, max_preview_rows)
        
        # Use LLM to check if data is clean and propose transformations
        prompt = self._build_custom_prompt(user_input,structure_info)
        try:

            parse_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_structure_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            parsed = json.loads(parse_response.choices[0].message.content)
            transform_type = parsed.get("type")
            
            known_types = ["use_row_as_header", "skip_rows", "drop_columns", "rename_columns", "drop_rows"]
            if transform_type in known_types:
                transformation = Transformation(
                    id=f"custom_{transform_type}",
                    type=TransformationType(transform_type),
                    description=parsed.get("description", user_input),
                    params=parsed.get("params", {}),
                    confidence=0.9
                )
                df_result = self.apply_transformation(df, transformation)
                logger.info(f"✓ Applied structured transformation: {transformation.description}")
                return df_result
            
            # Otherwise, generate custom Python code
            logger.info(f"⚡ Generating custom code for: {user_input}")
            
            # Get DataFrame info for context
            
            code_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_custom_system_prompt()},
                    {"role": "user", "content": f"""DataFrame info:
    {json.dumps(structure_info, indent=2, default=str)}

    User request: {user_input}

    Generate Python code to transform 'df' according to the request.
    Store result in 'df_result'.
    """}
                ],
                temperature=0.3
            )
            
            # Extract and clean the code
            generated_code = code_response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if "```python" in generated_code:
                generated_code = generated_code.split("```python")[1].split("```")[0].strip()
            elif "```" in generated_code:
                generated_code = generated_code.split("```")[1].split("```")[0].strip()
            
            logger.info(f"📝 Generated code:\n{generated_code}")
            
            # Execute the code safely
            local_vars = {"df": df.copy(), "pd": pd, "np": np}
            exec(generated_code, {"__builtins__": __builtins__, "pd": pd, "np": np}, local_vars)
            
            if "df_result" not in local_vars:
                raise RuntimeError("Generated code did not produce 'df_result' variable")
            
            df_result = local_vars["df_result"]
            
            # Validate the result
            if not isinstance(df_result, pd.DataFrame):
                raise RuntimeError(f"Result is not a DataFrame, got {type(df_result)}")
            
            # Apply column validation and fixing
            df_result = self.validate_and_fix_columns(df_result, "[custom_transform] ")
            
            logger.info(f"✓ Custom transformation successful: {df.shape} → {df_result.shape}")
            return df_result
            
        except json.JSONDecodeError as e:
            logger.error(f"✗ Failed to parse LLM response: {str(e)}")
            raise RuntimeError(f"Failed to parse transformation request: {str(e)}")
        except Exception as e:
            logger.error(f"✗ Custom transformation failed: {str(e)}")
            raise RuntimeError(f"Custom transformation failed: {str(e)}")


    def _get_custom_system_prompt(self) -> str:
        """System prompt for generating custom transformation code"""
        return """You are an expert Python Data Scientist specializing in pandas DataFrame transformations.

    Your task is to write clean, efficient Python code to transform a DataFrame named 'df'.
    The transformed result MUST be assigned to a variable named 'df_result'.

    CRITICAL RULES:
    1. ONLY use pandas (pd) and numpy (np) - they are already imported
    2. Work with the existing DataFrame 'df' - do NOT read files
    3. Store the final result in 'df_result'
    4. Write production-ready code with proper error handling
    5. Use .copy() when modifying DataFrames to avoid warnings
    6. RETURN ONLY THE CODE - no markdown, no explanations, no comments outside code

    COMMON OPERATIONS:
    - Filtering: df_result = df[df['column'] > value].copy()
    - Aggregation: df_result = df.groupby('col').agg({'col2': 'sum'}).reset_index()
    - Merging: df_result = pd.merge(df, other_df, on='key')
    - Pivoting: df_result = df.pivot_table(values='val', index='idx', columns='col')
    - String operations: df_result = df.copy(); df_result['col'] = df['col'].str.upper()
    - Date operations: df_result = df.copy(); df_result['date'] = pd.to_datetime(df['date'])
    - Creating columns: df_result = df.copy(); df_result['new'] = df['a'] + df['b']
    - Sorting: df_result = df.sort_values('column').reset_index(drop=True)

    EXAMPLE OUTPUT:
    df_result = df[df['sales'] > 1000].copy()
    df_result = df_result.sort_values('date', ascending=False).reset_index(drop=True)

    Remember: Output ONLY executable Python code, nothing else."""


    def _build_custom_prompt(self, user_input: str,info: Dict[str,Any]) -> str:
        """Build prompt to parse custom transformation request"""
        return f"""Parse this transformation request into a structured format.

    Request: "{user_input}"
    **DataFrame Info:**
        - Shape: {info['shape']['rows']} rows × {info['shape']['columns']} columns
        - Current column names: {info['column_names']}
        - Data types: {json.dumps(info['dtypes'], indent=2)}
        - Null counts: {json.dumps(info['null_counts'], indent=2)}

        **First row values:**
        {json.dumps(info['first_row_values'], indent=2, default=str)}

        **Preview (first few rows):**
        ```json
        {json.dumps(info['preview_data'], indent=2, default=str)}
        ```

        **Last few rows (for context):**
        ```json
        {json.dumps(info.get('last_few_rows', []), indent=2, default=str)}
        ```
    Analyze if this matches any standard transformation pattern:
    - "Use row X as header" → use_row_as_header
    - "Skip first N rows" → skip_rows  
    - "Drop column(s) X" → drop_columns
    - "Rename column A to B" → rename_columns
    - "Drop row(s) X" → drop_rows

    If it matches, provide structured JSON. If it's a custom operation (filtering, aggregating, pivoting, etc.), use type "custom".

    RESPONSE FORMAT:
    {{
    "type": "use_row_as_header" | "skip_rows" | "drop_columns" | "rename_columns" | "drop_rows" | "custom",
    "params": {{
        // For standard types, use appropriate params:
        "row_index": 2,              // for use_row_as_header
        "rows_to_skip": 3,           // for skip_rows
        "columns": ["col1", "col2"], // for drop_columns
        "mapping": {{"old": "new"}},   // for rename_columns
        "indices": [0, 1, 2]         // for drop_rows
        
        // For "custom" type, params can be empty
    }},
    "description": "Human-readable description of what will be done"
    }}

    EXAMPLES:

    Input: "Use row 2 as header"
    Output: {{"type": "use_row_as_header", "params": {{"row_index": 2}}, "description": "Use row 2 as column headers"}}

    Input: "Skip first 5 rows"
    Output: {{"type": "skip_rows", "params": {{"rows_to_skip": 5}}, "description": "Skip first 5 rows"}}

    Input: "Drop columns A, B, C"
    Output: {{"type": "drop_columns", "params": {{"columns": ["A", "B", "C"]}}, "description": "Drop columns A, B, C"}}

    Input: "Rename Sales to Revenue"
    Output: {{"type": "rename_columns", "params": {{"mapping": {{"Sales": "Revenue"}}}}, "description": "Rename Sales to Revenue"}}

    Input: "Filter rows where sales > 1000"
    Output: {{"type": "custom", "params": {{}}, "description": "Filter rows where sales > 1000"}}

    Input: "Calculate average by category"
    Output: {{"type": "custom", "params": {{}}, "description": "Calculate average by category"}}

    Respond with ONLY the JSON object."""

    def _build_structure_prompt(self, info: Dict[str, Any]) -> str:
        """Build prompt for structure analysis"""
        return f"""Analyze this raw DataFrame structure to determine if it's clean or needs transformations.

**DataFrame Info:**
- Shape: {info['shape']['rows']} rows × {info['shape']['columns']} columns
- Current column names: {info['column_names']}
- Data types: {json.dumps(info['dtypes'], indent=2)}
- Null counts: {json.dumps(info['null_counts'], indent=2)}

**First row values:**
{json.dumps(info['first_row_values'], indent=2, default=str)}

**Preview (first few rows):**
```json
{json.dumps(info['preview_data'], indent=2, default=str)}
```

**Last few rows (for context):**
```json
{json.dumps(info.get('last_few_rows', []), indent=2, default=str)}
```

**Instructions:**
1. **First, determine if this data is CLEAN or needs transformations**
   
   A clean dataset has:
   - Proper descriptive column headers (not numeric indices like 0, 1, 2...)
   - Column names are not in the first data row
   - No empty/unnamed columns
   - No duplicate column names
   - Data starts from the correct row (no metadata rows at top)
   - Consistent data structure throughout

2. **If data is CLEAN:**
   - Set "is_clean": true
   - Leave "transformations" and "detected_issues" empty
   - No questions needed

3. **If data needs transformations:**
   - Set "is_clean": false
   - List all "detected_issues" 
   - Propose transformations to fix each issue:
     * Use first row as header
     * Skip empty/metadata rows at the top
     * Drop empty columns
     * Rename poorly named columns
     * Handle merged headers
     * Remove duplicate columns
   
4. **For each transformation, include:**
   - Unique ID (e.g., "use_row_0_as_header")
   - Type of transformation
   - Clear description
   - Required parameters
   - Confidence score (0.0-1.0)

5. **Ask questions only when:**
   - You're uncertain about a transformation
   - Multiple valid approaches exist
   - User input is needed to decide

**Response Format:**
{{
  "is_clean": true/false,
  "detected_issues": ["issue 1", "issue 2", ...],
  "transformations": [
    {{
      "id": "unique_id",
      "type": "use_row_as_header",
      "description": "Human-readable description",
      "params": {{"row_index": 0}},
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

**IMPORTANT:** 
- Be thorough in examining the data structure
- Look at column names, first row, data types, and null patterns
- If everything looks good, confidently mark as clean
- Don't propose unnecessary transformations
"""
    
    def _get_structure_system_prompt(self) -> str:
        """System prompt for structure analysis"""
        return """You are an expert data analyst specializing in detecting structural issues in raw tabular data.

Your job is to:
1. Carefully examine the DataFrame structure by looking at column names, first rows, data types, and patterns
2. Determine if the data is CLEAN (ready to use) or needs TRANSFORMATIONS
3. If transformations are needed, identify all structural problems and propose specific fixes
4. Ask clarifying questions only when genuinely uncertain

**What makes data CLEAN:**
- Descriptive column headers (not 0, 1, 2, or "Unnamed: X")
- Headers are in the column name row, not in the first data row
- No empty or duplicate columns
- Data starts from the appropriate row
- Consistent structure throughout

**Common structural issues:**
- Numeric column indices (0, 1, 2...) instead of headers → first row likely contains real headers
- First row contains text headers while column names are numeric → use first row as header
- Empty rows at top (metadata, titles) → skip those rows
- Unnamed or poorly named columns → rename or drop them
- Merged or multi-level headers → special handling needed

Be precise and confident in your assessment. Always respond in valid JSON format."""
    def _extract_structure_info(self, df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
        """Extract structure information for LLM analysis"""
        preview_rows = min(max_rows, len(df))
        
        return {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "preview_data": df.head(preview_rows).to_dict(orient='records'),
            "null_counts": df.isna().sum().to_dict(),
            "first_row_values": df.iloc[0].tolist() if len(df) > 0 else [],
            "last_few_rows": df.tail(3).to_dict(orient='records') if len(df) > 3 else []
        }

    @staticmethod
    def apply_transformation(df: pd.DataFrame, trans: Transformation) -> pd.DataFrame:
        """Apply a single transformation to DataFrame"""
        df_result = df.copy()
        
        try:
            if trans.type == TransformationType.USE_ROW_AS_HEADER:
                row_idx = trans.params["row_index"]
                df_result.columns = df_result.iloc[row_idx].astype(str).togilist()
                df_result = df_result.iloc[row_idx + 1:].reset_index(drop=True)
                df_result = StructureAnalyzer.validate_and_fix_columns(df_result, f"[{trans.id}] ")
                
            elif trans.type == TransformationType.SKIP_ROWS:
                rows_to_skip = trans.params["rows_to_skip"]
                df_result = df_result.iloc[rows_to_skip:].reset_index(drop=True)
                df_result = StructureAnalyzer.validate_and_fix_columns(df_result, f"[{trans.id}] ")
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
    def validate_and_fix_columns(df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
        """Validate and fix column names after transformation"""
        df_result = df.copy()
        
        # Convert to strings
        df_result.columns = [str(col).strip() if col is not None else '' for col in df_result.columns]
        
        # Fix empty columns
        new_columns = []
        unnamed_counter = 1
        for col in df_result.columns:
            if not col or col == '' or col.lower() == 'nan':
                new_columns.append(f'Unnamed_{unnamed_counter}')
                unnamed_counter += 1
            else:
                new_columns.append(col)
        df_result.columns = new_columns
        
        # Fix duplicates
        if df_result.columns.duplicated().any():
            col_counts = {}
            final_columns = []
            for col in df_result.columns:
                if col not in col_counts:
                    col_counts[col] = 0
                    final_columns.append(col)
                else:
                    col_counts[col] += 1
                    final_columns.append(f"{col}_{col_counts[col]}")
            df_result.columns = final_columns
            logger.info(f"{log_prefix}✓ Fixed duplicate columns")
        
        return df_result
class DataInsightsAnalyzer:
    """Analyze data patterns, anomalies, and distributions using LLM"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze_insights(
        self,
        df: pd.DataFrame,
        profiles: Dict[str, ColumnProfile]
    ) -> DataInsights:
        """
        Analyze data patterns, anomalies, correlations, and distributions.
        This runs AFTER structure is fixed and types are inferred.
        """

        # Build analysis prompt
        prompt = self._build_insights_prompt(df, profiles)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_insights_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            result = json.loads(response.choices[0].message.content)

            # Parse patterns
            patterns = [DataPattern(**p) for p in result.get("patterns", [])]

            # Parse anomalies
            anomalies = [DataAnomaly(**a) for a in result.get("anomalies", [])]

            insights = DataInsights(
                patterns=patterns,
                anomalies=anomalies,
                distributions=result.get("distributions", {}),
                correlations=result.get("correlations", []),
                summary=result.get("summary", "")
            )

            logger.info(f"✓ Data insights: {len(patterns)} patterns, {len(anomalies)} anomalies detected")
            return insights

        except Exception as e:
            raise RuntimeError(f"Insights analysis failed: {str(e)}")

    def _build_insights_prompt(
        self,
        df: pd.DataFrame,
        profiles: Dict[str, ColumnProfile]
    ) -> str:
        """Build prompt for data insights analysis"""

        # Sample data
        sample_rows = df.head(20).to_dict(orient='records')

        # Basic statistics for numeric columns
        numeric_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            numeric_stats[str(col)] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "median": float(df[col].median()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None
            }

        # Categorical columns
        categorical_info = {}
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 50:  # Only for low cardinality
                categorical_info[str(col)] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(10).to_dict()
                }

        return f"""Analyze this dataset and identify patterns, anomalies, and insights.

**Dataset Overview:**
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}

**Column Profiles:**
{json.dumps({col: {"type": profile.inferred_type, "null_ratio": profile.null_ratio, "unique": profile.n_unique} for col, profile in list(profiles.items())[:20]}, indent=2)}

**Numeric Statistics:**
{json.dumps(numeric_stats, indent=2, default=str)}

**Categorical Info:**
{json.dumps(categorical_info, indent=2, default=str)}

**Sample Data (first 20 rows):**
{json.dumps(sample_rows[:10], indent=2, default=str)}

**Instructions:**
Analyze the data and identify:

1. **Patterns** (pattern_type, column, description, confidence, details):
   - Trends (increasing/decreasing over time)
   - Seasonality or cycles
   - Unique identifiers (ID columns)
   - Categorical groupings
   - Date/time patterns
   - Hierarchical relationships

2. **Anomalies** (anomaly_type, column, description, severity, affected_rows, examples):
   - Outliers in numeric data
   - Missing data patterns
   - Inconsistent formats
   - Duplicate entries
   - Data quality issues

3. **Distributions** (column -> distribution type):
   - "normal", "uniform", "skewed_left", "skewed_right", "bimodal", "categorical", etc.

4. **Correlations** (list of related columns):
   - Strong correlations between numeric columns
   - Relationships between categorical and numeric

5. **Summary**: Brief overview of key findings

**Response Format:**
{{
  "patterns": [
    {{
      "pattern_type": "unique_id" | "trend" | "categorical" | "temporal" | ...,
      "column": "column_name",
      "description": "Description of the pattern",
      "confidence": 0.95,
      "details": {{"key": "value"}}
    }}
  ],
  "anomalies": [
    {{
      "anomaly_type": "outlier" | "missing_pattern" | "inconsistent_format" | ...,
      "column": "column_name",
      "description": "Description of the anomaly",
      "severity": "low" | "medium" | "high",
      "affected_rows": 10,
      "examples": ["example1", "example2"]
    }}
  ],
  "distributions": {{
    "column_name": "normal",
    "another_column": "skewed_right"
  }},
  "correlations": [
    {{
      "columns": ["col1", "col2"],
      "type": "positive" | "negative",
      "strength": "strong" | "moderate" | "weak",
      "description": "Description"
    }}
  ],
  "summary": "Brief summary of key insights about this dataset"
}}
"""

    def _get_insights_system_prompt(self) -> str:
        """System prompt for insights analysis"""
        return """You are an expert data scientist specializing in exploratory data analysis.

Your job is to:
1. Identify meaningful patterns in the data
2. Detect anomalies and data quality issues
3. Determine statistical distributions
4. Find correlations and relationships between variables
5. Provide actionable insights

Be thorough but concise. Focus on insights that would help users understand their data better.

Always respond in valid JSON format."""
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
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(
                f"Duplicate column names found: {duplicates}. "
                f"Please apply transformations to fix column names first."
            )
        
        empty_cols = [col for col in df.columns if not str(col).strip()]
        if empty_cols:
            raise ValueError(
                f"Empty column names found. Please rename these columns first."
            )
        
        df = df.copy()
        df.columns = [str(col) for col in df.columns]
        profiles = {}
        
        for col in df.columns:
            series = df[col]
            non_null = series.notna().sum()
            null = series.isna().sum()
            total = len(series)

            try:
                samples = series.dropna().head(max_samples).tolist()
                raw_samples = series.dropna().astype(str).head(max_samples).tolist()
            except:
                logger.info(f'DF: {df}')
            # Type inference
            inferred_type, metadata = TypeInferenceEngine.infer_type(series)

            # Detect data issues
            data_issues = TypeInferenceEngine.detect_data_issues(series)

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
                decimal_separator=metadata.get('decimal_separator'),
                data_issues=data_issues
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
#         return f"""
# **Response Format:**
# {{
#   "questions": [
#     {{
#       "id": "salary_unit",
#       "question": "What is the currency unit for the 'Salary' column?",
#       "suggested_answer": "VND",
#       "target": "Salary.unit",
#       "question_type": "semantic"
#     }},
#     {{
#       "id": "area_unit",
#       "question": "What is the unit of measurement for 'Area'?",
#       "suggested_answer": "m2",
#       "target": "Area.unit",
#       "question_type": "semantic"
#     }}
#   ]
# }}
# Return all NONE and just 1 question
# """
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

Generate necessary clarification questions to has a deep understand about schema.
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

Always respond in valid JSON format. Response in Vietnamese
"""

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

            # Create case-insensitive mapping for column names
            profile_mapping = {col.lower(): col for col in profiles.keys()}

            schema = {}
            for col, spec in result["schema"].items():
                # Try to find matching column in profiles (case-insensitive)
                col_lower = col.lower()
                original_col = profile_mapping.get(col_lower, col)

                # Add original_type from profile if found
                if original_col in profiles:
                    original_type = profiles[original_col].pandas_dtype
                    spec['original_type'] = original_type
                else:
                    # Fallback: try to find best match or use 'object' as default
                    logger.warning(f"Column '{col}' not found in profiles, using default 'object' type")
                    spec['original_type'] = 'object'

                # Use original column name from data
                spec['name'] = original_col if original_col in profiles else col
                schema[original_col if original_col in profiles else col] = ColumnSchema(**spec)
            
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
        """Build validation prompt with robust serialization"""
        
        schema_json = {k: v.to_dict() for k, v in schema.items()} if schema else {}
        
        questions_json = []
        if question_set and question_set.user_questions:
            questions_json = [q.to_dict() for q in question_set.user_questions]
        fields_json = []
        if question_set and question_set.output_fields:
            fields_json = [f.to_dict() for f in question_set.output_fields]
            
        # Lấy Notes
        additional_json = "No additional notes."
        if question_set and question_set.additional_notes:
            additional_json = question_set.additional_notes
            
        logger.info(f'Sending Validation Prompt with Notes: {additional_json}')

        return f"""Validate if the current schema can adequately answer the user's questions.

    **🚨 EXISTING BUSINESS LOGIC & NOTES (CHECK HERE FIRST):**
    ```text
    {additional_json}
    ```
    *(If the answer to a question is found above, DO NOT ask for it again. Mark as sufficient.)*

    **User Questions:**
    {json.dumps(questions_json, indent=2)}

    **Expected Output Fields:**
    {json.dumps(fields_json, indent=2)}

    **Current Schema:**
    {json.dumps(schema_json, indent=2, default=str)}

    **Sample Data:**
    {json.dumps(sample_rows[:5] if sample_rows else [], indent=2, default=str)}

    **Instructions:**
    1. Check if "Existing Business Logic" answers the User Questions.
    2. If a user asks for a field (e.g., "Net Worth") that is NOT in schema BUT is in Business Logic -> It is SUFFICIENT.
    3. Only if truly missing:
       - Ask for the FORMULA or LOGIC.
       - Set target as "Global.logic" or "ColumnName.calculation".

    **Response Format:**
    {{
    "is_sufficient": false,
    "validation_report": "Missing 'Net Worth' column.",
    "additional_questions": [
        {{
        "id": "missing_net_worth",
        "question": "The dataset has 'rawsalary' but no 'Net Worth'. How should Net Worth be calculated?",
        "suggested_answer": "Net Worth = rawsalary - 10%",
        "target": "Global.calculation",
        "question_type": "semantic"
        }}
    ]
    }}
    """

    def _get_validation_system_prompt(self) -> str:
        """System prompt for validation"""
        return """You are an expert data schema validator.

Your task is to determine if a data schema is sufficient to answer specific user questions.

🚨 **CRITICAL RULE - READ THIS FIRST**:
You MUST check the "Additional Notes / Business Logic" section provided in the context.
1. **IF A QUESTION IS ALREADY ANSWERED IN "ADDITIONAL NOTES":**
   - You must consider the schema **SUFFICIENT** for that question.
   - Do **NOT** generate a new clarification question for it.
   - Assume the user knows how to calculate it based on the note.

2. **IF A COLUMN IS MISSING BUT A FORMULA IS IN "ADDITIONAL NOTES":**
   - This counts as SUFFICIENT. Do not ask for the formula again.

Analyze:
1. Can each user question be answered with the current schema OR the provided Business Logic/Notes?
2. Are the expected output fields mappable to schema columns OR calculate-able via Notes?
3. Only generate new questions for information that is **completely missing** from BOTH the Schema AND the Additional Notes.

Be thorough but practical. STOP asking about things that are already defined in the Notes.

Always respond in valid JSON format."""

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
        """Default system prompt for data schema agent with scenarios support"""
        base_prompt = """You are an intelligent data analyst assistant with SQL query capabilities.

🚨 **CRITICAL: SCENARIO-FIRST APPROACH** 🚨
When answering user questions, you MUST follow this priority order:

1. **CHECK FOR SCENARIO MATCH FIRST** (HIGHEST PRIORITY)
   - Look at the user's question
   - Check if it matches any defined scenario pattern in the scenarios section below
   - If YES: Follow that scenario's output template EXACTLY
   - Use SQL to get data values, then format according to the template
   
2. **FREE-FORM QUERY** (Only if no scenario matches)
   - Use SQL queries to analyze data
   - Provide insights and recommendations

**Your Capabilities:**
- Understand and explain data schemas
- Execute SQL queries to analyze data
- Follow predefined output templates for common questions
- Provide insights and recommendations

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

        # Add scenarios instruction if available
        scenarios_instruction = self._format_scenarios_for_prompt()
        if scenarios_instruction:
            base_prompt = f"""{base_prompt}

{scenarios_instruction}

🔴 REMEMBER: Always check for scenario matches FIRST before doing free-form analysis!"""
        
        return base_prompt


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
            scenarios_instruction = self._format_scenarios_for_prompt()
            try:
                interp_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a data analyst. Interpret SQL query results and provide insights to answer the user's question."},
                        {"role": "user", "content": f"**Original Question:** {original_question}\n\n**Query Results:**\n{result_summary}\n\n**Scenarios:**{scenarios_instruction}\n\nPlease interpret these results and answer the question:"}
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
    def _format_scenarios_for_prompt(self) -> str:
        """
        Format scenarios into instructions for the LLM.
        Handles pattern matching and variable templating instructions.
        """
        if not self.session.scenarios:
            return ""

        prompt_parts = ["**🎯 DEFINED SCENARIOS & OUTPUT PATTERNS:**"]
        prompt_parts.append("You MUST follow these patterns when the user's question matches the intent of a scenario.")
        prompt_parts.append("IMPORTANT: These scenarios are EXAMPLES. Apply the same logic/formulas to ANY entity (e.g., if defined for Employee A, apply to Employee B).")

        for idx, sc in enumerate(self.session.scenarios):
            output_desc = sc.output_format.get("description", "") if sc.output_format else ""
            
            scenario_text = f"""
    [Scenario {idx+1}: {sc.name}]
    - Intent/Trigger: Questions like {json.dumps(sc.questions)}
    - Relevant Fields: {', '.join(sc.selected_fields)}
    - Output Template: "{output_desc}"
    """
            prompt_parts.append(scenario_text)

        prompt_parts.append("""
    **Instructions for Scenarios:**
    1. **Pattern Matching**: If the user asks about a different person/item than in the scenario example, use the SAME formula and format.
    2. **Placeholders**: When you see syntax like `{data.ColumnName}` in the Output Template:
       - First, execute SQL to get the value of 'ColumnName' for the requested entity.
       - Then, REPLACE `{data.ColumnName}` with the actual value in the response.
    3. **Calculations**: If the template implies a calculation (e.g., "Salary * 0.9"), perform the calculation using the SQL values.
    """)
        
        return "\n".join(prompt_parts)

    def _build_context(self) -> str:
        """Build context including Business Logic/Notes and Scenarios"""
        parts = []

        # 1. Data Source Info
        if self.session.sources:
            parts.append(f"**Data Source:** {self.session.sources[0].file_path}")

        # 2. Schema Summary
        if self.session.schema:
            schema_lines = []
            for col, col_schema in list(self.session.schema.items())[:30]: # Limit to avoid context overflow
                line = f"- {col}: {col_schema.semantic_type} ({col_schema.physical_type})"
                if col_schema.description:
                    line += f" | Desc: {col_schema.description}"
                schema_lines.append(line)
            
            parts.append("**Schema Structure:**\n" + "\n".join(schema_lines))

        # 3. Business Rules (General)
        if self.session.question_set and self.session.question_set.additional_notes:
            parts.append(f"**⚠️ GLOBAL BUSINESS RULES:**\n{self.session.question_set.additional_notes}")

        # 4. SCENARIOS (New Section)
        scenario_context = self._format_scenarios_for_prompt()
        if scenario_context:
            parts.append(scenario_context)

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
        """Apply user answer to schema with fallback for general notes"""
        try:
            question = next((q for q in self.session.questions if q.id == answer.question_id), None)
            if not question:
                return False
            
            parts = question.target.split(".")
            target_col = parts[0]
            
            if target_col.lower() == "global" or target_col not in self.session.schema:
                if self.session.question_set:
                    timestamp = datetime.now().strftime("%H:%M")

                    new_logic = f"\n- [Business Logic] {question.question} -> Rule: {answer.answer}"
                    self.session.question_set.additional_notes += new_logic
                    
                    logger.info(f"✓ Added business logic: {answer.answer}")
                    answer.applied = True
                    return True
            
            if target_col in self.session.schema:
                schema_col = self.session.schema[target_col]
                field = parts[1] if len(parts) > 1 else "description"
                
                if field == "unit":
                    schema_col.unit = answer.answer
                elif field in ["description", "calculation", "formula"]:
                    # Nếu là calculation, cộng dồn vào description
                    schema_col.description += f" | Calculation: {answer.answer}"
                elif field == "semantic_type":
                    schema_col.semantic_type = answer.answer
                elif field == "physical_type":
                    schema_col.physical_type = answer.answer
                elif field == "is_required":
                    schema_col.is_required = answer.answer.lower() in ('true', 'yes', '1')
                else:
                    # Fallback cho trường hợp field lạ cũng đưa vào description
                    schema_col.description += f" ({field}: {answer.answer})"
                
                answer.applied = True
                
                # Update history... (giữ nguyên code cũ)
                self.session.history.append(HistoryEntry(
                    timestamp=datetime.now().isoformat(),
                    action="question_answered",
                    details={
                        "question_id": answer.question_id,
                        "answer": answer.answer,
                        "target": question.target
                    },
                    schema_version=self.session.schema_version
                ))
                
                self.session.schema_version += 1
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

        df_save = df.copy()
        df_save.columns = df_save.columns.astype(str)
        if not df_save.columns.is_unique:
            new_columns = []
            seen = {}
            for col in df_save.columns:
                if col in seen:
                    seen[col] += 1
                    new_columns.append(f"{col}.{seen[col]}")
                else:
                    seen[col] = 0
                    new_columns.append(col)
            df_save.columns = new_columns
        for col in df_save.columns:
            if df_save[col].dtype == 'object':
                df_save[col] = df_save[col].astype(str)
        try:
            df.to_parquet(checkpoint_file, index=False)
        except Exception as e:
            logger.warning(f"Parquet save failed: {e}. Falling back to CSV")
            checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.csv"
            df.to_csv(checkpoint_file,index=False)
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
        path = Path(checkpoint.file_path)
        
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {path.suffix}")
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
        
        logger.info("\nAnalyzing structure...")
        analyzer = StructureAnalyzer(api_key=self.args.api_key, model=self.args.model)
        
        structure_info, transformations, structural_qs, is_clean = analyzer.analyze_structure(df_raw)
        
        session.raw_structure_info[source.source_id] = structure_info
        session.is_clean_structure = is_clean
        
        df_clean = df_raw.copy()
        applied_trans = []
        
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