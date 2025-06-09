# Three Variable Fitting 809 Data - Linter Configuration and Guidelines

## Overview

This document outlines the linting standards, configuration, and best practices for the Legacy Reimbursement Analysis System codebase.

## Python Linting Standards

### **Primary Tools**
- **flake8**: Style guide enforcement (PEP 8)
- **pylint**: Code quality analysis
- **mypy**: Type checking
- **black**: Code formatting (optional)

### **Configuration Files**

#### **setup.cfg**
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
exclude = venv/, __pycache__/, .git/

[pylint]
max-line-length = 88
disable = C0114, C0115, C0116, R0903, R0913
```

#### **pyproject.toml**
```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

## Code Quality Standards

### **Naming Conventions**
- **Classes**: PascalCase (`AnalysisOrchestrator`, `LinearModel`)
- **Functions/Methods**: snake_case (`run_analysis`, `calculate_metrics`)
- **Variables**: snake_case (`r2_score`, `output_dir`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_CONFIG`, `MAX_ITERATIONS`)
- **Private**: Leading underscore (`_internal_method`, `_temp_data`)

### **Documentation Requirements**
- **Public functions**: Docstrings with parameters, returns, and examples
- **Classes**: Class-level docstrings with purpose and usage
- **Modules**: Module-level docstrings with overview
- **Complex logic**: Inline comments for clarity

### **Type Hints**
```python
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

def analyze_data(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: List[str]
) -> Dict[str, Union[float, np.ndarray]]:
    """Analyze dataset with specified target and features."""
    pass
```

## Current Linting Status

### **Clean Files**
- `src/core/main.py`: ✅ No linting issues
- `src/core/utils.py`: ✅ No linting issues
- `src/analysis/models/linear.py`: ✅ No linting issues
- `src/analysis/models/ensemble.py`: ✅ No linting issues

### **Files with Suppressed Warnings**
- `src/analysis/binned/*.py`: Line length warnings suppressed for data processing
- `src/analysis/analyzers/*.py`: Import order warnings suppressed for compatibility

### **Common Suppressions**
```python
# Line length for data processing
# pylint: disable=line-too-long

# Import order for compatibility
# pylint: disable=wrong-import-order

# Too many arguments for analysis functions
# pylint: disable=too-many-arguments

# Too many local variables for complex analysis
# pylint: disable=too-many-locals
```

## Automated Linting

### **Pre-commit Hooks**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.8
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

### **CI/CD Integration**
```bash
# GitHub Actions workflow
- name: Lint with flake8
  run: |
    pip install flake8
    flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
```

## Linting Commands

### **Manual Linting**
```bash
# Full codebase check
flake8 src/

# Specific file check
flake8 src/core/main.py

# With detailed output
flake8 src/ --show-source --statistics

# Type checking
mypy src/

# Code quality analysis
pylint src/
```

### **Auto-formatting**
```bash
# Format entire codebase
black src/

# Check formatting without changes
black --check src/

# Format specific file
black src/core/main.py
```

## Exception Guidelines

### **When to Suppress Warnings**
1. **Line Length**: Data processing with long pandas operations
2. **Import Order**: Compatibility with existing analysis tools
3. **Complexity**: Mathematical algorithms with inherent complexity
4. **Arguments**: Analysis functions requiring multiple parameters

### **Suppression Format**
```python
# Specific suppression with reason
def complex_analysis(  # pylint: disable=too-many-arguments
    data: pd.DataFrame,
    target: str,
    features: List[str],
    model_type: str,
    output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Complex analysis requiring multiple parameters for flexibility."""
    pass
```

### **File-level Suppressions**
```python
# At top of file for entire module
# pylint: disable=line-too-long,too-many-locals

"""
Module for complex data analysis operations.
Line length and local variable limits relaxed for data processing clarity.
"""
```

## Best Practices

### **Code Organization**
- **Imports**: Standard library, third-party, local imports (separated by blank lines)
- **Functions**: Keep under 50 lines when possible
- **Classes**: Single responsibility principle
- **Files**: Under 500 lines when possible

### **Error Handling**
```python
def safe_analysis(data: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Perform analysis with proper error handling."""
    try:
        if data.empty:
            logger.warning("Empty dataset provided")
            return None
        
        results = perform_analysis(data)
        return results
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return None
```

### **Logging Integration**
```python
import logging

logger = logging.getLogger(__name__)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process data with logging."""
    logger.info(f"Processing {len(data)} records")
    
    # Processing logic
    processed_data = data.copy()
    
    logger.info(f"Processing complete: {len(processed_data)} records")
    return processed_data
```

## IDE Configuration

### **VS Code Settings**
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.linting.flake8Args": ["--max-line-length=88"],
    "python.linting.pylintArgs": ["--max-line-length=88"]
}
```

### **PyCharm Settings**
- **Code Style**: Set line length to 88 characters
- **Inspections**: Enable PEP 8 coding style inspection
- **Type Checking**: Enable mypy integration
- **Auto-formatting**: Configure black as external tool

## Continuous Improvement

### **Regular Reviews**
- **Weekly**: Review new linting warnings
- **Monthly**: Update linting rules based on codebase evolution
- **Quarterly**: Review suppressed warnings for relevance

### **Team Standards**
- **New Code**: Must pass all linting checks
- **Legacy Code**: Gradual improvement with each modification
- **Documentation**: Update linting guidelines with new patterns

### **Metrics Tracking**
- **Warning Count**: Track reduction over time
- **Code Coverage**: Maintain above 80% for new code
- **Complexity**: Monitor cyclomatic complexity trends

## Integration with Analysis System

### **Analysis-Specific Rules**
- **Data Processing**: Allow longer lines for pandas operations
- **Mathematical Functions**: Allow higher complexity for algorithms
- **Configuration**: Allow many arguments for flexible analysis functions
- **Visualization**: Allow longer functions for plot generation

### **Output Generation**
- **Results Files**: Ensure JSON output is properly formatted
- **Logging**: Maintain consistent logging format across all analyzers
- **Error Messages**: Provide clear, actionable error messages

This linting system ensures code quality while accommodating the specific needs of data analysis and mathematical computation workflows. 