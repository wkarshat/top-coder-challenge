# Linter Warning Suppression Guide

This guide explains how to suppress line length and unused import warnings (and other common linter issues) in your Python project.

## Methods Overview

### 1. Per-Line Suppression (Inline Comments)

Add `# noqa` comments to specific lines:

```python
from typing import Dict, Any  # noqa: F401
very_long_line_that_exceeds_79_characters_but_is_necessary_for_functionality = True  # noqa: E501
```

**Specific error codes:**
- `# noqa: F401` - unused import
- `# noqa: E501` - line too long
- `# noqa: F841` - unused variable
- `# noqa` - suppress all warnings on that line

### 2. Project-Wide Configuration Files

#### setup.cfg (Recommended)
```ini
[flake8]
max-line-length = 100
ignore = E501,F401,F841,W503,E203,W291,W292,W293,E128,E129,E226,W504,F811,F541
exclude = 
    .git,
    __pycache__,
    venv,
    .venv,
    build,
    dist,
    *.egg-info,
    .pytest_cache

[pycodestyle]
max-line-length = 100
ignore = E501,W503,E203
```

#### .flake8 (Alternative)
```ini
[flake8]
max-line-length = 100
ignore = E501,F401,F841,W503,E203,W291,W292,W293,E128,E129,E226,W504,F811,F541
exclude = 
    .git,
    __pycache__,
    venv,
    .venv,
    build,
    dist,
    *.egg-info,
    .pytest_cache
per-file-ignores =
    __init__.py:F401
    */migrations/*:E501,F401
    tests/*:F401,F841
```

#### pyproject.toml (Modern Python)
```toml
[tool.ruff]
line-length = 100
target-version = "py38"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = [
    "E501",  # line-too-long
    "F401",  # unused-import
    "F841",  # unused-variable
    "W503",  # line-break-before-binary-operator
    "E203",  # whitespace-before-punctuation
    "W291",  # trailing-whitespace
    "W292",  # no-newline-at-end-of-file
    "W293",  # blank-line-contains-whitespace
    "E128",  # continuation-line-under-indented-for-visual-indent
    "E129",  # visually-indented-line-with-same-indent-as-next-logical-line
    "E226",  # missing-whitespace-around-arithmetic-operator
    "W504",  # line-break-after-binary-operator
    "F811",  # redefined-while-unused
    "F541",  # f-string-missing-placeholders
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py files
```

### 3. VS Code/Editor Configuration

#### .vscode/settings.json
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Args": [
        "--max-line-length=100",
        "--ignore=E501,F401,F841,W503,E203,W291,W292,W293,E128,E129,E226,W504,F811,F541"
    ],
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length=100"
    ],
    "editor.rulers": [100]
}
```

### 4. Command Line Usage

```bash
# Suppress specific warnings
flake8 --max-line-length=100 --ignore=E501,F401,F841 src/

# Use configuration file
flake8 src/  # Uses setup.cfg or .flake8 automatically

# Count remaining issues
flake8 src/ --count

# Show only specific error types
flake8 src/ --select=E501,F401
```

## Common Error Codes

| Code | Description | Suppression Reason |
|------|-------------|-------------------|
| E501 | Line too long | Allow longer lines for readability |
| F401 | Imported but unused | Allow imports for future use or __init__.py |
| F841 | Local variable assigned but never used | Allow temporary variables |
| W503 | Line break before binary operator | Conflicts with Black formatter |
| E203 | Whitespace before ':' | Conflicts with Black formatter |
| W291 | Trailing whitespace | Formatting issue |
| W292 | No newline at end of file | Formatting issue |
| W293 | Blank line contains whitespace | Formatting issue |
| E128 | Continuation line under-indented | Indentation style preference |
| E129 | Visually indented line with same indent | Indentation style preference |
| E226 | Missing whitespace around arithmetic operator | Style preference |
| W504 | Line break after binary operator | Style preference |
| F811 | Redefinition of unused variable | Method overriding |
| F541 | F-string is missing placeholders | Template strings |

## Best Practices

1. **Use project-wide configuration** rather than inline comments when possible
2. **Set line length to 100** instead of default 79 for modern development
3. **Suppress formatting issues** that conflict with Black or other formatters
4. **Allow unused imports in __init__.py** files for package exports
5. **Use per-file ignores** for specific file types (tests, migrations)
6. **Document suppression reasons** in configuration files

## Testing Configuration

After setting up suppression, test with:

```bash
# Should show 0 errors
flake8 src/ --count

# Verify specific files
flake8 src/analysis/models/ensemble.py

# Check what would be caught without suppression
flake8 src/ --ignore=
```

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/lint.yml
- name: Lint with flake8
  run: |
    flake8 src/ --count --show-source --statistics
```

The configuration files will be automatically used by the CI system. 