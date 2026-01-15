# Security Documentation for nba-game-predictor

This document outlines the security measures implemented in this project.

## Overview

The project handles sports data, machine learning models, and database operations. The following security measures have been implemented to protect against common vulnerabilities.

## Tech Stack Security Considerations

| Technology | Version | Security Notes |
|------------|---------|----------------|
| Python | 3.10+ | Type hints, strict input validation |
| SQLite | 3.x | Local database, parameterized queries |
| pandas | 2.0+ | Data validation before processing |
| NumPy | 1.24+ | Numeric type checking |
| scikit-learn | 1.3+ | Safe model serialization |
| XGBoost | 2.0+ | Trained locally, no external models |
| joblib | 1.3+ | Secure serialization (replaces pickle) |
| nba_api | 1.4+ | No API keys required |

---

## 1. Model Serialization (Pickle Vulnerability)

### Risk
Python's `pickle` module can execute arbitrary code when loading untrusted data. A malicious actor could craft a pickle file that executes harmful code when loaded.

### Mitigation
- **Replaced pickle with joblib**: The project uses `joblib` for model serialization instead of raw pickle. While joblib uses pickle internally, it provides better security practices for scikit-learn models.
  
- **Path Validation**: Models can only be loaded from the trusted `models/` directory. The `_validate_model_path()` function ensures:
  - The path is resolved to an absolute path
  - The path is within the designated `MODEL_DIR`
  - Path traversal attacks (e.g., `../../etc/passwd`) are prevented

- **No Legacy Pickle Support**: Legacy `.pkl` files are detected but NOT loaded. Users are prompted to migrate to the secure `.joblib` format.

```python
# Secure model loading
from src.predictor import load_model, save_model

model = load_model()  # Only loads from models/ directory
save_model(trained_model)  # Saves securely with joblib
```

### Best Practices
- Never load models from untrusted sources
- Always use `save_model()` to save new models
- Keep the `models/` directory secure with proper file permissions

---

## 2. SQL Injection Prevention

### Risk
Dynamic SQL queries constructed with string concatenation can be exploited to execute malicious SQL commands.

### Mitigation

#### Parameterized Queries
All user-provided values are passed as parameters (?) rather than concatenated into SQL strings:

```python
# SAFE: Parameterized query
cursor.execute("SELECT * FROM games WHERE team = ?", (team,))

# UNSAFE: String concatenation (NOT USED)
cursor.execute(f"SELECT * FROM games WHERE team = '{team}'")
```

#### Column Name Whitelisting
Since SQL parameterization doesn't work for column names, we validate all column names against a whitelist:

```python
ALLOWED_GAME_COLUMNS = frozenset([
    'game_id', 'season', 'date', 'regular', 'playoffs', ...
])

def insert_game(game_data):
    # Only allowed columns are used in the query
    valid_columns = _validate_columns(list(game_data.keys()), ALLOWED_GAME_COLUMNS)
    ...
```

#### Team Abbreviation Validation
Team abbreviations are validated before use in queries:

```python
ALLOWED_TEAMS = frozenset(['atl', 'bos', 'bkn', ...])

def get_team_stats(team):
    if not _validate_team(team):
        raise ValueError("Invalid team")
    ...
```

---

## 3. Input Validation

### Date Format Validation
All date inputs are validated to match `YYYY-MM-DD` format:

```python
def _validate_date_format(date_str: str) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(pattern, date_str))
```

### Numeric Input Validation
Numeric inputs (e.g., `vegas_total`, `predicted_total`) are type-checked and converted:

```python
if vegas_total is not None:
    try:
        vegas_total = float(vegas_total)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid vegas_total: {vegas_total}")
```

### String Sanitization
User-provided strings (e.g., `model_version`) are sanitized to allow only safe characters:

```python
model_version = re.sub(r'[^a-zA-Z0-9._-]', '', str(model_version))[:50]
```

---

## 4. API Security

### Rate Limiting
The `nba_api` client includes built-in delays to avoid rate limiting and potential IP bans:

```python
time.sleep(0.6)  # Delay between API calls
```

### No API Keys in Code
The project uses `nba_api` which doesn't require API keys, eliminating the risk of key exposure.

---

## 5. File System Security

### Path Traversal Prevention
All file paths are validated to prevent directory traversal:

```python
def _validate_model_path(model_path) -> bool:
    resolved_path = model_path.resolve()
    return str(resolved_path).startswith(str(MODEL_DIR.resolve()))
```

### Secure Default Paths
Default paths are defined in `config.py` and use `pathlib.Path` for safe path handling:

```python
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "nba.db"
MODEL_DIR = PROJECT_ROOT / "models"
```

---

## 6. Database Security

### Connection Management
Database connections are managed using context managers to ensure proper cleanup:

```python
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()
```

### Local Database
SQLite database is stored locally, not exposed to network access.

---

## Security Checklist

- [x] Replaced pickle with joblib for model loading
- [x] Added model path validation
- [x] Implemented SQL injection prevention via parameterized queries
- [x] Added column name whitelisting
- [x] Added team abbreviation validation
- [x] Added date format validation
- [x] Added numeric input validation
- [x] Added string sanitization
- [x] Added path traversal prevention
- [x] Using secure default paths with pathlib

---

## Dependencies

Install security-related dependencies:

```bash
pip install joblib>=1.2.0
```

---

## Reporting Security Issues

If you discover a security vulnerability, please:
1. Do NOT open a public issue
2. Contact the maintainers directly
3. Allow reasonable time for a fix before disclosure
