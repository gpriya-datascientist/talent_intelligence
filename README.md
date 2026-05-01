# talent_intelligence


config.py
What it does step by step:
BaseSettings from pydantic-settings reads every value from your .env file and validates the type automatically. So if ANTHROPIC_API_KEY is missing, the app crashes at startup with a clear error — not silently at runtime when you first make an LLM call. Every setting has a type annotation (str, float, Literal["faiss", "pinecone"]) so wrong values are caught immediately.
The ranking weights (WEIGHT_SKILL_MATCH, WEIGHT_RECENCY etc.) live here deliberately — not hardcoded in the ranking logic. This means you can tune the scoring model by changing one line in .env without touching code.
@lru_cache on get_settings() means the settings object is created exactly once when the app starts and reused everywhere. Without it, Python would re-read and re-parse the .env file on every single function call that needs a config value.
How to talk about it in an interview:
"I used pydantic-settings for config management so all environment variables are type-validated at startup. The app fails fast with a clear error if something is missing rather than failing silently mid-request. I also cached the settings object with lru_cache to avoid re-parsing the env file on every access. Ranking weights are config values not constants — so I can tune the scoring model without a code change or redeployment."

config.py explanation
# Pydantic `config.py` Complete Cheat Sheet

---

# 🧠 MAIN PURPOSE OF THIS FILE

This file creates a **central configuration system** for your app.

Instead of:

```python
import os

db = os.getenv("DATABASE_URL")
debug = os.getenv("DEBUG")
```

You do:

```python
settings.DATABASE_URL
settings.DEBUG
```

with:

* validation ✅
* type conversion ✅
* defaults ✅
* `.env` loading ✅

---

# 🔥 CORE IDEA

```python
class Settings(BaseSettings):
```

means:

> “Create a config schema that loads values from environment variables.”

---

# 🧩 BASIC STRUCTURE

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FIELD_NAME: TYPE = DEFAULT_VALUE

    class Config:
        env_file = ".env"

settings = Settings()
```

---

# 🧠 MOST IMPORTANT CONCEPT

## THIS:

```python
FIELD_NAME: TYPE = DEFAULT_VALUE
```

does NOT execute anything.

It ONLY defines rules/schema.

---

# 🟢 Meaning of each part

Example:

```python
DEBUG: bool = True
```

| Part    | Meaning                |
| ------- | ---------------------- |
| `DEBUG` | field name             |
| `bool`  | expected type          |
| `True`  | default/fallback value |

---

# 🔥 MOST IMPORTANT EXECUTION LINE

THIS line activates Pydantic:

```python
settings = Settings()
```

This is where:

* `.env` is read
* values are converted
* validation happens
* final object is created

---

# 🚀 COMPLETE FLOW OF EXECUTION

---

# STEP 1 — Python reads imports

```python
from pydantic_settings import BaseSettings
from typing import Literal
from functools import lru_cache
```

Only tools are loaded.

No env reading yet.

---

# STEP 2 — Class definition happens

```python
class Settings(BaseSettings):
```

Python creates the class blueprint.

Still:

* ❌ no `.env`
* ❌ no validation
* ❌ no conversion

---

# STEP 3 — Fields are registered

Example:

```python
DEBUG: bool = True
PORT: int = 8000
```

Pydantic stores metadata internally like:

```python
{
    "DEBUG": {
        "type": bool,
        "default": True
    },
    "PORT": {
        "type": int,
        "default": 8000
    }
}
```

---

# STEP 4 — Config class is read

```python
class Config:
    env_file = ".env"
```

Pydantic notes:

> “When object is created, load values from `.env`.”

---

# STEP 5 — Object creation starts

```python
settings = Settings()
```

🔥 THIS triggers everything.

Equivalent idea:

```python
Settings.__init__()
```

---

# STEP 6 — `.env` file is read

Suppose `.env` contains:

```env
DEBUG=false
PORT=9000
```

Pydantic internally reads:

```python
{
    "DEBUG": "false",
    "PORT": "9000"
}
```

⚠️ Environment variables are ALWAYS strings.

---

# STEP 7 — Field-by-field processing begins

For every field:

```python
FIELD: TYPE = DEFAULT
```

Pydantic follows this algorithm:

```python
if env_value_exists:
    value = env_value
elif default_exists:
    value = default
else:
    raise MissingFieldError

converted = convert(value, TYPE)

validate(converted)

store(converted)
```

---

# 🔥 THREE IMPORTANT CASES

---

# 🟢 CASE 1 — Correct env value

## Model

```python
PORT: int = 8000
```

## `.env`

```env
PORT=9000
```

---

## Flow

### 1. Check env

Found:

```python
"9000"
```

---

### 2. Ignore default

Default `8000` is NOT used.

---

### 3. Convert

```python
int("9000")
```

Result:

```python
9000
```

---

### 4. Validate

Valid integer ✅

---

### 5. Store

```python
settings.PORT == 9000
```

---

# 🟢 CASE 2 — Missing env value

## Model

```python
PORT: int = 8000
```

## `.env`

```env
# PORT missing
```

---

## Flow

### 1. Check env

❌ not found

---

### 2. Use default

```python
8000
```

---

### 3. Convert

Already int.

---

### 4. Validate

Valid ✅

---

### 5. Store

```python
settings.PORT == 8000
```

---

# 🔴 CASE 3 — Invalid env value

## Model

```python
PORT: int = 8000
```

## `.env`

```env
PORT=hello
```

---

## Flow

### 1. Check env

Found:

```python
"hello"
```

---

### 2. Ignore default

Default `8000` is NOT used.

---

### 3. Convert

```python
int("hello")
```

❌ fails

---

### 4. ValidationError raised

App crashes with:

```text
Input should be a valid integer
```

---

# 🚨 IMPORTANT RULE

| Situation     | Result        |
| ------------- | ------------- |
| Missing value | use default   |
| Invalid value | ERROR         |
| Correct value | convert + use |

---

# 🟢 REQUIRED FIELD

Example:

```python
API_KEY: str
```

No default exists.

---

# `.env`

```env
# missing API_KEY
```

---

# Flow

### 1. Check env

❌ not found

---

### 2. Check default

❌ no default

---

### 3. Error

```text
Field required
```

---

# 🧠 TYPE CONVERSION CHEAT SHEET

---

# bool conversion

## `.env`

```env
DEBUG=true
```

## Model

```python
DEBUG: bool
```

## Result

```python
True
```

---

# int conversion

## `.env`

```env
PORT=8000
```

## Model

```python
PORT: int
```

## Result

```python
8000
```

---

# float conversion

## `.env`

```env
THRESHOLD=0.75
```

## Model

```python
THRESHOLD: float
```

## Result

```python
0.75
```

---

# Literal validation

## Model

```python
ENV: Literal["development", "production"]
```

---

## Valid

```env
ENV=development
```

✅ works

---

## Invalid

```env
ENV=dev
```

❌ validation error

---

# 🔥 WHAT `Literal` REALLY DOES

```python
ENV: Literal["development", "production"]
```

means:

> “Only these exact values are allowed.”

---

# 🧠 HOW DEFAULTS REALLY WORK

Example:

```python
DEBUG: bool = True
```

means:

```text
Expected type = bool
Fallback value = True
```

NOT:

```text
Always use True
```

Env values override defaults.

---

# 🔥 PRIORITY ORDER

Pydantic checks values in this order:

```text
1. Environment variable
2. Default value
3. Error if neither exists
```

---

# 🧠 IMPORTANT UNDERSTANDING

This line:

```python
DEBUG: bool
```

does NOT convert anything.

It ONLY says:

```text
DEBUG should become bool
```

---

# 🔥 THIS line does actual conversion

```python
settings = Settings()
```

Because object creation triggers:

* env loading
* parsing
* conversion
* validation

---

# 🧩 INTERNAL MENTAL MODEL

Imagine Pydantic internally doing:

```python
raw_env = {
    "DEBUG": "true",
    "PORT": "8000"
}

schema = {
    "DEBUG": bool,
    "PORT": int
}

final = {}

for field in schema:

    value = get_env_or_default(field)

    converted = convert(value)

    validate(converted)

    final[field] = converted
```

---

# 🔥 `Config` CLASS CHEAT SHEET

```python
class Config:
    env_file = ".env"
```

means:

> “Load environment variables from `.env` file.”

---

# 🧠 `@lru_cache()` CHEAT SHEET

```python
@lru_cache()
def get_settings():
    return Settings()
```

means:

> “Create Settings object only once and reuse it.”

Without cache:

* `.env` reloads every time

With cache:

* faster
* singleton-style config

---

# 🚀 FINAL MENTAL MODEL

```text
Class definition
    ↓
Defines rules/schema

Settings()
    ↓
Reads .env

    ↓
Gets strings

    ↓
Chooses env value OR default

    ↓
Converts type

    ↓
Validates

    ↓
Creates final settings object
```

---

# 💥 MOST IMPORTANT THINGS TO REMEMBER

## 1.

```python
FIELD: TYPE = DEFAULT
```

means:

```text
Expected type + fallback value
```

---

## 2.

```python
Settings()
```

is where all magic happens.

---

## 3.

Defaults are used ONLY when value is missing.

---

## 4.

Invalid values NEVER fallback to defaults.

---

## 5.

Environment variables are always strings initially.
🧠 MAIN PURPOSE OF THIS FILE

This file creates:

settings

A single object that:

reads values from .env
converts types
validates values
provides defaults
is reusable everywhere
🔥 CORE IDEA
This line:
class Settings(BaseSettings):

means:

“This class will load configuration from environment variables.”

🧩 BASIC FIELD SYNTAX
FIELD_NAME: TYPE = DEFAULT_VALUE

Example:

DEBUG: bool = True
🧠 Meaning of each part
Part	Meaning
DEBUG	environment variable name
bool	expected type
True	fallback/default value
🔥 IMPORTANT

This line:

DEBUG: bool = True

DOES NOT:

read .env
convert types
validate

It ONLY defines rules/schema.

🚀 THE REAL MAGIC LINE
settings = Settings()

THIS triggers:

reading .env
type conversion
validation
object creation
🧠 COMPLETE EXECUTION FLOW
STEP 1 — Python reads imports
from pydantic_settings import BaseSettings
from typing import Literal
from functools import lru_cache

Only tools are loaded.

Nothing else happens yet.

STEP 2 — Class definition happens
class Settings(BaseSettings):

Python creates blueprint/schema.

Still:

no .env
no validation
no conversion
STEP 3 — Fields are registered

Example:

PORT: int = 8000
DEBUG: bool = True

Pydantic internally stores:

{
    "PORT": {
        "type": int,
        "default": 8000
    },
    "DEBUG": {
        "type": bool,
        "default": True
    }
}
STEP 4 — Config class is read
class Config:
    env_file = ".env"

Means:

“When Settings() runs, load values from .env”

STEP 5 — Function definition
@lru_cache()
def get_settings():
    return Settings()

Still no .env read yet.

STEP 6 — THIS LINE EXECUTES EVERYTHING
settings = get_settings()

which calls:

Settings()

NOW Pydantic engine starts.

🔥 INSIDE Settings()
STEP 7 — Read .env

Suppose .env:

DEBUG=false
PORT=8000

Pydantic reads raw values:

{
    "DEBUG": "false",
    "PORT": "8000"
}

⚠️ Environment variables are ALWAYS STRINGS.

STEP 8 — Decide value source

For EACH field:

FIELD: TYPE = DEFAULT

Pydantic asks:

Does env contain this field?
🟢 CASE 1 — ENV VALUE EXISTS

Model:

PORT: int = 8000

.env:

PORT=9000

Pydantic uses:

"9000"

Default ignored.

🟢 CASE 2 — ENV VALUE MISSING

Model:

PORT: int = 8000

.env:

# PORT missing

Pydantic uses default:

8000
🔥 IMPORTANT RULE
Default is ONLY used when value is missing.

NOT when invalid.

STEP 9 — TYPE CONVERSION

Pydantic compares:

Raw Value	Expected Type
"false"	bool
"8000"	int

Then converts:

"false" → False
"8000" → 8000
🔥 WHERE conversion happens?

During:

Settings()

NOT during field definition.

STEP 10 — VALIDATION

After conversion:

Pydantic checks:

"Is converted value valid?"
🟢 VALID CASE

Model:

PORT: int

Env:

PORT=8000

Conversion:

8000

Validation passes ✅

❌ INVALID CASE

Model:

PORT: int

Env:

PORT=hello

Conversion attempt:

int("hello")

Fails ❌

Pydantic raises:

ValidationError
🔥 IMPORTANT RULE
Invalid values NEVER fallback to default.
STEP 11 — Final object created

After successful conversion + validation:

settings.PORT
settings.DEBUG

contain final typed values.

🧠 FIELD TYPES CHEAT SHEET
String
NAME: str

Env:

NAME=John

Result:

"John"
Integer
PORT: int

Env:

PORT=8000

Result:

8000
Boolean
DEBUG: bool

Env:

DEBUG=true

Result:

True
Float
SCORE: float

Env:

SCORE=0.65

Result:

0.65
Literal
ENV: Literal["development", "production"]

Allowed ONLY:

"development"
"production"

Anything else:

❌ ValidationError

🧠 REQUIRED vs OPTIONAL FIELDS
REQUIRED FIELD
OPENAI_API_KEY: str

No default.

Must exist in .env.

Missing required field
# OPENAI_API_KEY missing

❌ Error:

Field required
OPTIONAL FIELD
GITHUB_TOKEN: str = ""

If missing:

""

used as default.

🧠 PRIORITY ORDER

Pydantic checks values in this order:

1. Environment variable
2. Default value
3. Error if neither exists
🔥 INTERNAL MENTAL MODEL

Pydantic internally behaves LIKE this:

for each field:

    if env_value_exists:
        value = env_value

    elif default_exists:
        value = default_value

    else:
        raise MissingFieldError

    converted = convert(value, expected_type)

    validate(converted)

    store(converted)
🧠 class Config
class Config:
    env_file = ".env"

Means:

“Load variables from .env file”

🧠 @lru_cache()
@lru_cache()

Caches settings object.

Without cache:

Settings()
Settings()

would reload .env every time.

With cache:

loaded once
reused everywhere
🧠 FINAL MENTAL MODEL
FIELD DEFINITION
PORT: int = 8000

means:

Expected type = int
Fallback value = 8000
EXECUTION
settings = Settings()

means:

Read .env
↓
Choose env value OR default
↓
Convert type
↓
Validate
↓
Store final typed value
🚀 MOST IMPORTANT THINGS TO REMEMBER
Concept	Key Idea
Field definition	only defines rules
Settings()	triggers all processing
Missing value	uses default
Invalid value	raises error
.env values	always strings
Pydantic	converts + validates
Literal	restricts allowed values
BaseSettings	reads env automatically
Config.env_file	tells where .env is
lru_cache	avoids reloading
