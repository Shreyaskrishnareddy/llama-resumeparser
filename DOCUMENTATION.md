# Llama Resume Parser — Technical Documentation

> LLM-powered resume parser using Llama 3.1 8B via Groq for structured data extraction.
> **Version**: 1.0.0
> **Last Updated**: 2026-02-28
> **Repository**: [github.com/Shreyaskrishnareddy/llama-resumeparser](https://github.com/Shreyaskrishnareddy/llama-resumeparser)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [API Reference](#4-api-reference)
5. [Data Schema](#5-data-schema)
6. [Post-Processing Pipeline](#6-post-processing-pipeline)
7. [Model Selection & Tradeoffs](#7-model-selection--tradeoffs)
8. [Rate Limits & Constraints](#8-rate-limits--constraints)
9. [Deployment](#9-deployment)
10. [Testing](#10-testing)
11. [Known Limitations](#11-known-limitations)
12. [Configuration Reference](#12-configuration-reference)
13. [Project Structure](#13-project-structure)
14. [Changelog](#14-changelog)

---

## 1. Overview

### What It Does

This application parses resumes (PDF, DOCX, DOC, TXT, images) into structured JSON using Llama 3.1 8B running on Groq's LPU inference engine. It extracts 11 categories of data: personal details, work experience, education, skills (with per-skill experience months), certifications, projects, achievements, and languages.

### Why This Approach

Traditional resume parsers use regex/NLP rules and break on non-standard formatting. LLM-based parsing handles arbitrary resume layouts, bullet styles, and date formats because the model understands context. Groq's hardware inference makes this fast enough for production (~2-7 seconds per resume).

### How It Works

```
Resume File
    |
    v
Text Extraction (PyMuPDF / docx2txt / antiword / Tesseract OCR)
    |
    v
Full text sent to Llama 3.1 8B via Groq API (single-pass, no chunking)
    |
    v
Robust JSON Extraction (handles markdown fences, malformed output)
    |
    v
Post-Processing (fix ExperienceInYears, SkillExperienceInMonths, EmploymentType)
    |
    v
Structured JSON Response + Metadata
```

---

## 2. Architecture

### System Design

```
                    +------------------+
                    |   Web UI (HTML)  |
                    |  Drag & Drop     |
                    +--------+---------+
                             |
                             v
+----------------------------+----------------------------+
|                     Flask API Server (app.py)            |
|                                                          |
|  /parse          Single file upload                      |
|  /parse/text     Raw text input                          |
|  /parse/bulk     Up to 50 files (5 concurrent workers)   |
|  /import/csv     CSV row-by-row parsing                  |
|  /parse/ats/*    ATS-mapped output (Bullhorn/Dice/Ceipal)|
|  /health         Health check                            |
+----------------------------+----------------------------+
                             |
                             v
+----------------------------+----------------------------+
|               groq_parser.py (Core Logic)                |
|                                                          |
|  extract_text_from_file()  Text extraction dispatcher    |
|  parse_resume()            Groq API call + prompt        |
|  _extract_json()           Robust JSON parser            |
|  _post_process()           Deterministic corrections     |
+----------------------------+----------------------------+
                             |
                             v
              +-----------------------------+
              |   Groq API (External)       |
              |   Model: llama-3.1-8b       |
              |   Endpoint: groq.com/openai |
              +-----------------------------+
```

### Request Flow

1. Client uploads file to `/parse`
2. `extract_text_from_file()` converts to raw text using the appropriate library
3. Full text is injected into the prompt template (no truncation, no chunking)
4. Single API call to Groq with the system prompt + user prompt
5. `_extract_json()` extracts JSON from the LLM response (handles edge cases)
6. `_post_process()` applies deterministic corrections to three known LLM error patterns
7. Metadata (model, timing, tokens, post-processing flags) is attached
8. JSON response returned to client

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single-pass (no chunking) | Chunking loses context across sections. Full resume text gives the LLM complete picture. Tradeoff: resumes >16K tokens fail with 413 error. |
| Post-processing over re-prompting | The 8B model consistently makes the same 3 mistakes. Fixing them in Python is faster and more reliable than multi-turn prompting. |
| No database | Stateless API. The parser doesn't store resumes or results. Keeps it simple and avoids PII storage concerns. |
| Gunicorn with 2 workers | Matches Groq's rate limits. More workers would just hit 429 errors. |
| Vanilla JS frontend | No build step, no dependencies. The UI is a single HTML file served by Flask. |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Llama 3.1 8B Instant | Resume text → structured JSON |
| **Inference** | Groq API (LPU) | Fast inference (~2s per resume) |
| **Backend** | Flask 3.1 + Gunicorn | API server |
| **Frontend** | Vanilla HTML/CSS/JS | Drag-and-drop web UI |
| **PDF Parsing** | PyMuPDF (fitz) | PDF text extraction |
| **DOCX Parsing** | docx2txt | DOCX text extraction |
| **DOC Parsing** | antiword + olefile | Legacy .doc support |
| **OCR** | Tesseract + Pillow | Scanned PDFs and images |
| **Deployment** | Docker / Render | Production hosting |

### Dependencies (requirements.txt)

```
Flask==3.1.2
flask-cors==6.0.1
gunicorn==23.0.0
requests==2.32.3
PyMuPDF==1.24.14
docx2txt==0.9
olefile==0.47
pytesseract==0.3.13
Pillow==11.1.0
```

Post-processing uses only stdlib (`calendar`, `re`) — no additional dependencies.

---

## 4. API Reference

### Security Headers

All responses include:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Cache-Control: no-store
```

---

### `GET /health`

Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "groq_configured": true,
  "model": "llama-3.1-8b-instant",
  "supported_formats": ["bmp","doc","docx","htm","html","jpeg","jpg","pdf","png","tiff","txt"],
  "max_bulk_files": 50,
  "timestamp": 1772335017.85
}
```

---

### `POST /parse`

Parse a single uploaded resume file.

**Request:** `multipart/form-data` with key `file`

```bash
curl -X POST http://localhost:8000/parse -F "file=@resume.pdf"
```

**Response (200):**
```json
{
  "filename": "resume.pdf",
  "processing_time_ms": 6824,
  "result": {
    "PersonalDetails": { ... },
    "OverallSummary": { ... },
    "ListOfExperiences": [ ... ],
    "ListOfSkills": [ ... ],
    "PrimarySkills": [ ... ],
    "SecondarySkills": [ ... ],
    "ListOfEducation": [ ... ],
    "Certifications": [ ... ],
    "Projects": [ ... ],
    "Achievements": [ ... ],
    "Languages": [ ... ],
    "_metadata": {
      "parser": "groq",
      "model": "llama-3.1-8b-instant",
      "processing_time_ms": 6675,
      "finish_reason": "stop",
      "prompt_tokens": 4933,
      "completion_tokens": 3494,
      "total_tokens": 8427,
      "_post_processed": ["experience_years", "skill_experience", "employment_type"]
    }
  }
}
```

**Error Responses:**

| Code | Cause |
|------|-------|
| 400 | No file, empty filename, unsupported format |
| 502 | Groq API error (rate limit, model error, timeout) |

**Supported Formats:** PDF, DOC, DOCX, TXT, HTML, HTM, JPG, JPEG, PNG, TIFF, BMP
**Max File Size:** 10 MB

---

### `POST /parse/text`

Parse raw resume text (no file upload).

**Request:** `application/json`

```bash
curl -X POST http://localhost:8000/parse/text \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith\njohn@email.com\n\nEXPERIENCE\nSenior Engineer at Google..."}'
```

**Response:** Same structure as `/parse`, without `filename` wrapper.

**Constraints:** Text must be at least 50 characters.

---

### `POST /parse/bulk`

Parse up to 50 resume files concurrently.

**Request:** `multipart/form-data` with key `files` (multiple)

```bash
curl -X POST http://localhost:8000/parse/bulk \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx"
```

**Response (200):**
```json
{
  "total_files": 2,
  "successful": 2,
  "failed": 0,
  "total_processing_time_ms": 14500,
  "results": [ ... ]
}
```

**Constraints:** Max 50 files, 50 MB total. Uses 5 concurrent workers.

---

### `POST /import/csv`

Import candidate records from CSV. Each row's columns are concatenated as resume text.

```bash
curl -X POST http://localhost:8000/import/csv -F "file=@candidates.csv"
```

**Constraints:** Max 50 rows per import.

---

### `POST /parse/ats/<ats_name>`

Parse resume and return fields mapped to a specific ATS system.

**Supported ATS:** `bullhorn`, `dice`, `ceipal`

```bash
curl -X POST http://localhost:8000/parse/ats/bullhorn -F "file=@resume.pdf"
```

**Response (200):**
```json
{
  "filename": "resume.pdf",
  "ats": "bullhorn",
  "processing_time_ms": 7200,
  "data": {
    "firstName": "Ahmad",
    "lastName": "Elsheikh",
    "email": "ahmad@gmail.com",
    "phone": "+1 312 723 2889",
    "occupation": "Project Manager III",
    "skillList": ["MS Project", "JIRA", "SharePoint"]
  },
  "full_result": { ... }
}
```

**ATS Field Mappings:**

| Bullhorn | Dice | Ceipal | Source Field |
|----------|------|--------|-------------|
| firstName | - | FirstName | PersonalDetails.FirstName |
| lastName | - | LastName | PersonalDetails.LastName |
| email | email_address | Email | PersonalDetails.EmailID |
| phone | phone_number | Phone | PersonalDetails.PhoneNumber |
| occupation | current_title | JobTitle | OverallSummary.CurrentJobRole |
| skillList | skills | Skills | PrimarySkills / ListOfSkills |
| - | work_history | - | ListOfExperiences |
| educationDegree | education | Education | ListOfEducation |

---

## 5. Data Schema

### Full Output Structure

```json
{
  "PersonalDetails": {
    "FullName": "string",
    "FirstName": "string",
    "MiddleName": "string | null",
    "LastName": "string",
    "EmailID": "string | null",
    "PhoneNumber": "string | null",
    "CountryCode": "string (e.g. +1)",
    "Location": "string | null",
    "LinkedIn": "string | null",
    "GitHub": "string | null",
    "Portfolio": "string | null"
  },
  "OverallSummary": {
    "Summary": "string",
    "CurrentJobRole": "string",
    "RelevantJobTitles": ["string"],
    "TotalExperience": "string (e.g. 10 years)",
    "Domain": "string"
  },
  "ListOfExperiences": [
    {
      "JobTitle": "string",
      "CompanyName": "string",
      "Location": "string | null",
      "StartDate": "string (e.g. July 2021)",
      "EndDate": "string (e.g. Present)",
      "EmploymentType": "string (Contract/Part-time/Internship) | null",
      "ExperienceInYears": "string (e.g. 4.6)",
      "Summary": "string",
      "KeyResponsibilities": ["string"]
    }
  ],
  "ListOfSkills": [
    {
      "SkillName": "string",
      "SkillExperienceInMonths": "integer | null",
      "LastUsed": "string (e.g. February 2026) | null",
      "RelevantSkills": ["string"]
    }
  ],
  "PrimarySkills": ["string"],
  "SecondarySkills": ["string"],
  "ListOfEducation": [
    {
      "Degree": "string",
      "TypeOfEducation": "string (Full-time/Part-time/Online/Distance)",
      "Field": "string",
      "Institution": "string",
      "Location": "string | null",
      "YearPassed": "string | null",
      "GPA": "string | null"
    }
  ],
  "Certifications": [
    {
      "CertificationName": "string",
      "IssuerName": "string | null",
      "IssuedYear": "string | null"
    }
  ],
  "Projects": [
    {
      "ProjectName": "string",
      "Description": "string",
      "CompanyWorked": "string | null",
      "RoleInProject": "string | null",
      "Technologies": ["string"],
      "StartDate": "string | null",
      "EndDate": "string | null",
      "Link": "string | null"
    }
  ],
  "Achievements": ["string or object"],
  "Languages": ["string or object"],
  "_metadata": {
    "parser": "groq",
    "model": "string",
    "processing_time_ms": "integer",
    "finish_reason": "string",
    "prompt_tokens": "integer",
    "completion_tokens": "integer",
    "total_tokens": "integer",
    "_post_processed": ["string"]
  }
}
```

### Field Rules (enforced via prompt)

| Rule | Details |
|------|---------|
| Name splitting | Last word = LastName, first word = FirstName, middle = MiddleName |
| CountryCode | Always prefixed with "+" (e.g. "+1", "+91") |
| EmploymentType | Only if explicitly stated in resume. null otherwise. Never defaults to "Full-time". |
| Skills | Only explicitly written skills. No certifications, no spoken languages, no soft skills. |
| Languages | Only spoken/human languages. Programming languages go to Skills. |
| Achievements | Max 5 items, must contain quantified metrics from the resume. Never fabricated. |
| KeyResponsibilities | ALL bullets from resume included. Never truncated. |
| CurrentJobRole | Must be from the most recent experience entry, not a section header. |

---

## 6. Post-Processing Pipeline

### Why Post-Processing Is Needed

The Llama 3.1 8B model produces good structured JSON but consistently makes three types of errors that cannot be fixed through prompt engineering alone:

| Problem | LLM Behavior | Example |
|---------|-------------|---------|
| **ExperienceInYears** | Math errors in date calculations | July 2021 → Present = "5.2" (should be 4.6) |
| **SkillExperienceInMonths** | Fabricates identical values for all skills | All skills get 120 or 180 months |
| **EmploymentType** | Defaults to "Full-time" even when not stated | Every role gets "Full-time" |

### Solution

Deterministic Python post-processing in `_post_process()` runs after every successful LLM parse. Each fix is wrapped in try/except so one failure doesn't block others.

### Fix 1: `_fix_experience_years()`

**What:** Recalculates `ExperienceInYears` from `StartDate` and `EndDate` for every experience entry.

**How:**
1. `_parse_date()` converts LLM date strings to `(year, month)` tuples
   - Handles: "July 2021", "Jan 2020", "04/2015", "2021", "2020-08", "Present", "Current"
   - "Present" = (2026, 2)
   - Year-only dates default to January
2. `_calc_months()` computes months between two tuples (minimum 1)
3. Result is `round(months / 12, 1)` stored as a string like "4.6"

**Verified Results (Ahmad Qasem):**

| Role | StartDate | EndDate | LLM Output | Corrected |
|------|-----------|---------|------------|-----------|
| United Airline | July 2021 | Present | 5.2 | **4.6** |
| Emburse | Jan 2021 | Jun 2021 | 0.8 | **0.4** |
| PepsiCo | Aug 2020 | Dec 2020 | 0.5 | **0.3** |

### Fix 2: `_fix_skill_experience()`

**What:** Recalculates `SkillExperienceInMonths` by searching each experience's text for skill name mentions.

**How:**
1. Pre-computes a list of `(searchable_text, duration_months, end_date)` for each experience entry
   - `searchable_text` = Summary + all KeyResponsibilities joined, lowercased
2. For each skill, does a case-insensitive substring search across all experiences
3. Sums months for matching experiences
4. Skills not found in any experience text get `null`
5. `LastUsed` is updated to the end date of the most recent matching role
6. Skills with names shorter than 2 characters are skipped (avoids false positives)

**Known Limitation:** Substring matching can produce false positives (e.g., "Go" matches "Django"). This is an accepted tradeoff for simplicity. Skills with very short names (1 char) are skipped entirely.

**Verified Results (Mutchie):**

| Skill | LLM Output | Corrected | Reason |
|-------|-----------|-----------|--------|
| Linux | 180 | **205** | Mentioned across many roles |
| Solaris | 180 | **48** | Only in Sun-era roles |
| VDI | 180 | **48** | Only in Oracle/Sun roles |
| Python | 180 | **null** | In skills section only, not in experience text |
| FORTRAN | 120 | **null** | In skills section only |

### Fix 3: `_fix_employment_type()`

**What:** Nulls out "Full-time", "Full Time", and "Fulltime" values that the LLM fabricates when the resume doesn't state employment type.

**How:** Simple string comparison against a set of default values. "Contract", "Part-time", "Internship", "Freelance", "Temporary" are kept as-is.

**Verified Results (Ahmad Qasem):**

| Role | LLM Output | Corrected |
|------|-----------|-----------|
| EtQ | Full Time | **null** |
| United Airline | Contract | **Contract** (kept) |

### Metadata

All post-processing results are tracked in `_metadata._post_processed`:

```json
"_metadata": {
  "_post_processed": ["experience_years", "skill_experience", "employment_type"]
}
```

If a fix fails (bad data), it's silently skipped and omitted from the list.

---

## 7. Model Selection & Tradeoffs

### Models Evaluated

We tested three models on the Groq free tier for resume parsing:

| Model | Params | TPM | RPD | TPD | Speed |
|-------|--------|-----|-----|-----|-------|
| **llama-3.1-8b-instant** | 8B | 6K | 14.4K | 500K | ~2-7s |
| llama-3.3-70b-versatile | 70B | 12K | 1K | 100K | ~5s |
| llama-4-scout-17b-16e-instruct | 17B MoE | 30K | 1K | 500K | ~3s |

### Why We Chose llama-3.1-8b-instant

**We tested `llama-4-scout-17b` against `llama-3.1-8b` on the same resume (Ahmad Qasem).** Results:

| Metric | 8B | Scout 17B | Winner |
|--------|-----|-----------|--------|
| Skills extracted | 20 | 5 | **8B** |
| PrimarySkills populated | Yes | Empty | **8B** |
| Projects extracted | 1 | 0 | **8B** |
| KeyResponsibilities per role | 14, 10, 10, 9, 13 | 13, 9, 9, 8, 11 | **8B** |
| FullName accuracy | "Ahmad Qassem Ahmad Elsheikh" | "Ahmad Elsh eikh" (broken) | **8B** |
| Tokens used | 8,427 | 7,102 | Scout |
| Parse time | 6.8s | 6.4s | Scout |
| EmploymentType hallucination | Yes (fixed by post-processing) | Less frequent | Scout |

**Verdict:** The 8B model extracts significantly more data. Its three weaknesses (ExperienceInYears math, SkillExperienceInMonths fabrication, EmploymentType defaults) are fully corrected by our post-processing pipeline. The Scout model is faster and has higher TPM limits but misses too much data for an ATS use case where completeness matters.

**The 70B model** would give the best quality but has severe rate limits: 1K RPD and 100K TPD, making it impractical for any real workload.

### Tradeoff Summary

```
Chose: Completeness of extraction (8B) + deterministic post-processing
Over:  Smarter model (Scout/70B) with less post-processing needed

Reasoning:
- An ATS parser that misses 75% of skills is worse than one that needs math corrections
- Post-processing is cheap (microseconds, no API calls)
- 8B has 14.4K RPD vs Scout's 1K RPD (14x more daily requests)
```

---

## 8. Rate Limits & Constraints

### Groq Free Tier Limits (llama-3.1-8b-instant)

| Limit | Value | Impact |
|-------|-------|--------|
| **TPM** (tokens/min) | 6,000 | Main bottleneck. A single resume uses ~5K-9K tokens. Effectively 1 resume/minute. |
| **RPM** (requests/min) | 30 | Not a bottleneck in practice. |
| **RPD** (requests/day) | 14,400 | Sufficient for moderate workloads. |
| **TPD** (tokens/day) | 500,000 | ~55-100 resumes/day depending on length. |

### Request Size Limits

| Constraint | Value | Impact |
|-----------|-------|--------|
| Max tokens per request | ~6K prompt | Resumes over ~16K characters hit 413 errors |
| Max file size | 10 MB | Set in Flask config |
| Max bulk files | 50 | Per bulk upload request |
| Bulk total size | 50 MB | All files combined |
| CSV max rows | 50 | Per CSV import |
| API timeout | 60 seconds | Per Groq API call |

### Test Suite Results (22 resumes)

| Category | Count | Percentage |
|----------|-------|-----------|
| Successfully parsed | 7 | 32% |
| 413 Too Large (>16K tokens) | 9 | 41% |
| 429 Rate Limited | 5 | 23% |
| JSON Parse Failure | 2 | 9% |
| Text Extraction Error | 1 | 5% |

**Key Insight:** The 6K TPM limit is the primary constraint. 41% of resumes exceeded the 8B model's context window. Upgrading to Groq's Developer tier ($0) raises TPM to 20K-60K and enables larger resumes.

---

## 9. Deployment

### Render (Current)

The project deploys to Render via `render.yaml`:

```yaml
services:
  - type: web
    name: llama-resumeparser
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GROQ_MODEL
        value: llama-3.1-8b-instant
    healthCheckPath: /health
```

**Steps:**
1. Push code to GitHub
2. Create Web Service on [dashboard.render.com](https://dashboard.render.com)
3. Connect repository, select **Python** runtime
4. Set `GROQ_API_KEY` environment variable
5. Change health check path to `/health`
6. Deploy

**Note:** Free tier instances spin down after inactivity. First request after idle will take 30-60 seconds (cold start). Use Starter ($7/mo) for always-on.

### Docker

```bash
docker build -t llama-resumeparser .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_your_key llama-resumeparser
```

The Dockerfile includes system dependencies (antiword for .doc, tesseract for OCR).

### Local Development

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_your_key
python app.py
# Server starts on http://localhost:8000 with debug mode
```

---

## 10. Testing

### Test Suite (`test_resumes.py`)

The test suite:
1. Iterates through 22 test resumes (PDF, DOCX, DOC)
2. Extracts raw text from each file
3. Sends each file to the `/parse` API endpoint
4. Cross-verifies every extracted field against the raw resume text
5. Generates an Excel report with per-field pass/fail/warn results

**Running Tests:**
```bash
# Ensure the API server is running on port 8000
python app.py &

# Run the test suite (takes ~10 min due to rate limits)
python -u test_resumes.py
```

**Output:** `test-results/Resume_Parser_Test_Report.xlsx` with two sheets:
- **Summary** — per-resume status (PASS/FAIL/WARN/ERROR), check counts, issues
- **Field Details** — every individual field check with extracted value, result, and notes

**Rate Limit Delay:** 25 seconds between API calls (configured in `RATE_LIMIT_DELAY`).

### Manual Verification

Post-processing was manually verified on two resumes:

1. **Ahmad Qasem** — ExperienceInYears corrected (5.2→4.6, 0.8→0.4, 0.5→0.3), EmploymentType "Full Time"→null, Contract values preserved
2. **Mutchie** — All 9 "Full-time" values nulled, SkillExperienceInMonths varies per skill (was all 180), ExperienceInYears recalculated for all 9 roles

---

## 11. Known Limitations

### Model Limitations

| Limitation | Details | Mitigation |
|-----------|---------|------------|
| **Context window** | Resumes >16K chars get 413 errors | Upgrade to Groq Developer tier or use a larger model |
| **JSON reliability** | ~10% of parses return malformed JSON | `_extract_json()` handles markdown fences, brace matching. Retry on failure. |
| **Name parsing** | PDF text extraction sometimes introduces spaces mid-word | None currently — depends on PDF quality |
| **Hallucination** | Model may infer skills or job titles not in the resume | Prompt engineering reduces this but doesn't eliminate it |

### Post-Processing Limitations

| Limitation | Details |
|-----------|---------|
| **Substring matching** | "Go" matches "Django", "C" matches "Cisco". Short skill names cause false positives. |
| **Year-only dates** | "2018" defaults to January 2018. Could be any month. |
| **Present date hardcoded** | "Present" = February 2026. Must be updated over time. |
| **Skills only matched in KeyResponsibilities + Summary** | Skills used in a role but not mentioned in bullets won't be matched. |

### Infrastructure Limitations

| Limitation | Details |
|-----------|---------|
| **Free tier TPM** | 6K tokens/min = ~1 resume per minute |
| **No persistence** | Parsed results are not stored. Client must save the response. |
| **No authentication** | API is open. Add auth middleware for production. |
| **No retry logic** | Rate limit errors (429) are returned directly to the client. |
| **No .doc support without antiword** | Legacy .doc files require antiword system package. |

---

## 12. Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key from [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Model ID to use for parsing |
| `PORT` | No | `8000` | Server port |

### Prompt Configuration

The system prompt and parse prompt are defined as constants in `groq_parser.py`:
- `SYSTEM_PROMPT` — Sets the LLM's role as an expert resume parser
- `PARSE_PROMPT` — Contains all extraction rules + JSON schema template

**LLM Parameters:**
```python
temperature: 0.1    # Low randomness for consistent structured output
top_p: 0.9
max_tokens: 8192    # Max response length
timeout: 60s        # Per-request timeout
```

---

## 13. Project Structure

```
llama-resumeparser/
|
+-- app.py                  # Flask API server (all endpoints, ATS mappings, security headers)
+-- groq_parser.py          # Core logic (text extraction, LLM call, JSON parsing, post-processing)
+-- index.html              # Web UI (drag-drop upload, structured result display)
+-- requirements.txt        # Python dependencies
+-- Dockerfile              # Docker build (Python 3.11 + antiword + tesseract)
+-- render.yaml             # Render deployment config
+-- .env.example            # Environment variable template
+-- .gitignore              # Git ignore rules
+-- LICENSE                 # MIT License
+-- README.md               # Project README
+-- DOCUMENTATION.md        # This file
|
+-- test_resumes.py         # Test suite (22 resumes, cross-verification, Excel report)
+-- test-results/           # Test output (JSON results per resume + Excel report)
```

### File Responsibilities

| File | Lines | Responsibility |
|------|-------|---------------|
| `groq_parser.py` | ~500 | Text extraction, Groq API integration, JSON extraction, post-processing pipeline |
| `app.py` | ~375 | HTTP endpoints, file handling, ATS field mapping, security, CORS |
| `index.html` | ~350 | Web UI with drag-drop, result rendering, JSON toggle |
| `test_resumes.py` | ~590 | Automated cross-verification of parsed fields against raw text |

---

## 14. Changelog

### v1.0.0 (2026-02-28)

**Post-Processing Pipeline**
- Added `_fix_experience_years()` — recalculates ExperienceInYears from StartDate/EndDate
- Added `_fix_skill_experience()` — computes SkillExperienceInMonths from experience text search
- Added `_fix_employment_type()` — nulls fabricated "Full-time" defaults
- Added `_post_process()` orchestrator with per-fix error isolation
- Added `_parse_date()` flexible date parser (7 formats + "Present")
- Added `_metadata._post_processed` tracking array

**Prompt Improvements**
- Added explicit ExperienceInYears calculation instructions with worked examples
- Added EmploymentType rule: null if not stated, never default to "Full-time"
- Added SkillExperienceInMonths rule: per-skill calculation, no identical values
- Added rules for: name splitting, certifications vs skills, spoken vs programming languages
- Added FINAL REMINDERS section enforcing all critical rules

**Infrastructure**
- Fixed `app.py` health endpoint to use `GROQ_MODEL` constant instead of separate hardcoded default
- Model evaluation: tested llama-4-scout-17b vs llama-3.1-8b, kept 8B for extraction completeness

### v0.x (prior)

- Initial Groq resume parser with Llama 3.1 8B
- Flask API with single file, bulk, CSV, and ATS endpoints
- Web UI with drag-and-drop
- Docker + Render deployment support
- DOC/OCR support, security headers
