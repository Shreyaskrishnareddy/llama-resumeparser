# Llama Resume Parser

LLM-powered resume parser using **Llama 3.1 8B** via [Groq](https://groq.com) for ultra-fast inference. Parses resumes into structured JSON in ~2 seconds with high accuracy.

## Features

- **Single-pass full parsing** — no chunking, no token limits, reads the entire resume
- **Structured JSON output** — personal details, experiences, education, skills, certifications, projects, achievements
- **~2 second latency** — Groq's LPU inference engine makes LLM parsing practical for production
- **File upload + raw text APIs** — supports PDF, DOCX, and TXT
- **Web UI included** — drag-and-drop interface with structured result display
- **Docker-ready** — one command to deploy anywhere

## Extracted Fields (BRD-Compliant)

| Category | Fields |
|----------|--------|
| Personal Details | FullName, FirstName, MiddleName, LastName, EmailID, PhoneNumber, CountryCode, Location, LinkedIn, GitHub, Portfolio |
| Overall Summary | Summary, CurrentJobRole, RelevantJobTitles (synonyms), TotalExperience, Domain |
| Experience | JobTitle, CompanyName, Location, StartDate, EndDate, ExperienceInYears, Summary, KeyResponsibilities |
| Skills | SkillName, SkillExperienceInMonths, LastUsed, RelevantSkills (synonyms), PrimarySkills, SecondarySkills |
| Education | Degree, TypeOfEducation, Field, Institution, Location, YearPassed, GPA |
| Certifications | CertificationName, IssuerName, IssuedYear |
| Projects | ProjectName, Description, CompanyWorked, RoleInProject, Technologies, StartDate, EndDate, Link |
| Achievements | Quantified accomplishments extracted from entire resume |
| Languages | Language names |

## Quick Start

### Prerequisites

- Python 3.9+
- [Groq API key](https://console.groq.com/keys) (free tier available)

### Setup

```bash
git clone https://github.com/Shreyaskrishnareddy/llama-resumeparser.git
cd llama-resumeparser

pip install -r requirements.txt

export GROQ_API_KEY=gsk_your_key_here

python app.py
```

Open http://localhost:8000 in your browser.

### Docker

```bash
docker build -t llama-resumeparser .

docker run -p 8000:8000 -e GROQ_API_KEY=gsk_your_key_here llama-resumeparser
```

## API Endpoints

### `POST /parse` — Parse a resume file

Upload a PDF, DOCX, or TXT file.

```bash
curl -X POST http://localhost:8000/parse \
  -F "file=@resume.pdf"
```

**Response:**

```json
{
  "filename": "resume.pdf",
  "processing_time_ms": 2100,
  "result": {
    "PersonalDetails": {
      "FullName": "John Smith",
      "Email": "john@email.com",
      "Phone": "(555) 123-4567",
      "Location": "San Francisco, CA",
      "LinkedIn": "linkedin.com/in/johnsmith",
      "GitHub": null,
      "Portfolio": null
    },
    "Summary": "Senior Software Engineer with 8 years of experience...",
    "CurrentJobRole": "Senior Software Engineer",
    "TotalExperience": "8 years",
    "ListOfExperiences": [
      {
        "Company": "Google",
        "Title": "Senior Software Engineer",
        "StartDate": "Jan 2021",
        "EndDate": "Present",
        "Location": "Mountain View, CA",
        "Responsibilities": [
          "Led payment system migration to microservices",
          "Reduced API latency by 40%"
        ]
      }
    ],
    "ListOfEducation": [...],
    "ListOfSkills": ["Python", "Java", "Go", "Docker", "Kubernetes", "AWS"],
    "PrimarySkills": ["Python", "Java", "Go"],
    "SecondarySkills": ["Docker", "Kubernetes", "AWS"],
    "Certifications": ["AWS Solutions Architect Professional"],
    "Projects": [...],
    "Achievements": [...],
    "_metadata": {
      "parser": "groq",
      "model": "llama-3.1-8b-instant",
      "processing_time_ms": 2050,
      "finish_reason": "stop",
      "prompt_tokens": 850,
      "completion_tokens": 1200,
      "total_tokens": 2050
    }
  }
}
```

### `POST /parse/text` — Parse raw text

Send resume text directly as JSON.

```bash
curl -X POST http://localhost:8000/parse/text \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith\njohn@email.com\n\nEXPERIENCE\n..."}'
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "groq_configured": true,
  "model": "llama-3.1-8b-instant"
}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `PORT` | `8000` | Server port |

### Supported Models

Any model available on [Groq](https://console.groq.com/docs/models) works. Tested with:

| Model | Speed | Accuracy |
|-------|-------|----------|
| `llama-3.1-8b-instant` | ~2s | High |
| `llama-3.3-70b-versatile` | ~5s | Very High |
| `meta-llama/llama-4-scout-17b-16e-instruct` | ~3s | Very High |

## Project Structure

```
llama-resumeparser/
  app.py              # Flask API server
  groq_parser.py      # Core parser — Groq API + JSON extraction
  index.html          # Web UI (drag-and-drop upload)
  requirements.txt    # Python dependencies
  Dockerfile          # Production container
  .env.example        # Environment variable template
```

## How It Works

1. **Text extraction** — PyMuPDF (PDF) or docx2txt (DOCX) extracts raw text from the uploaded file
2. **LLM parsing** — Full resume text is sent to Llama 3.1 via Groq with a structured JSON schema prompt
3. **JSON extraction** — Response is parsed with a robust extractor that handles markdown fences, whitespace, and malformed output
4. **Structured response** — Clean JSON with all fields + metadata (timing, token usage, model info)

## Deployment

### Render

1. Fork this repo
2. Create a new **Web Service** on [Render](https://render.com)
3. Set environment variable: `GROQ_API_KEY`
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

### Railway / Fly.io / Any Docker Host

```bash
docker build -t llama-resumeparser .
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... llama-resumeparser
```

## License

MIT
