"""
Groq Resume Parser — Llama 3.1 8B
Single-pass, full resume text, no token limits.
"""

import json
import os
import re
import time
import requests


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


SYSTEM_PROMPT = """You are an expert resume parser. You extract structured data from resumes with perfect accuracy. Return ONLY valid JSON. No explanations, no markdown fences, no extra text."""

PARSE_PROMPT = """Extract ALL information from this resume into the following JSON structure. Use null for missing fields, empty arrays [] where no items found. Be thorough — capture every skill, every experience, every detail.

{
  "PersonalDetails": {
    "FullName": "",
    "Email": "",
    "Phone": "",
    "Location": "",
    "LinkedIn": "",
    "GitHub": "",
    "Portfolio": ""
  },
  "Summary": "",
  "CurrentJobRole": "",
  "TotalExperience": "",
  "ListOfExperiences": [
    {
      "Company": "",
      "Title": "",
      "StartDate": "",
      "EndDate": "",
      "Location": "",
      "Responsibilities": []
    }
  ],
  "ListOfEducation": [
    {
      "Institution": "",
      "Degree": "",
      "Field": "",
      "GraduationDate": "",
      "GPA": ""
    }
  ],
  "ListOfSkills": [],
  "PrimarySkills": ["top 8-10 core languages/frameworks"],
  "SecondarySkills": ["supporting tools, methodologies, soft skills"],
  "Domain": "",
  "Certifications": [],
  "Achievements": [],
  "Projects": [
    {
      "Name": "",
      "Description": "",
      "Technologies": [],
      "Link": ""
    }
  ],
  "Languages": []
}

RESUME:
---
RESUME_TEXT_HERE
---

Return ONLY the JSON object."""


def is_groq_configured():
    """Check if Groq API key is set."""
    return bool(GROQ_API_KEY)


def parse_resume(resume_text, model=None, api_key=None):
    """
    Parse resume text using Groq API.

    Args:
        resume_text: Full raw text from resume (no truncation)
        model: Override model name
        api_key: Override API key

    Returns:
        dict with parsed fields + _metadata
    """
    key = api_key or GROQ_API_KEY
    mdl = model or GROQ_MODEL

    if not key:
        return {"error": "GROQ_API_KEY not set. Set it as environment variable or pass api_key parameter."}

    prompt = PARSE_PROMPT.replace("RESUME_TEXT_HERE", resume_text)
    start = time.time()

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": mdl,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "top_p": 0.9,
            },
            timeout=60,
        )

        elapsed_ms = int((time.time() - start) * 1000)

        if resp.status_code != 200:
            return {
                "error": f"Groq API error {resp.status_code}: {resp.text[:300]}",
                "processing_time_ms": elapsed_ms,
            }

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        finish = data["choices"][0].get("finish_reason", "unknown")

        parsed = _extract_json(content)

        if parsed is None:
            return {
                "error": "Failed to parse JSON from model response",
                "raw_response": content[:500],
                "finish_reason": finish,
                "processing_time_ms": elapsed_ms,
            }

        parsed["_metadata"] = {
            "parser": "groq",
            "model": mdl,
            "processing_time_ms": elapsed_ms,
            "finish_reason": finish,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

        return parsed

    except requests.exceptions.Timeout:
        return {"error": "Groq API timed out after 60s", "processing_time_ms": int((time.time() - start) * 1000)}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to Groq API. Check your internet connection."}
    except Exception as e:
        return {"error": str(e), "processing_time_ms": int((time.time() - start) * 1000)}


def _extract_json(text):
    """Robust JSON extraction from LLM response."""
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Markdown code block
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Brace matching — find outermost { }
    start = text.find('{')
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == '\\' and in_str:
                esc = True
                continue
            if c == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def extract_text_from_file(filepath):
    """Extract text from PDF, DOCX, DOC, or TXT files."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        return _extract_pdf(filepath)
    elif ext == '.docx':
        return _extract_docx(filepath)
    elif ext == '.txt':
        with open(filepath, 'r', errors='ignore') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(filepath):
    """Extract text from PDF using PyMuPDF."""
    import fitz
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def _extract_docx(filepath):
    """Extract text from DOCX."""
    import docx2txt
    return docx2txt.process(filepath)


if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("Set GROQ_API_KEY first:")
        print("  export GROQ_API_KEY=gsk_...")
        exit(1)

    test = """
John Smith | john@email.com | (555) 123-4567 | San Francisco, CA
LinkedIn: linkedin.com/in/johnsmith

SUMMARY
Senior Software Engineer with 8 years of experience in full-stack development.

EXPERIENCE
Senior Software Engineer | Google | Jan 2021 - Present
- Led payment system migration to microservices
- Reduced API latency by 40%

Software Engineer | Meta | Jun 2018 - Dec 2020
- Built notification system handling 1M+ events/day

EDUCATION
B.S. Computer Science | Stanford University | 2016 | GPA: 3.8

SKILLS
Python, Java, Go, Docker, Kubernetes, AWS

CERTIFICATIONS
AWS Solutions Architect Professional
"""
    result = parse_resume(test)
    print(json.dumps(result, indent=2))
