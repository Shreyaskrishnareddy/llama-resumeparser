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


SYSTEM_PROMPT = """You are an expert resume parser built for an Applicant Tracking System. You extract structured data from resumes with perfect accuracy. You must return ONLY valid JSON. No explanations, no markdown fences, no extra text."""

PARSE_PROMPT = """Extract ALL information from the resume below into the exact JSON structure shown. Follow every instruction carefully.

RULES:
- Use null for missing fields, empty arrays [] where no items found.
- Be thorough — capture every skill, every experience, every detail.
- CRITICAL name splitting rule: The LAST word of the full name is ALWAYS the LastName. The FIRST word is ALWAYS the FirstName. Everything in between is MiddleName. Examples: "John Smith" → First="John", Middle=null, Last="Smith". "John Michael Smith" → First="John", Middle="Michael", Last="Smith". "Shreyas Krishnareddy" → First="Shreyas", Middle=null, Last="Krishnareddy". LastName must NEVER be null or empty.
- For CountryCode: extract from the phone number (e.g., "+1" for US, "+91" for India). If no country code is visible, infer from the resume's location.
- For ExperienceInYears per role: calculate the difference between StartDate and EndDate (e.g., "Jan 2021" to "Present" with today being Feb 2026 = "5.2 years"). Use "Present" as Feb 2026 for calculation.
- For RelevantJobTitles: list 3-5 synonymous or closely related job titles for the CurrentJobRole (e.g., "Software Engineer" → ["Software Developer", "SDE", "Application Developer", "Full Stack Developer"]).
- For each skill: SkillExperienceInMonths = total months across all roles where the skill was used. LastUsed = the EndDate of the most recent role where it was used. If a skill appears in the Skills section but not tied to a specific role, estimate from overall experience.
- For RelevantSkills per skill: list 1-3 related/synonymous skills from the same technology family (e.g., "Angular 11" → ["Angular", "Angular 12", "AngularJS"]; "Python" → ["Python 3", "CPython"]).
- For Education Type: infer "Full-time", "Part-time", "Online", or "Distance" if possible. Default to "Full-time" if not stated.
- For Certifications: split into name, issuing organization, and year if mentioned (e.g., "AWS Solutions Architect - 2023" → Name: "AWS Solutions Architect", Issuer: "Amazon Web Services", IssuedYear: "2023").
- For Projects: link to the company/experience where the project was done if evident. Extract or infer the candidate's role in the project.
- Keep all string values concise. KeyResponsibilities should be short bullet strings, not paragraphs.
- Use JSON null (not the string "null") for any field where information is not available in the resume.
- For OverallSummary.Summary: if no explicit Summary/Objective section exists, generate a 1-2 sentence professional summary from the candidate's experience and skills.
- For Project descriptions: extract what the project does from the resume text. Never leave Description empty if the resume mentions what the project is about.
- For Achievements: extract quantified accomplishments from anywhere in the resume (experience bullets, projects, etc.) even if there is no dedicated Achievements section. Look for metrics like percentages, numbers, scale (e.g., "95% accuracy", "10K+ resumes", "reduced costs by 100%").

{
  "PersonalDetails": {
    "FullName": "",
    "FirstName": "",
    "MiddleName": null,
    "LastName": "",
    "EmailID": "",
    "PhoneNumber": "",
    "CountryCode": "",
    "Location": "",
    "LinkedIn": "",
    "GitHub": "",
    "Portfolio": ""
  },
  "OverallSummary": {
    "Summary": "",
    "CurrentJobRole": "",
    "RelevantJobTitles": [],
    "TotalExperience": "",
    "Domain": ""
  },
  "ListOfExperiences": [
    {
      "JobTitle": "",
      "CompanyName": "",
      "Location": "",
      "StartDate": "",
      "EndDate": "",
      "ExperienceInYears": "",
      "Summary": "",
      "KeyResponsibilities": []
    }
  ],
  "ListOfSkills": [
    {
      "SkillName": "",
      "SkillExperienceInMonths": 0,
      "LastUsed": "",
      "RelevantSkills": []
    }
  ],
  "PrimarySkills": [],
  "SecondarySkills": [],
  "ListOfEducation": [
    {
      "Degree": "",
      "TypeOfEducation": "",
      "Field": "",
      "Institution": "",
      "Location": "",
      "YearPassed": "",
      "GPA": ""
    }
  ],
  "Certifications": [
    {
      "CertificationName": "",
      "IssuerName": "",
      "IssuedYear": ""
    }
  ],
  "Projects": [
    {
      "ProjectName": "",
      "Description": "",
      "CompanyWorked": "",
      "RoleInProject": "",
      "Technologies": [],
      "StartDate": "",
      "EndDate": "",
      "Link": ""
    }
  ],
  "Achievements": [],
  "Languages": []
}

RESUME:
---
RESUME_TEXT_HERE
---

Return ONLY the JSON object. No other text."""


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
                "max_tokens": 8192,
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
