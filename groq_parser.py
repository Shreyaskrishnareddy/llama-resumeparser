"""
Groq Resume Parser — Llama 3.1 8B
Single-pass, full resume text, no token limits.
"""

import calendar
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
- CRITICAL name splitting rule: The LAST word of the full name is ALWAYS the LastName. The FIRST word is ALWAYS the FirstName. Everything in between is MiddleName. For a 2-word name, there is no MiddleName (null). For a 3-word name, the middle word is MiddleName. LastName must NEVER be null or empty. If the resume does not contain a name, set all name fields to null — NEVER invent or guess a name.
- For CountryCode: extract from the phone number and ALWAYS include the "+" prefix (e.g., "+1" for US, "+91" for India, "+44" for UK). If no country code is visible in the phone number, infer from the resume's location. The value must always start with "+".
- For ExperienceInYears per role: calculate (EndYear - StartYear) + (EndMonth - StartMonth)/12, rounded to 1 decimal. Use "Present" = February 2026. Step by step: convert both dates to months since year 0 (year*12 + month), subtract, divide by 12. For example: Oct 2023 to Feb 2026 = (2026*12+2) - (2023*12+10) = 24314 - 24286 = 28 months = 2.3 years. Feb 2016 to Feb 2026 = 120 months = 10.0 years.
- For RelevantJobTitles: list 3-5 synonymous or closely related job titles for the CurrentJobRole (e.g., "Software Engineer" → ["Software Developer", "SDE", "Application Developer", "Full Stack Developer"]).
- For each skill: SkillExperienceInMonths = sum of months for each role where the skill was explicitly used or mentioned. Only count roles that actually reference or use the skill. LastUsed = the EndDate of the most recent role where it was used. If a skill only appears in a Skills section and not tied to any specific role, use null for SkillExperienceInMonths. Do NOT assign the same number to every skill — each skill should have a different value based on actual usage across roles.
- IMPORTANT: Certifications (PMP, Scrum Master, CCNA, AWS Certified, etc.) are NOT skills — put them ONLY in Certifications. Spoken/human languages (English, Spanish, Arabic, etc.) are NOT skills — put them ONLY in Languages. ONLY extract skills that are explicitly written/named in the resume text. NEVER add skills that do not appear in the resume — if "Python" is not written anywhere in the resume, do not add Python.
- For Languages: ONLY extract spoken/human languages that are explicitly listed in the resume (e.g., "Languages: English, Spanish"). Do NOT confuse programming languages (C, Java, Python, etc.) with spoken languages. If the resume has a "Languages" section that lists programming languages, those go in Skills, NOT Languages. If no spoken languages are mentioned anywhere in the resume, return an empty array []. NEVER assume or add "English" — only include it if the resume explicitly states it.
- For each experience entry, extract ONLY the job title written directly next to that company name. Each position has its own unique title — never copy a title from one position to another.
- For experience Location: extract the city, state, or country where the role was based. Industry descriptors in parentheses like "(Banking)", "(Healthcare)", "(Entertainment / Wireless)" are NOT locations — ignore them. Look for actual geographic names like city and state.
- For EmploymentType: ONLY use a value if the resume explicitly says "Full-time", "Part-time", "Contract", "Internship", "Freelance", or "Temporary" near that role. If the employment type is NOT written in the resume text, you MUST use null. Do NOT assume or default to "Full-time".
- For RelevantSkills per skill: list 1-3 related/synonymous skills from the same technology family (e.g., "Angular 11" → ["Angular", "Angular 12", "AngularJS"]; "Python" → ["Python 3", "CPython"]).
- For Education Type: infer "Full-time", "Part-time", "Online", or "Distance" if possible. Default to "Full-time" if not stated.
- For Certifications: split into name, issuing organization, and year if mentioned (e.g., "AWS Solutions Architect - 2023" → Name: "AWS Solutions Architect", Issuer: "Amazon Web Services", IssuedYear: "2023").
- For Projects: link to the company/experience where the project was done if evident. Extract or infer the candidate's role in the project.
- Keep all string values concise. KeyResponsibilities should be short bullet strings, not paragraphs. Include ALL responsibility bullets from the resume for each role — do not truncate or limit to 5.
- Extract ALL skills from the resume's skills/technologies sections. If the resume has a technical skills table, extract every technology listed there.
- Use JSON null (not the string "null") for any field where information is not available in the resume.
- For CurrentJobRole: use the job title from the MOST RECENT (first listed) experience entry. Do NOT use section headers like "Executive Briefing", "Summary", "Profile", or resume subtitles. The CurrentJobRole must be an actual job title held at a company.
- For OverallSummary.Summary: if no explicit Summary/Objective section exists, generate a 1-2 sentence professional summary from the candidate's experience and skills.
- For Project descriptions: extract what the project does from the resume text. Never leave Description empty if the resume mentions what the project is about.
- For Achievements: extract ONLY bullet points or sentences from the resume that contain a specific number, percentage, dollar amount, or measurable metric AND describe something the candidate accomplished. If the resume has no such quantified accomplishments, return an empty array []. NEVER generate, infer, or reword text that is not explicitly in the resume. Return [] if unsure.

FINAL REMINDERS (MUST FOLLOW):
1. ListOfSkills must NEVER include: spoken language names (English, Arabic, Spanish, Hindi, etc.), certifications (PMP, AWS, CCNA, etc.), or soft skills. Soft skills include but are not limited to: communication, problem solving, leadership, teamwork, multi-tasking, time management, analytical skills, organizational skills, detail-oriented, adaptable, creative, interpersonal skills. If the resume has a "Personal Skills" or "Soft Skills" section, SKIP it entirely — do not add those items to ListOfSkills.
2. Achievements must be MAX 5 items, each containing a specific metric/number from the resume. Regular job responsibilities go in KeyResponsibilities, NOT Achievements. If no quantified achievements exist, return [].
3. Each experience's JobTitle must be copied exactly from the resume text for that specific role — do not merge titles across roles.
4. KeyResponsibilities is HIGH PRIORITY — include ALL bullets from each role. Do not truncate.
5. EmploymentType: only use values explicitly written in the resume (Contract, Full-time, etc.). Use null if not stated.

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
      "EmploymentType": "",
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

        parsed = _post_process(parsed)

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


# --- Post-processing constants ---
MONTH_MAP = {}
for _i in range(1, 13):
    MONTH_MAP[calendar.month_name[_i].lower()] = _i
    MONTH_MAP[calendar.month_abbr[_i].lower()] = _i

_DEFAULT_EMPLOYMENT_TYPES = {"full-time", "full time", "fulltime"}


def _parse_date(date_str):
    """Convert LLM date strings to (year, month) tuple.

    Handles: "July 2021", "07/2021", "2021", "Present", "Jan 2020",
    "04/2015", "current", "till date", etc.
    Returns None on failure.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip().lower()
    if s in ("present", "current", "till date", "now"):
        return (2026, 2)

    # Try "Month Year" — e.g. "July 2021", "Jan 2020"
    m = re.match(r'^([a-z]+)\s+(\d{4})$', s)
    if m:
        month_name, year_str = m.group(1), m.group(2)
        month = MONTH_MAP.get(month_name)
        if month:
            return (int(year_str), month)

    # Try "MM/YYYY" or "MM-YYYY"
    m = re.match(r'^(\d{1,2})[/\-](\d{4})$', s)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    # Try "YYYY-MM" or "YYYY/MM"
    m = re.match(r'^(\d{4})[/\-](\d{1,2})$', s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    # Try bare year "2021"
    m = re.match(r'^(\d{4})$', s)
    if m:
        return (int(m.group(1)), 1)

    return None


def _calc_months(start, end):
    """Return integer months between two (year, month) tuples. Minimum 1."""
    months = (end[0] - start[0]) * 12 + (end[1] - start[1])
    return max(months, 1)


def _fix_experience_years(parsed):
    """Recalculate ExperienceInYears for each experience from StartDate/EndDate."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start = _parse_date(exp.get("StartDate"))
        end = _parse_date(exp.get("EndDate"))
        if start and end:
            months = _calc_months(start, end)
            years = round(months / 12, 1)
            exp["ExperienceInYears"] = str(years)


def _fix_skill_experience(parsed):
    """Recalculate SkillExperienceInMonths by searching experience text for skill mentions."""
    skills = parsed.get("ListOfSkills")
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(skills, list) or not isinstance(experiences, list):
        return

    # Pre-compute experience text blobs and durations
    exp_data = []
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start = _parse_date(exp.get("StartDate"))
        end = _parse_date(exp.get("EndDate"))
        if not start or not end:
            continue
        months = _calc_months(start, end)
        # Combine summary + responsibilities into searchable text
        parts = []
        summary = exp.get("Summary")
        if isinstance(summary, str):
            parts.append(summary)
        resps = exp.get("KeyResponsibilities")
        if isinstance(resps, list):
            for r in resps:
                if isinstance(r, str):
                    parts.append(r)
        text = " ".join(parts).lower()
        end_date_str = exp.get("EndDate", "")
        exp_data.append((text, months, end_date_str))

    for skill in skills:
        if not isinstance(skill, dict):
            continue
        name = skill.get("SkillName")
        if not isinstance(name, str) or len(name) < 2:
            continue
        name_lower = name.lower()
        total_months = 0
        latest_end = None
        for text, months, end_date_str in exp_data:
            if name_lower in text:
                total_months += months
                parsed_end = _parse_date(end_date_str)
                if parsed_end and (latest_end is None or parsed_end > latest_end):
                    latest_end = parsed_end
        if total_months > 0:
            skill["SkillExperienceInMonths"] = total_months
            if latest_end:
                # Format as "Month Year"
                skill["LastUsed"] = f"{calendar.month_name[latest_end[1]]} {latest_end[0]}"
        else:
            skill["SkillExperienceInMonths"] = None


def _fix_name_splitting(parsed):
    """Deterministically split FullName into First/Middle/Last by whitespace."""
    pd = parsed.get("PersonalDetails")
    if not isinstance(pd, dict):
        return
    full_name = pd.get("FullName")
    if not isinstance(full_name, str) or not full_name.strip():
        return
    parts = full_name.strip().split()
    if len(parts) == 1:
        pd["FirstName"] = parts[0].title()
        pd["MiddleName"] = None
        pd["LastName"] = parts[0].title()
    elif len(parts) == 2:
        pd["FirstName"] = parts[0].title()
        pd["MiddleName"] = None
        pd["LastName"] = parts[1].title()
    else:
        pd["FirstName"] = parts[0].title()
        pd["MiddleName"] = " ".join(parts[1:-1]).title()
        pd["LastName"] = parts[-1].title()
    pd["FullName"] = " ".join(p.title() for p in parts)


def _fix_employment_type(parsed):
    """Null out default 'Full-time' employment types that the LLM fabricates."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        emp_type = exp.get("EmploymentType")
        if isinstance(emp_type, str) and emp_type.strip().lower() in _DEFAULT_EMPLOYMENT_TYPES:
            exp["EmploymentType"] = None


def _post_process(parsed):
    """Orchestrator: apply all post-processing fixes to parsed resume data."""
    if not isinstance(parsed, dict):
        return parsed

    applied = []

    try:
        _fix_name_splitting(parsed)
        applied.append("name_splitting")
    except Exception:
        pass

    try:
        _fix_experience_years(parsed)
        applied.append("experience_years")
    except Exception:
        pass

    try:
        _fix_skill_experience(parsed)
        applied.append("skill_experience")
    except Exception:
        pass

    try:
        _fix_employment_type(parsed)
        applied.append("employment_type")
    except Exception:
        pass

    metadata = parsed.get("_metadata")
    if isinstance(metadata, dict):
        metadata["_post_processed"] = applied

    return parsed


def extract_text_from_file(filepath):
    """Extract text from PDF, DOCX, DOC, TXT, or image files."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        return _extract_pdf(filepath)
    elif ext == '.docx':
        return _extract_docx(filepath)
    elif ext == '.doc':
        return _extract_doc(filepath)
    elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp'):
        return _extract_image_ocr(filepath)
    elif ext in ('.txt', '.html', '.htm'):
        with open(filepath, 'r', errors='ignore') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(filepath):
    """Extract text from PDF using PyMuPDF. Falls back to OCR for scanned PDFs."""
    import fitz
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # If very little text extracted, try OCR (scanned PDF)
    if len(text.strip()) < 50:
        try:
            return _extract_image_ocr(filepath)
        except Exception:
            pass

    return text


def _extract_docx(filepath):
    """Extract text from DOCX."""
    import docx2txt
    return docx2txt.process(filepath)


def _extract_doc(filepath):
    """Extract text from legacy DOC using antiword."""
    import subprocess
    try:
        result = subprocess.run(
            ['antiword', filepath],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except FileNotFoundError:
        pass

    # Fallback: try reading as binary with olefile
    try:
        import olefile
        ole = olefile.OleFileIO(filepath)
        if ole.exists('WordDocument'):
            stream = ole.openstream('WordDocument')
            data = stream.read()
            # Extract ASCII text from binary
            text = data.decode('latin-1', errors='ignore')
            # Filter printable characters
            clean = ''.join(c if c.isprintable() or c in '\n\r\t' else ' ' for c in text)
            ole.close()
            if len(clean.strip()) > 50:
                return clean
    except Exception:
        pass

    raise ValueError("Cannot extract text from DOC file. Install antiword: apt-get install antiword")


def _extract_image_ocr(filepath):
    """Extract text from images using Tesseract OCR."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise ValueError("OCR not available. Install: pip install pytesseract Pillow")

    ext = os.path.splitext(filepath)[1].lower()

    # For PDFs, convert pages to images first
    if ext == '.pdf':
        import fitz
        doc = fitz.open(filepath)
        full_text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            from io import BytesIO
            img = Image.open(BytesIO(img_data))
            full_text += pytesseract.image_to_string(img) + "\n"
        doc.close()
        return full_text
    else:
        img = Image.open(filepath)
        return pytesseract.image_to_string(img)


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
