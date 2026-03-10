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
- For Languages: ONLY extract spoken/human languages if the resume has an explicit "Languages" section that lists them (e.g., "Languages: English, Spanish"). Do NOT confuse programming languages (C, Java, Python, etc.) with spoken languages. If the resume does NOT have a dedicated "Languages" section listing spoken languages, return an empty array []. NEVER assume or add "English" just because the resume is written in English — only include it if the word "English" appears in a Languages section. When in doubt, return [].
- For each experience entry, extract ONLY the job title written directly next to that company name. Each position has its own unique title — never copy a title from one position to another.
- For experience Location: extract the city, state, or country where the role was based. Industry descriptors in parentheses like "(Banking)", "(Healthcare)", "(Entertainment / Wireless)" are NOT locations — ignore them. Look for actual geographic names like city and state.
- For EmploymentType: ONLY use a value if the resume explicitly says "Full-time", "Part-time", "Contract", "Internship", "Freelance", or "Temporary" near that role. If the employment type is NOT written in the resume text, you MUST use null. Do NOT assume or default to "Full-time".
- For RelevantSkills per skill: list 1-3 related/synonymous skills from the same technology family (e.g., "Angular 11" → ["Angular", "Angular 12", "AngularJS"]; "Python" → ["Python 3", "CPython"]).
- For Education Type: infer "Full-time", "Part-time", "Online", or "Distance" if possible. Default to "Full-time" if not stated.
- For Certifications: split into name, issuing organization, and year if mentioned (e.g., "AWS Solutions Architect - 2023" → Name: "AWS Solutions Architect", Issuer: "Amazon Web Services", IssuedYear: "2023").
- For Projects: link to the company/experience where the project was done if evident. Extract or infer the candidate's role in the project.
- For each experience entry's Summary: extract the brief description or overview paragraph that appears before the bullet points for that role. This is typically 1-2 sentences describing the overall scope or purpose of the role. If no such paragraph exists, use null.
- Keep all string values concise. KeyResponsibilities should be short bullet strings (max 1 sentence each), not paragraphs. Include up to 10 of the most important responsibility bullets per role. If a role has more than 10 bullets, pick the 10 most relevant ones.
- Extract ALL skills from the resume's skills/technologies sections. If the resume has a technical skills table, extract every technology listed there.
- Use JSON null (not the string "null") for any field where information is not available in the resume.
- For CurrentJobRole: use the job title from the MOST RECENT (first listed) experience entry. Do NOT use section headers like "Executive Briefing", "Summary", "Profile", or resume subtitles. The CurrentJobRole must be an actual job title held at a company.
- For OverallSummary.Summary: if the resume has an explicit Summary, Objective, Profile, or Qualification Summary section, copy the ENTIRE text of that section verbatim — do NOT truncate or shorten it. Include ALL sentences. If no such section exists, generate a 2-3 sentence professional summary from the candidate's experience and skills.
- For Project descriptions: extract what the project does from the resume text. Never leave Description empty if the resume mentions what the project is about.
- For Achievements: extract ONLY bullet points or sentences from the resume that contain a specific number, percentage, dollar amount, or measurable metric AND describe something the candidate accomplished. If the resume has no such quantified accomplishments, return an empty array []. NEVER generate, infer, or reword text that is not explicitly in the resume. Return [] if unsure.

FINAL REMINDERS (MUST FOLLOW):
1. ListOfSkills must NEVER include ANY of the following — remove them if present:
   - Certifications: PMP, Scrum Master, CSM, CCNA, ITIL, CISSP, AWS Certified, Azure Certified, SAFe, Six Sigma, CISM, CISA, or any other certification. These go ONLY in Certifications.
   - Spoken languages: English, Spanish, Arabic, Hindi, Telugu, French, German, etc. These go ONLY in Languages.
   - Soft skills: communication, problem solving, leadership, teamwork, multi-tasking, time management, analytical skills, organizational skills, detail-oriented, adaptable, creative, interpersonal skills, process improvement, organizational efficacy.
   If the resume has a "Personal Skills", "Soft Skills", or "Management Skills" section, SKIP it entirely.
2. Achievements must be MAX 5 items, each containing a specific metric/number from the resume. Regular job responsibilities go in KeyResponsibilities, NOT Achievements. If no quantified achievements exist, return [].
3. Each experience's JobTitle must be copied exactly from the resume text for that specific role — do not merge titles across roles.
4. KeyResponsibilities — include up to 5 key bullets per role. Keep each bullet to 1 concise sentence (max 20 words).
5. EmploymentType: only use values explicitly written in the resume (Contract, Full-time, etc.). Use null if not stated.
6. CRITICAL: Your entire response must be valid JSON only — no markdown, no explanations, no text before or after the JSON. Do NOT use smart/curly quotes (use straight quotes only). Keep your total response CONCISE — MAX 25 skills, MAX 5 responsibilities per role, MAX 3 RelevantSkills per skill. Aim for under 6000 tokens total.

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
                "max_completion_tokens": 32768,
            },
            timeout=120,
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

        # Remove hallucinated skills not found in resume text
        try:
            _fix_skill_hallucination(parsed, resume_text)
        except Exception:
            pass

        # Deduplicate skills and split into Primary/Secondary (after hallucination filter)
        try:
            _fix_skill_dedup_and_cap(parsed)
        except Exception:
            pass

        # Enrich with taxonomy normalization (canonical IDs, categories, etc.)
        try:
            from groq_taxonomy import enrich_resume
            parsed = enrich_resume(parsed)
        except ImportError:
            pass

        return parsed

    except requests.exceptions.Timeout:
        return {"error": "Groq API timed out after 120s", "processing_time_ms": int((time.time() - start) * 1000)}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to Groq API. Check your internet connection."}
    except Exception as e:
        return {"error": str(e), "processing_time_ms": int((time.time() - start) * 1000)}


def _sanitize_json_text(text):
    """Sanitize Unicode characters that break JSON parsing.

    Handles smart quotes, em/en dashes, and other common Unicode
    that LLMs copy from resume text into JSON string values.
    """
    # First pass: simple replacements that don't need string-state tracking
    text = text.replace('\u2018', "'").replace('\u2019', "'")   # smart single quotes
    text = text.replace('\u2013', '-').replace('\u2014', '-')   # en/em dashes
    text = text.replace('\u2026', '...')                         # ellipsis

    # Handle smart double quotes and unescaped straight quotes inside JSON strings.
    # Smart quotes (\u201c, \u201d) and bare " inside string values break JSON.
    # We track JSON string state to escape any problematic quotes.
    result = []
    in_json_str = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_json_str and c == '\\' and i + 1 < len(text):
            result.append(c)
            result.append(text[i + 1])
            i += 2
            continue
        if c in ('\u201c', '\u201d'):
            # Smart double quotes are always inside string values — escape them
            result.append('\\"')
        elif c == '"':
            # Heuristic: is this a JSON structural quote or a stray quote inside a value?
            # JSON structural quotes are preceded by: {, [, ,, :, whitespace, or start of text
            # and followed by: }, ], ,, :, whitespace, or end of text
            if in_json_str:
                # We're inside a string. Check if this quote is structural (closes the string)
                # or stray (part of the text content).
                # Look ahead: if followed by , : } ] or whitespace+key pattern, it's structural.
                rest = text[i + 1:i + 10].lstrip()
                if rest and rest[0] in (',', ':', '}', ']', '\n'):
                    in_json_str = False
                    result.append(c)
                elif not rest:
                    # End of text — structural
                    in_json_str = False
                    result.append(c)
                else:
                    # Stray quote inside string value — escape it
                    result.append('\\"')
            else:
                in_json_str = True
                result.append(c)
        elif ord(c) < 32 and c not in ('\n', '\r', '\t'):
            result.append(' ')
        else:
            result.append(c)
        i += 1

    return ''.join(result)


def _repair_truncated_json(text):
    """Attempt to repair truncated JSON from a model that hit its token limit.

    Strategy: find the JSON start, walk character by character tracking
    nesting depth properly (handling strings and escapes), then close
    any unclosed structures at the truncation point.
    """
    start = text.find('{')
    if start == -1:
        return None

    fragment = text[start:]

    # Trim the ragged end: find the last complete JSON token boundary.
    # Walk backward from the end to find a safe cut point: after a
    # comma, colon, closing bracket/brace, or end of a string value.
    # This removes partial keys, values, or strings at the truncation edge.
    last_safe = len(fragment)
    for j in range(len(fragment) - 1, max(len(fragment) - 500, 0), -1):
        c = fragment[j]
        if c in (',', ':', ']', '}', '\n'):
            last_safe = j + 1
            break
        if c == '"':
            # Check if this quote ends a complete string value
            last_safe = j + 1
            break

    fragment = fragment[:last_safe].rstrip().rstrip(',')

    # Track nesting to know what closers are needed
    depth_brace = 0
    depth_bracket = 0
    in_str = False
    esc = False
    for c in fragment:
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
            depth_brace += 1
        elif c == '}':
            depth_brace -= 1
        elif c == '[':
            depth_bracket += 1
        elif c == ']':
            depth_bracket -= 1

    # If we're inside an open string at EOF, close it
    if in_str:
        fragment += '"'

    # Close unclosed structures (innermost first)
    fragment += ']' * max(depth_bracket, 0)
    fragment += '}' * max(depth_brace, 0)

    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass

    # Last resort: try progressively stripping more from the end
    for trim in range(1, 50):
        lines = fragment.rsplit('\n', trim)
        if len(lines) > 1:
            candidate = lines[0].rstrip().rstrip(',')
            # Recount nesting
            db, dbk, ins, esc2 = 0, 0, False, False
            for c in candidate:
                if esc2:
                    esc2 = False
                    continue
                if c == '\\' and ins:
                    esc2 = True
                    continue
                if c == '"' and not esc2:
                    ins = not ins
                    continue
                if ins:
                    continue
                if c == '{':
                    db += 1
                elif c == '}':
                    db -= 1
                elif c == '[':
                    dbk += 1
                elif c == ']':
                    dbk -= 1
            if ins:
                candidate += '"'
            candidate += ']' * max(dbk, 0) + '}' * max(db, 0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Aggressive fallback for very long truncated responses:
    # Find the last complete "]," or "]" that closes a top-level array,
    # then close remaining sections with empty defaults.
    _SECTION_KEYS = [
        "PersonalDetails", "OverallSummary", "ListOfExperiences",
        "ListOfSkills", "PrimarySkills", "SecondarySkills",
        "ListOfEducation", "Certifications", "Projects",
        "Achievements", "Languages",
    ]

    # Strategy: find each top-level section's start position
    section_positions = []
    for key in _SECTION_KEYS:
        pattern = '"' + key + '"'
        idx = fragment.find(pattern)
        if idx >= 0:
            section_positions.append((idx, key))
    section_positions.sort()

    # Find the last section that was likely completed:
    # walk backwards through sections and try cutting after each one
    for i in range(len(section_positions) - 1, -1, -1):
        cut_key = section_positions[i][1]
        cut_idx = section_positions[i][0]

        # If there's a next section, cut just before it
        if i + 1 < len(section_positions):
            end_idx = section_positions[i + 1][0]
        else:
            end_idx = len(fragment)

        # Find the last "]" or "}" before the next section
        search_region = fragment[cut_idx:end_idx]
        last_close = max(search_region.rfind('],'), search_region.rfind('],\n'),
                         search_region.rfind(']\n'))
        if last_close < 0:
            last_close = max(search_region.rfind('},'), search_region.rfind('},\n'))
        if last_close < 0:
            continue

        candidate = fragment[:cut_idx + last_close + 1].rstrip().rstrip(',')

        # Add empty defaults for remaining sections
        found_keys = set()
        for pos, key in section_positions:
            if pos < cut_idx + last_close:
                found_keys.add(key)
        missing = []
        for key in _SECTION_KEYS:
            if key not in found_keys:
                if key in ("PersonalDetails", "OverallSummary"):
                    missing.append(f'  "{key}": {{}}')
                else:
                    missing.append(f'  "{key}": []')
        if missing:
            candidate += ',\n' + ',\n'.join(missing)
        candidate += '\n}'

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _extract_json(text):
    """Robust JSON extraction from LLM response.

    Handles: clean JSON, markdown-wrapped JSON, smart quotes/Unicode,
    and truncated JSON from token-limited responses.
    """
    text = text.strip()

    # Step 1: Sanitize Unicode characters
    text = _sanitize_json_text(text)

    # Step 2: Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3: Try markdown code block extraction
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Step 4: Brace matching — find outermost complete { }
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

    # Step 5: Attempt to repair truncated JSON (from token limit cutoff)
    return _repair_truncated_json(text)


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
        # Combine job title + summary + responsibilities into searchable text
        parts = []
        title = exp.get("JobTitle")
        if isinstance(title, str):
            parts.append(title)
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
        # Build search names: main name + RelevantSkills
        search_names = [name.lower()]
        relevant = skill.get("RelevantSkills")
        if isinstance(relevant, list):
            for rs in relevant:
                if isinstance(rs, str) and len(rs) >= 2:
                    search_names.append(rs.lower())
        total_months = 0
        latest_end = None
        for text, months, end_date_str in exp_data:
            matched = any(sn in text for sn in search_names)
            if not matched:
                for sn in search_names:
                    if len(sn) <= 3:
                        try:
                            if re.search(r'\b' + re.escape(sn) + r'\b', text):
                                matched = True
                                break
                        except re.error:
                            pass
            if matched:
                total_months += months
                parsed_end = _parse_date(end_date_str)
                if parsed_end and (latest_end is None or parsed_end > latest_end):
                    latest_end = parsed_end
        if total_months > 0:
            skill["SkillExperienceInMonths"] = total_months
            if latest_end:
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



def _fix_merge_summary(parsed):
    """Merge Summary into KeyResponsibilities as first bullet, then clear it."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        summary = exp.get("Summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        resps = exp.get("KeyResponsibilities")
        if not isinstance(resps, list):
            resps = []
        # Avoid duplicate if first bullet already matches summary
        if resps and isinstance(resps[0], str) and resps[0].strip() == summary.strip():
            pass  # already there as first bullet
        else:
            resps.insert(0, summary.strip())
        exp["KeyResponsibilities"] = resps


def _fix_project_company(parsed):
    """Infer CompanyWorked for projects from overlapping experience dates."""
    projects = parsed.get("Projects")
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(projects, list) or not isinstance(experiences, list):
        return

    exp_ranges = []
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start = _parse_date(exp.get("StartDate"))
        end = _parse_date(exp.get("EndDate"))
        company = exp.get("CompanyName", "")
        parts = []
        for r in (exp.get("KeyResponsibilities") or []):
            if isinstance(r, str):
                parts.append(r)
        if isinstance(exp.get("Summary"), str):
            parts.append(exp["Summary"])
        text = " ".join(parts).lower()
        exp_ranges.append((start, end, company, text))

    for proj in projects:
        if not isinstance(proj, dict):
            continue
        existing = proj.get("CompanyWorked")
        if isinstance(existing, str) and existing.strip() and existing.strip().lower() not in ("null", "none", "n/a"):
            continue

        proj_start = _parse_date(proj.get("StartDate"))
        proj_end = _parse_date(proj.get("EndDate"))
        best_company = None

        if proj_start and proj_end:
            for exp_start, exp_end, company, _ in exp_ranges:
                if exp_start and exp_end and company:
                    if proj_start <= exp_end and proj_end >= exp_start:
                        best_company = company
                        break

        if not best_company:
            techs = proj.get("Technologies") or []
            proj_name = (proj.get("ProjectName") or "").lower()
            for _, _, company, exp_text in exp_ranges:
                if not company:
                    continue
                matches = sum(1 for t in techs if isinstance(t, str) and t.lower() in exp_text)
                if matches >= 2 or (proj_name and len(proj_name) > 3 and proj_name in exp_text):
                    best_company = company
                    break

        if best_company:
            proj["CompanyWorked"] = best_company


_CERT_KEYWORDS = {"pmp", "scrum master", "csm", "ccna", "itil", "cissp", "cism", "cisa",
                   "aws certified", "azure certified", "safe", "six sigma", "prince2",
                   "togaf", "comptia", "certified scrum"}
_SOFT_SKILL_KEYWORDS = {"communication", "leadership", "teamwork", "problem solving",
                        "time management", "multi-tasking", "analytical skills",
                        "organizational skills", "detail-oriented", "interpersonal",
                        "adaptable", "creative", "organizational efficacy",
                        "process improvement", "process & organizational"}


_JOB_TITLE_KEYWORDS = {"manager", "engineer", "developer", "analyst", "architect",
                       "administrator", "consultant", "director", "lead", "specialist",
                       "coordinator", "supervisor", "officer", "executive", "intern",
                       "associate", "senior", "junior", "staff", "principal", "vp",
                       "president", "head of"}


def _fix_skill_contamination(parsed):
    """Remove certifications, soft skills, and job titles that leaked into ListOfSkills."""
    skills = parsed.get("ListOfSkills")
    if not isinstance(skills, list):
        return
    cleaned = []
    for skill in skills:
        if not isinstance(skill, dict):
            cleaned.append(skill)
            continue
        name = skill.get("SkillName", "")
        if not isinstance(name, str):
            cleaned.append(skill)
            continue
        name_lower = name.strip().lower()
        # Check if it's a certification
        if any(kw in name_lower for kw in _CERT_KEYWORDS):
            continue
        # Check if it's a soft skill
        if any(kw in name_lower for kw in _SOFT_SKILL_KEYWORDS):
            continue
        # Check if it's a job title (e.g., "Project Manager", "Software Developer")
        words = name_lower.split()
        if len(words) <= 3 and any(w in _JOB_TITLE_KEYWORDS for w in words):
            continue
        # Skip very long names (likely responsibility text, not a skill)
        if len(name) > 60:
            continue
        cleaned.append(skill)
    parsed["ListOfSkills"] = cleaned


def _fix_empty_certs(parsed):
    """Remove hallucinated placeholder certification objects with null/empty names."""
    certs = parsed.get("Certifications")
    if not isinstance(certs, list):
        return
    cleaned = []
    for cert in certs:
        if not isinstance(cert, dict):
            continue
        name = cert.get("CertificationName") or cert.get("Name") or ""
        if isinstance(name, str) and name.strip().lower() not in ("", "null", "none", "n/a"):
            cleaned.append(cert)
    parsed["Certifications"] = cleaned


def _find_skill_in_text(skill_name, text):
    """Check if a skill name appears in the resume text.

    Handles short names (C, R, Go) with word boundaries,
    special chars (C#, .NET, C++) with direct substring matching,
    and standard case-insensitive substring matching.
    """
    if not skill_name or not text:
        return False

    name = skill_name.strip()
    name_lower = name.lower()
    text_lower = text.lower()

    # Names with special chars (C#, C++, .NET, etc.) — use direct substring
    if any(c in name for c in '#.+/&'):
        return name_lower in text_lower

    # Very short names (1-2 chars like C, R) need word boundary matching
    if len(name) <= 2:
        try:
            return bool(re.search(r'\b' + re.escape(name_lower) + r'\b', text_lower))
        except re.error:
            return name_lower in text_lower

    # Direct case-insensitive substring match (handles most cases)
    if name_lower in text_lower:
        return True

    # Word-boundary match for medium-length names (avoids partial matches)
    if len(name) <= 20:
        try:
            if re.search(r'\b' + re.escape(name_lower) + r'\b', text_lower):
                return True
        except re.error:
            pass

    return False


def _fix_skill_hallucination(parsed, resume_text):
    """Remove skills whose SkillName does not appear in the resume text.

    The LLM sometimes generates plausible skills that aren't actually
    mentioned in the resume. This checks each SkillName against the
    original text and removes any that can't be found.
    """
    skills = parsed.get("ListOfSkills")
    if not isinstance(skills, list) or not resume_text:
        return

    cleaned = []
    for skill in skills:
        if not isinstance(skill, dict):
            cleaned.append(skill)
            continue
        name = skill.get("SkillName", "")
        if not isinstance(name, str) or not name.strip():
            continue  # drop skills with no name
        if _find_skill_in_text(name, resume_text):
            cleaned.append(skill)
        # else: skill not found in resume text, drop it

    parsed["ListOfSkills"] = cleaned


def _fix_total_experience(parsed):
    """Calculate TotalExperience from experience dates when missing or summary-only."""
    experiences = parsed.get("ListOfExperiences")
    summary = parsed.get("OverallSummary")
    if not isinstance(experiences, list) or not isinstance(summary, dict):
        return

    earliest_start = None
    latest_end = None
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        start = _parse_date(exp.get("StartDate"))
        end = _parse_date(exp.get("EndDate"))
        if start and (earliest_start is None or start < earliest_start):
            earliest_start = start
        if end and (latest_end is None or end > latest_end):
            latest_end = end

    if earliest_start and latest_end:
        total_months = _calc_months(earliest_start, latest_end)
        years = round(total_months / 12, 1)
        summary["TotalExperience"] = f"{years} years"


def _fix_languages(parsed):
    """Normalize Languages array — convert objects to strings, remove empty."""
    langs = parsed.get("Languages")
    if not isinstance(langs, list):
        return
    cleaned = []
    for lang in langs:
        if isinstance(lang, str) and lang.strip():
            cleaned.append(lang.strip())
        elif isinstance(lang, dict):
            val = (lang.get("Language") or lang.get("language")
                   or lang.get("Name") or lang.get("name") or "")
            if isinstance(val, str) and val.strip():
                cleaned.append(val.strip())
    parsed["Languages"] = cleaned


_METRIC_PATTERN = re.compile(r'\d+[\d,]*\.?\d*\s*[%$]|\$[\d,]+|\d+[kKmM]\+?|\d{2,}')


def _fix_achievements(parsed):
    """Deduplicate achievements, validate metrics, cap at 5."""
    achievements = parsed.get("Achievements")
    if not isinstance(achievements, list):
        return
    seen = set()
    cleaned = []
    for ach in achievements:
        if isinstance(ach, str):
            desc = ach
        elif isinstance(ach, dict):
            desc = ach.get("Description") or ach.get("description") or ""
        else:
            continue
        if not isinstance(desc, str) or not desc.strip():
            continue
        key = desc.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        if _METRIC_PATTERN.search(desc):
            cleaned.append(desc.strip() if isinstance(ach, str) else ach)
    parsed["Achievements"] = cleaned[:5]


def _fix_location_cleanup(parsed):
    """Remove industry descriptors from location fields."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        loc = exp.get("Location")
        if not isinstance(loc, str) or not loc.strip():
            continue
        cleaned = loc.strip()
        cleaned = re.sub(
            r'\s*\((?:Banking|Healthcare|Finance|Insurance|Entertainment|'
            r'Wireless|Retail|IT|Technology|Telecom|Manufacturing|'
            r'Energy|Automotive|Pharma|Media)[^)]*\)',
            '', cleaned, flags=re.IGNORECASE
        )
        company = exp.get("CompanyName", "")
        if isinstance(company, str) and company.strip() and len(company.strip()) > 2:
            cn = company.strip()
            if cn.lower() in cleaned.lower() and cn.lower() != cleaned.lower():
                cleaned = re.sub(re.escape(cn), '', cleaned, flags=re.IGNORECASE).strip()
                cleaned = re.sub(r'^[,\s\-/]+|[,\s\-/]+$', '', cleaned)
        if cleaned:
            exp["Location"] = cleaned

    # Clean education locations too
    education = parsed.get("ListOfEducation")
    if isinstance(education, list):
        for edu in education:
            if not isinstance(edu, dict):
                continue
            loc = edu.get("Location")
            inst = edu.get("Institution")
            if isinstance(loc, str) and isinstance(inst, str) and inst.strip():
                if inst.strip().lower() in loc.lower() and inst.strip().lower() != loc.lower():
                    cleaned = re.sub(re.escape(inst.strip()), '', loc, flags=re.IGNORECASE).strip()
                    cleaned = re.sub(r'^[,\s\-/]+|[,\s\-/]+$', '', cleaned)
                    if cleaned:
                        edu["Location"] = cleaned


def _fix_overall_responsibilities(parsed):
    """Aggregate top responsibilities from all roles into a top-level field."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    all_resps = []
    seen = set()
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        resps = exp.get("KeyResponsibilities")
        if not isinstance(resps, list):
            continue
        for r in resps[:3]:
            if isinstance(r, str) and r.strip():
                key = r.strip().lower()
                if key not in seen:
                    seen.add(key)
                    all_resps.append(r.strip())
    parsed["KeyResponsibilities"] = all_resps[:15]


def _fix_skill_dedup_and_cap(parsed):
    """Deduplicate skills case-insensitively, split into Primary/Secondary, cap at 25."""
    skills = parsed.get("ListOfSkills")
    if not isinstance(skills, list):
        return
    seen = set()
    deduped = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        name = skill.get("SkillName", "")
        if not isinstance(name, str) or not name.strip():
            continue
        key = name.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(skill)

    def sort_key(s):
        months = s.get("SkillExperienceInMonths")
        return months if isinstance(months, (int, float)) and months > 0 else -1
    deduped.sort(key=sort_key, reverse=True)

    parsed["PrimarySkills"] = [s["SkillName"] for s in deduped[:20]]
    parsed["SecondarySkills"] = [s["SkillName"] for s in deduped[20:]]
    parsed["ListOfSkills"] = deduped[:25]


def _fix_cert_validation(parsed):
    """Remove certifications with garbage data (responsibility text, too-long names)."""
    certs = parsed.get("Certifications")
    if not isinstance(certs, list):
        return
    cleaned = []
    for cert in certs:
        if not isinstance(cert, dict):
            continue
        name = cert.get("CertificationName") or cert.get("Name") or ""
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        # Skip if name is too long (likely responsibility text leaked in)
        if len(name) > 100:
            continue
        # Skip if name contains common responsibility verbs
        lower = name.lower()
        if any(verb in lower for verb in ["responsible for", "managed", "developed", "implemented",
                                           "designed", "coordinated", "led the", "worked on"]):
            continue
        cleaned.append(cert)
    parsed["Certifications"] = cleaned


def _fix_current_job_role(parsed):
    """Ensure CurrentJobRole is a proper job title, not a paragraph or responsibility."""
    summary = parsed.get("OverallSummary")
    if not isinstance(summary, dict):
        return
    role = summary.get("CurrentJobRole")
    if not isinstance(role, str) or not role.strip():
        # Try to get from most recent experience
        exps = parsed.get("ListOfExperiences")
        if isinstance(exps, list) and exps:
            first_exp = exps[0]
            if isinstance(first_exp, dict):
                title = first_exp.get("JobTitle")
                if isinstance(title, str) and title.strip():
                    summary["CurrentJobRole"] = title.strip()
        return

    # If role is too long (>80 chars), it's likely responsibility text
    if len(role.strip()) > 80:
        # Extract just the title portion (before comma, dash, or pipe)
        short = re.split(r'[,|\-–—]', role.strip())[0].strip()
        if len(short) > 5 and len(short) <= 80:
            summary["CurrentJobRole"] = short
        else:
            # Fall back to most recent experience title
            exps = parsed.get("ListOfExperiences")
            if isinstance(exps, list) and exps and isinstance(exps[0], dict):
                title = exps[0].get("JobTitle")
                if isinstance(title, str) and title.strip():
                    summary["CurrentJobRole"] = title.strip()


def _fix_responsibility_format(parsed):
    """Split long paragraph responsibilities into sentence-length bullets."""
    experiences = parsed.get("ListOfExperiences")
    if not isinstance(experiences, list):
        return
    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        resps = exp.get("KeyResponsibilities")
        if not isinstance(resps, list):
            continue
        cleaned = []
        for r in resps:
            if not isinstance(r, str) or not r.strip():
                continue
            text = r.strip()
            # If a single bullet is very long (>300 chars), split into sentences
            if len(text) > 300:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for s in sentences:
                    s = s.strip()
                    if s and len(s) > 10:
                        cleaned.append(s)
            else:
                cleaned.append(text)
        exp["KeyResponsibilities"] = cleaned[:10]  # Cap at 10 per role


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
        _fix_languages(parsed)
        applied.append("languages")
    except Exception:
        pass

    try:
        _fix_location_cleanup(parsed)
        applied.append("location_cleanup")
    except Exception:
        pass

    try:
        _fix_experience_years(parsed)
        applied.append("experience_years")
    except Exception:
        pass

    try:
        _fix_total_experience(parsed)
        applied.append("total_experience")
    except Exception:
        pass

    try:
        _fix_skill_experience(parsed)
        applied.append("skill_experience")
    except Exception:
        pass

    try:
        _fix_merge_summary(parsed)
        applied.append("merge_summary")
    except Exception:
        pass

    try:
        _fix_project_company(parsed)
        applied.append("project_company")
    except Exception:
        pass

    try:
        _fix_skill_contamination(parsed)
        applied.append("skill_contamination")
    except Exception:
        pass

    try:
        _fix_empty_certs(parsed)
        applied.append("empty_certs")
    except Exception:
        pass

    try:
        _fix_cert_validation(parsed)
        applied.append("cert_validation")
    except Exception:
        pass

    try:
        _fix_achievements(parsed)
        applied.append("achievements")
    except Exception:
        pass

    try:
        _fix_employment_type(parsed)
        applied.append("employment_type")
    except Exception:
        pass

    try:
        _fix_current_job_role(parsed)
        applied.append("current_job_role")
    except Exception:
        pass

    try:
        _fix_responsibility_format(parsed)
        applied.append("responsibility_format")
    except Exception:
        pass

    try:
        _fix_overall_responsibilities(parsed)
        applied.append("overall_responsibilities")
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
