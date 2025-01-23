#!/usr/bin/env python3

"""
findcandi.py

A command line tool to analyze multiple CVs (PDF or Word) against a given query,
and select the top candidate(s) based on a Large Language Model "structured response" analysis
using enforced schemas.

Usage:
    python findcandi.py --cv_dir /path/to/cv/folder --query "Your requirement text" [--select 2]

Requirements:
    - PyPDF2 for PDF text extraction (pip install PyPDF2)
    - python-docx for DOCX text extraction (pip install python-docx)
    - openai library (pip install openai)

Environment:
    - Make sure you have your OpenAI API key available, typically set as
      an environment variable, e.g. export OPENAI_API_KEY="YOUR_OPENAI_KEY"
"""

import os
import io
import sys
import argparse
import json
from datetime import datetime

# PDF and DOCX text extraction
import PyPDF2
import docx

# Pydantic for schema enforcement
from pydantic import BaseModel, RootModel, Field
from typing import List


from flask import Flask, request, render_template_string
from openai import OpenAI

###############################################################################
# Flask App Initialization
###############################################################################
app = Flask(__name__)

###############################################################################
# Pydantic Schemas
###############################################################################
class AnalysisSchema(BaseModel):
    """
    Schema for the analysis step of each CV.
    Enforced fields in order:
      1. Name
      2. Age
      3. Gender
      4. WorkExperience
      5. Education
      6. Skills
      7. Highlights
      8. Fitness
    """
    Name: str = Field(..., description="Candidate name.")
    Age: int = Field(..., description="Candidate age in years.")
    Gender: str = Field(..., description="Male or Female.")
    WorkExperience: str = Field(..., description="Summary of the candidate's work experience.")
    Education: str = Field(..., description="Summary of the candidate's education history.")
    Skills: str = Field(..., description="Short, comma-separated keywords summarizing candidate skills.")
    Highlights: str = Field(..., description="Highlight some achievements or valuable skills, independent of Fitness.")
    Fitness: str = Field(..., description="Explanation (point-wise) of how well the candidate fits the query requirement.")


class SelectionSchema(BaseModel):
    """
    Schema for the final selection step of each chosen candidate.
    Fields in order:
      1. Name
      2. Age
      3. Gender
      4. Selection
      5. filename
    """
    Name: str = Field(..., description="Candidate name.")
    Age: int = Field(..., description="Candidate age in years.")
    Gender: str = Field(..., description="Male or Female.")
    Selection: str = Field(..., description="Detailed explanation for selecting this candidate.")
    filename: str = Field(..., description="The filename from which the candidate data originated.")

class SelectionList(BaseModel):
    # This is the top-level object, containing an array field:
    candidates: List[SelectionSchema] = Field(
        ..., 
        description="List of the selected candidates, each with relevant fields."
    )
    rejection: str = Field(
        ..., 
        description="Point-wise Explanation for why the other candidates were not selected."
    )

###############################################################################
# OpenAI Client + Helper for LLM calls
###############################################################################
client = OpenAI()

def _get_llm_response(messages, response_format=None, model="gpt-4o", temperature=0):
    """
    Uses the new (hypothetical) client.beta.chat.completions.parse(...) method
    to enforce a JSON schema response. The 'response_format' can be a single
    pydantic schema (e.g., AnalysisSchema) or a type like List[SelectionSchema].
    """
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
    )
    return response


###############################################################################
# File text extraction
###############################################################################
def extract_text_from_pdf(pdf_data: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    text_content = []
    pdf_stream = io.BytesIO(pdf_data)  # Wrap the bytes in a BytesIO object
    
    reader = PyPDF2.PdfReader(pdf_stream)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_content.append(page_text)
    return "\n".join(text_content)


def extract_text_from_docx(docx_data: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    doc_stream = io.BytesIO(docx_data)
    doc_ = docx.Document(doc_stream)
    full_text = [para.text for para in doc_.paragraphs]
    return "\n".join(full_text)


def get_file_text(file_path: str) -> str:
    """Return the text content from either a PDF or DOCX file."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        # For .doc or other formats, we can either skip or raise a warning.
        print(f"Warning: Unsupported file format '{ext}'. Skipping file: {file_path}")
        return ""


###############################################################################
# Main Analysis & Selection Logic
###############################################################################
def analyze_candidate(text_content: str, query: str, current_date: str) -> AnalysisSchema:
    """
    Analysis LLM call. Returns the enforced schema (AnalysisSchema).
    The system message instructs the model to produce ONLY valid JSON that
    can match the AnalysisSchema.
    """
    system_message = (
        "You are an assistant that must respond with JSON ONLY, conforming to this structure:\n"
        "Name (str), Age (int), Gender (str), WorkExperience (str), Education (str), "
        "Skills (str), Highlights (str), Fitness (str).\n\n"
        "Use the candidate's CV text, the query, and the current date to produce the best estimates.\n\n"
        "CV TEXT:\n"
        f"{text_content}\n\n"
        "QUERY:\n"
        f"{query}\n\n"
        f"CURRENT DATE: {current_date}\n\n"
        "Return only valid JSON matching the schema. No additional keys."
    )

    messages = [{"role": "system", "content": system_message}]
    # Enforce the schema for the response
    try:
        analysis: AnalysisSchema = _get_llm_response(
            messages=messages,
            response_format=AnalysisSchema,
            model="gpt-4o",
            temperature=0
        )
        return analysis
    except Exception as e:
        # If there's a parsing or model error, return a blank schema as fallback
        print(f"Warning: Could not parse analysis. Error: {e}")
        return AnalysisSchema(
            Name="",
            Age=0,
            Gender="",
            WorkExperience="",
            Education="",
            Skills="",
            Highlights="",
            Fitness=""
        )


def select_candidates(
    analysis_data: List[AnalysisSchema],
    select_count: int,
    query: str
) -> List[SelectionSchema]:
    """
    Selection LLM call. Returns a list of SelectionSchema items of length = select_count.
    The system message includes the entire analysis_data (in JSON form) and instructs the
    model to pick the top 'select_count' candidates, with an explanation in the "Selection" field.
    """
    # Convert analysis_data to a safe JSON for reference
    # We attach "filename" from the analysis step separately, so let's build a list of dict
    # that includes the existing fields plus 'filename' if it was stored separately.
    # We'll store each as dict and pass it in.
    analysis_data_as_dicts = []
    for ad in analysis_data:
        # ad is an AnalysisSchema but might have an extra "filename" if we appended it.
        # Because pydantic won't store unexpected fields, let's do a small hack:
        # We'll convert to dict and see if "filename" is in a hidden attr or we stored it externally.
        d = ad.model_dump()
        if hasattr(ad, "filename"):
            d["filename"] = getattr(ad, "filename")
        analysis_data_as_dicts.append(d)

    analysis_json_str = json.dumps(analysis_data_as_dicts, indent=2)

    system_message = f"""
        You are given an array of analyzed candidate data, each describing a person's profile.
        Your goal is to identify the top {select_count} candidates who best match the user's requirements
        based on the given query. The query is as follows:

        User Query: {query}

        You must:
        1. Rank the candidates from best to worst according to their suitability.
        2. Return exactly {select_count} candidates in that order (best first).
        3. Include a thorough justification in "Selection" for why each candidate was chosen.       
        """

    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": analysis_json_str}]

    try:
        # We expect a list of SelectionSchema
        response = _get_llm_response(
            messages=messages,
            response_format=SelectionList,
            model="gpt-4o",
            temperature=0
        )
        selection_list = json.loads(response.choices[0].message.content)
        return selection_list
    except Exception as e:
        print(f"Warning: Could not parse selection data. Error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CVs using an LLM (schema-enforced) and select top candidates."
    )
    parser.add_argument(
        "--cv_dir", required=True,
        help="Path to a directory containing CVs (PDF or Word DOCX)."
    )
    parser.add_argument(
        "--query", required=True,
        help="A string specifying the requirements."
    )
    parser.add_argument(
        "--select", type=int, default=1,
        help="How many candidates to select (default: 1)."
    )

    args = parser.parse_args()

    cv_dir = args.cv_dir
    query = args.query
    select_count = args.select

    # Check if we have an OpenAI API Key (the user might set it in their environment, or it may not be needed)
    # But let's at least warn if not found:
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set. "
              "Ensure your library is configured for authentication.")

    # 1) Gather all CV files in the directory
    all_files = os.listdir(cv_dir)
    cv_files = [
        f for f in all_files
        if f.lower().endswith(".pdf") or f.lower().endswith(".docx")
    ]

    current_date = str(datetime.now().date())
    analysis_results: List[AnalysisSchema] = []

    # Print a message so we know we are waiting for the CV analysis
    print("Analyzing CVs...")
    print("-" * 40)

    # 2) Analysis step for each CV
    for cv_file in cv_files:
        full_path = os.path.join(cv_dir, cv_file)
        extracted_text = get_file_text(full_path)
        if not extracted_text.strip():
            continue  # skip if we can't extract text

        # Call the LLM for analysis, enforcing the AnalysisSchema
        analysis_obj = analyze_candidate(extracted_text, query, current_date)
        # pretty print the analysis object
        print(json.dumps(json.loads(analysis_obj.choices[0].message.content), indent=2))
        print("-" * 40)

        # We can attach the filename after the schema is returned, since the schema doesn't have that field
        # We'll do a small hack by storing as a dynamic attribute or keep them separate:
        setattr(analysis_obj, "filename", cv_file)

        analysis_results.append(analysis_obj)

    print("Analysis complete. Selecting candidates...")

    # 3) Selection step
    selected_candidates = select_candidates(analysis_results, select_count, query)

    print("Selection complete. Results:")
    print("-" * 40)

    # 4) Print the final selection results
    for i, candidate in enumerate(selected_candidates["candidates"], start=1):
        print(f"Selection {i}: {candidate['Name']} (Age: {candidate['Age']}, Gender: {candidate['Gender']})")
        print(f"Why selected: {candidate['Selection']}")
        print(f"CV File: {candidate['filename']}")
        print("-" * 40)

    # Print the rejection message
    print("Rejection Explanation:")
    print(selected_candidates["rejection"])
    print("-" * 40)

###############################################################################
# Flask Routes & Templates
###############################################################################
FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FindCandi Web App</title>
</head>
<body>
    <h1>Upload CVs &amp; Enter Your Query</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="query">Your Requirement Query:</label><br/>
        <textarea name="query" id="query" rows="4" cols="50" required></textarea><br/><br/>

        <label for="select_count">Number of Candidates to Select:</label>
        <input type="number" name="select_count" id="select_count" value="1" min="1" style="width:50px;"><br/><br/>

        <label for="cv_files">Upload CVs (PDF/DOCX):</label><br/>
        <input type="file" name="cv_files" multiple accept=".pdf,.docx"/><br/><br/>

        <button type="submit">Analyze &amp; Select</button>
    </form>
</body>
</html>
"""

RESULT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FindCandi Results</title>
</head>
<body>
    <h1>Analysis &amp; Selection Results</h1>
    <div>
      {% if not selected_candidates.candidates and selected_candidates.rejection %}
         <p><strong>Rejection:</strong> {{ selected_candidates.rejection }}</p>
      {% else %}
         <h2>Selected Candidates:</h2>
         {% for c in selected_candidates.candidates %}
           <h3>Candidate {{ loop.index }}: {{ c.Name }}</h3>
           <ul>
               <li>Age: {{ c.Age }}</li>
               <li>Gender: {{ c.Gender }}</li>
               <li>Filename: {{ c.filename }}</li>
               <li><strong>Why Selected:</strong> {{ c.Selection }}</li>
           </ul>
         {% endfor %}
         <h2>Rejection Explanation:</h2>
         <p>{{ selected_candidates.rejection }}</p>
      {% endif %}
    </div>
</body>
</html>
"""

from jinja2 import Template

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Show the upload form
        return FORM_HTML
    
    # If POST, handle the uploaded files + query
    query = request.form.get("query", "")
    select_count = int(request.form.get("select_count", "1"))

    if not query.strip():
        return "Error: query cannot be empty.", 400

    # Extract files
    uploaded_files = request.files.getlist("cv_files")
    if not uploaded_files:
        return "Error: No files uploaded.", 400

    current_date = str(datetime.now().date())
    analysis_results = []

    # Analyze each uploaded file
    for file in uploaded_files:
        filename = file.filename.lower()
        file_bytes = file.read()
        if filename.endswith(".pdf"):
            cv_text = extract_text_from_pdf(file_bytes)
        elif filename.endswith(".docx"):
            cv_text = extract_text_from_docx(file_bytes)
        else:
            continue  # skip unsupported formats

        if cv_text.strip():
            analysis_obj = analyze_candidate(cv_text, query, current_date)
            # Attach filename for reference
            setattr(analysis_obj, "filename", file.filename)
            analysis_results.append(analysis_obj)

    if not analysis_results:
        return "Error: Could not extract data from uploaded files.", 400

    # Perform selection
    selected_candidates = select_candidates(analysis_results, select_count, query)

    # Render results
    template = Template(RESULT_HTML_TEMPLATE)
    html_output = template.render(selected_candidates=selected_candidates)
    return html_output


if __name__ == "__main__":
    # Run locally
    app.run(host="0.0.0.0", port=8000, debug=True)
