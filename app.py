#!/usr/bin/env python3
"""
Groq Resume Parser — API Server
Flask API for parsing resumes using Llama 3.1 via Groq.
"""

import os
import time
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from groq_parser import parse_resume, extract_text_from_file, is_groq_configured

app = Flask(__name__, static_folder='.')
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'groq_configured': is_groq_configured(),
        'model': os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant'),
        'timestamp': time.time(),
    })


@app.route('/parse', methods=['POST'])
def parse():
    """Parse an uploaded resume file."""
    start = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Send a file with key "file".'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Extract text
        resume_text = extract_text_from_file(filepath)
        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({'error': 'Could not extract text from file. Is it a valid resume?'}), 400

        # Parse with Groq
        result = parse_resume(resume_text)

        if 'error' in result:
            return jsonify(result), 502

        elapsed = int((time.time() - start) * 1000)

        return jsonify({
            'filename': filename,
            'processing_time_ms': elapsed,
            'result': result,
        })

    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


@app.route('/parse/text', methods=['POST'])
def parse_text():
    """Parse raw resume text (no file upload)."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Send JSON with "text" field containing resume text.'}), 400

    resume_text = data['text']
    if len(resume_text.strip()) < 50:
        return jsonify({'error': 'Resume text is too short.'}), 400

    result = parse_resume(resume_text)

    if 'error' in result:
        return jsonify(result), 502

    return jsonify({'result': result})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting Groq Resume Parser on port {port}")
    print(f"Model: {os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant')}")
    print(f"Groq API: {'configured' if is_groq_configured() else 'NOT SET — set GROQ_API_KEY'}")
    app.run(host='0.0.0.0', port=port, debug=True)
