"""
Flask web application for AI Recruiter
Integrates with CrewAI agents for resume processing and candidate matching
"""

import os
import json
import uuid
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import threading

# Import our CrewAI agents
import sys
sys.path.insert(0, os.getcwd())
from agents.resume_ingress import ResumeIngressAgent
from agents.candidate_matcher import CandidateMatcherAgent

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ensure upload folder exists
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path('parsed_resumes').mkdir(exist_ok=True)

# Initialize CrewAI agents
resume_agent = ResumeIngressAgent()
matcher_agent = CandidateMatcherAgent()

# Store processing status (in production, use Redis or database)
processing_status = {}
recent_uploads = []
search_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page - redirect to upload"""
    return render_template('upload.html', recent_uploads=recent_uploads[-5:])

@app.route('/upload')
def upload_page():
    """Resume upload page"""
    return render_template('upload.html', recent_uploads=recent_uploads[-5:])

@app.route('/search')
def search_page():
    """Job description search page"""
    return render_template('search.html')

@app.route('/api/upload', methods=['POST'])
def upload_resume():
    """Handle resume file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF and DOCX allowed'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this upload
        upload_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{upload_id}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Initialize processing status
        processing_status[upload_id] = {
            'status': 'uploading',
            'filename': filename,
            'progress': 0,
            'steps': {
                'uploaded': False,
                'extracting': False,
                'parsing': False,
                'embedding': False,
                'storing': False
            },
            'error': None,
            'result': None
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=process_resume_background, args=(upload_id, filepath, filename))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'upload_id': upload_id,
            'filename': filename,
            'message': 'Upload successful, processing started'
        }), 200

def process_resume_background(upload_id, filepath, filename):
    """Process resume in background using CrewAI agent"""
    try:
        # Update status - file uploaded
        processing_status[upload_id]['steps']['uploaded'] = True
        processing_status[upload_id]['progress'] = 20
        processing_status[upload_id]['status'] = 'extracting'
        time.sleep(1)  # Simulate processing time
        
        # Update status - extracting text
        processing_status[upload_id]['steps']['extracting'] = True
        processing_status[upload_id]['progress'] = 40
        processing_status[upload_id]['status'] = 'parsing'
        time.sleep(1)
        
        # Update status - parsing resume
        processing_status[upload_id]['steps']['parsing'] = True
        processing_status[upload_id]['progress'] = 60
        processing_status[upload_id]['status'] = 'embedding'
        
        # Actually process the resume using CrewAI agent
        resume_id = resume_agent.process_resume(filepath)
        
        # Update status - generating embeddings
        processing_status[upload_id]['steps']['embedding'] = True
        processing_status[upload_id]['progress'] = 80
        processing_status[upload_id]['status'] = 'storing'
        time.sleep(1)
        
        # Update status - storing in database
        processing_status[upload_id]['steps']['storing'] = True
        processing_status[upload_id]['progress'] = 100
        processing_status[upload_id]['status'] = 'completed'
        processing_status[upload_id]['result'] = {
            'resume_id': resume_id,
            'message': 'Resume processed successfully'
        }
        
        # Add to recent uploads
        recent_uploads.insert(0, {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'resume_id': resume_id
        })
        
        # Keep only last 20 uploads
        if len(recent_uploads) > 20:
            recent_uploads.pop()
            
    except Exception as e:
        processing_status[upload_id]['status'] = 'error'
        processing_status[upload_id]['error'] = str(e)
        
        recent_uploads.insert(0, {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        })

@app.route('/api/status/<upload_id>')
def get_status(upload_id):
    """Get processing status for an upload"""
    if upload_id not in processing_status:
        return jsonify({'error': 'Upload ID not found'}), 404
    
    return jsonify(processing_status[upload_id])

@app.route('/api/search', methods=['POST'])
def search_candidates():
    """Search for candidates based on job description"""
    data = request.get_json()
    
    if not data or 'job_description' not in data:
        return jsonify({'error': 'Job description required'}), 400
    
    job_description = data['job_description']
    job_title = data.get('job_title', 'Position')
    
    # Generate search ID
    search_id = str(uuid.uuid4())
    
    # Start search in background
    thread = threading.Thread(target=search_candidates_background, args=(search_id, job_description, job_title))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'search_id': search_id,
        'message': 'Search started'
    }), 200

def search_candidates_background(search_id, job_description, job_title):
    """Search for candidates in background using CrewAI agent"""
    try:
        # Initialize search status
        search_results[search_id] = {
            'status': 'searching',
            'progress': 'Analyzing job requirements...',
            'results': None,
            'error': None
        }
        
        # Use the candidate matcher agent
        result = matcher_agent.find_best_candidates(job_description, job_title)
        
        # Format results for frontend
        formatted_results = {
            'job_title': result.job_title,
            'total_found': result.total_candidates_found,
            'search_query': result.search_query_used,
            'summary': result.summary,
            'candidates': []
        }
        
        for candidate in result.top_matches:
            formatted_results['candidates'].append({
                'id': candidate.candidate_id,
                'name': candidate.name,
                'match_score': int(candidate.match_score * 100),
                'rationale': candidate.rationale,
                'strengths': candidate.key_strengths,
                'concerns': candidate.potential_concerns,
                'experience': candidate.relevant_experience
            })
        
        search_results[search_id] = {
            'status': 'completed',
            'progress': 'Search completed',
            'results': formatted_results,
            'error': None
        }
        
    except Exception as e:
        search_results[search_id] = {
            'status': 'error',
            'progress': 'Search failed',
            'results': None,
            'error': str(e)
        }

@app.route('/api/search/<search_id>')
def get_search_results(search_id):
    """Get search results"""
    if search_id not in search_results:
        return jsonify({'error': 'Search ID not found'}), 404
    
    return jsonify(search_results[search_id])

@app.route('/api/extract_requirements', methods=['POST'])
def extract_requirements():
    """Extract key requirements from job description"""
    data = request.get_json()
    
    if not data or 'job_description' not in data:
        return jsonify({'error': 'Job description required'}), 400
    
    # Simple keyword extraction (in production, use NLP)
    job_desc = data['job_description'].lower()
    
    requirements = []
    
    # Check for years of experience
    import re
    years_match = re.search(r'(\d+)\+?\s*years?', job_desc)
    if years_match:
        requirements.append(f"{years_match.group(1)}+ years experience")
    
    # Check for common skills/technologies
    tech_keywords = ['python', 'javascript', 'java', 'react', 'node', 'aws', 'docker', 
                     'kubernetes', 'machine learning', 'ml', 'ai', 'sql', 'mongodb',
                     'tensorflow', 'pytorch', 'adobe', 'photoshop', 'illustrator']
    
    for tech in tech_keywords:
        if tech in job_desc:
            requirements.append(tech.title())
    
    # Check for degree requirements
    if 'bachelor' in job_desc or 'bs' in job_desc or 'ba' in job_desc:
        requirements.append("Bachelor's degree")
    if 'master' in job_desc or 'ms' in job_desc or 'ma' in job_desc:
        requirements.append("Master's degree")
    
    # Check for soft skills
    if 'leadership' in job_desc or 'lead' in job_desc or 'manage' in job_desc:
        requirements.append("Leadership experience")
    
    if 'team' in job_desc:
        requirements.append("Team collaboration")
    
    return jsonify({'requirements': requirements[:6]})  # Return top 6 requirements

if __name__ == '__main__':
    app.run(debug=True, port=5000)