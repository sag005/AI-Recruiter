# AI Recruiter justfile - Process resumes with Anthropic and Voyage AI

# Default recipe - show available commands
default:
    @just --list

# Process a single PDF resume
process-pdf pdf_path:
    #!/usr/bin/env bash
    uv run python3 - <<'EOF'
    import sys
    import os
    from pathlib import Path
    
    # Add current directory to path for imports
    sys.path.insert(0, os.getcwd())
    
    from real_resume_parser import ResumeParser
    from dotenv import load_dotenv
    
    load_dotenv()
    
    def process_single_pdf(pdf_path):
        """Process a single PDF and display results."""
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file not found: {pdf_path}")
            return
        
        print("🎯 AI RECRUITER - Single PDF Processor")
        print("=" * 60)
        print(f"📄 Processing: {Path(pdf_path).name}")
        print("=" * 60)
        
        parser = ResumeParser()
        
        # Parse resume
        print("🔍 PARSING RESUME...")
        try:
            resume_data = parser.parse_resume(pdf_path)
            
            print("✅ PARSING SUCCESSFUL!")
            print(f"👤 Name: {resume_data.personal_info.name or 'Not found'}")
            print(f"📧 Email: {resume_data.personal_info.email or 'Not found'}")
            print(f"📞 Phone: {resume_data.personal_info.phone or 'Not found'}")
            print(f"📍 Location: {resume_data.personal_info.location or 'Not found'}")
            
            if resume_data.professional_summary:
                print(f"📝 Summary: {resume_data.professional_summary[:100]}...")
            
            print(f"\n🏢 WORK EXPERIENCE ({len(resume_data.work_experience)} jobs):")
            for i, job in enumerate(resume_data.work_experience[:3], 1):  # Show first 3 jobs
                print(f"  {i}. {job.job_title} at {job.company}")
                print(f"     Duration: {job.duration or 'Not specified'}")
                if job.responsibilities:
                    print(f"     Key responsibility: {job.responsibilities[0]}")
            
            print(f"\n🎓 EDUCATION ({len(resume_data.education)} entries):")
            for edu in resume_data.education:
                print(f"  • {edu.degree} from {edu.institution}")
                if edu.graduation_year:
                    print(f"    Year: {edu.graduation_year}")
            
            print(f"\n🔧 SKILLS ({len(resume_data.skills)} total):")
            skills_display = ', '.join(resume_data.skills[:10])
            if len(resume_data.skills) > 10:
                skills_display += f"... (+{len(resume_data.skills) - 10} more)"
            print(f"  {skills_display}")
            
            print(f"\n📊 ANALYSIS:")
            print(f"  Category: {resume_data.category or 'Unknown'}")
            print(f"  Experience: {resume_data.years_of_experience or 'Unknown'} years")
            print(f"  Level: {resume_data.seniority_level or 'Unknown'}")
            
            if resume_data.certifications:
                print(f"\n🏆 CERTIFICATIONS ({len(resume_data.certifications)}):")
                for cert in resume_data.certifications[:3]:
                    print(f"  • {cert.name}")
            
            if resume_data.processing_notes:
                print(f"\n⚠️  PROCESSING NOTES:")
                for note in resume_data.processing_notes:
                    print(f"  - {note}")
            
        except Exception as e:
            print(f"❌ PARSING FAILED: {str(e)}")
            return
        
        # Create embeddings
        print(f"\n🔢 CREATING EMBEDDINGS...")
        try:
            embedding_result = parser.create_embeddings(pdf_path)
            
            if "error" not in embedding_result:
                print(f"✅ Embedding created successfully!")
                print(f"📐 Dimensions: {embedding_result['embedding_dimension']}")
                print(f"📝 Text length: {embedding_result['text_length']} characters")
                print(f"🤖 Model: {embedding_result['model_used']}")
                print(f"🏢 Provider: {embedding_result['provider']}")
            else:
                print(f"❌ Embedding failed: {embedding_result['error']}")
        
        except Exception as e:
            print(f"❌ Embedding generation failed: {str(e)}")
        
        print(f"\n{'=' * 60}")
        print("🎉 Processing complete!")
    
    # Run the processing
    pdf_path = "{{pdf_path}}"
    process_single_pdf(pdf_path)

# Process multiple PDFs from a directory
process-dir directory limit="5":
    #!/usr/bin/env python3
    import sys
    import os
    from pathlib import Path
    
    # Add current directory to path for imports
    sys.path.insert(0, os.getcwd())
    
    from real_resume_parser import ResumeParser
    from dotenv import load_dotenv
    
    load_dotenv()
    
    def process_directory(directory, limit):
        """Process multiple PDFs from a directory."""
        if not os.path.exists(directory):
            print(f"❌ Error: Directory not found: {directory}")
            return
        
        pdf_files = list(Path(directory).glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in {directory}")
            return
        
        # Limit the number of files
        limit = int(limit)
        if limit > 0:
            pdf_files = pdf_files[:limit]
        
        print("🎯 AI RECRUITER - Batch PDF Processor")
        print("=" * 60)
        print(f"📂 Directory: {directory}")
        print(f"📄 Processing {len(pdf_files)} PDFs (limit: {limit})")
        print("=" * 60)
        
        parser = ResumeParser()
        results = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            print("-" * 40)
            
            try:
                resume_data = parser.parse_resume(str(pdf_path))
                embedding_result = parser.create_embeddings(str(pdf_path))
                
                # Summary
                name = resume_data.personal_info.name or "Unknown"
                category = resume_data.category or "Unknown"
                experience = resume_data.years_of_experience or "Unknown"
                skills_count = len(resume_data.skills)
                
                embedding_status = "✅" if "error" not in embedding_result else "❌"
                
                print(f"  Name: {name}")
                print(f"  Category: {category}")
                print(f"  Experience: {experience} years")
                print(f"  Skills: {skills_count}")
                print(f"  Embedding: {embedding_status}")
                
                results.append({
                    "file": pdf_path.name,
                    "name": name,
                    "category": category,
                    "experience": experience,
                    "skills_count": skills_count,
                    "embedding_success": "error" not in embedding_result
                })
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                results.append({
                    "file": pdf_path.name,
                    "error": str(e)
                })
        
        # Summary
        print(f"\n{'=' * 60}")
        print("📊 BATCH PROCESSING SUMMARY")
        print(f"Total processed: {len(results)}")
        successful = len([r for r in results if "error" not in r])
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        # Category breakdown
        categories = {}
        for result in results:
            if "error" not in result:
                cat = result["category"]
                categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            print(f"\nCategory breakdown:")
            for cat, count in categories.items():
                print(f"  {cat}: {count}")
    
    # Run the processing
    directory = "{{directory}}"
    limit = "{{limit}}"
    process_directory(directory, limit)

# Process sample resumes from filtered data
demo:
    @echo "🎯 Running demo with sample resumes..."
    uv run python real_resume_parser.py

# Setup development environment
setup:
    @echo "🔧 Setting up development environment..."
    uv sync
    @echo "✅ Setup complete! Make sure to set ANTHROPIC_API_KEY and VOYAGE_API_KEY in .env file"

# Run tests
test:
    @echo "🧪 Running tests..."
    uv run pytest

# Format code
format:
    @echo "🎨 Formatting code..."
    uv run black src/
    uv run isort src/

# Lint code  
lint:
    @echo "🔍 Linting code..."
    uv run flake8 src/
    uv run mypy src/

# Clean up generated files
clean:
    @echo "🧹 Cleaning up..."
    rm -f *.json
    rm -f *.log
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete
    @echo "✅ Cleanup complete!"

# Show system info
info:
    @echo "🔍 AI Recruiter System Information"
    @echo "=================================="
    @echo "Python version:"
    @python3 --version
    @echo ""
    @echo "Dependencies:"
    @uv tree --depth 1
    @echo ""
    @echo "Data directory status:"
    @ls -la data_filtered/ 2>/dev/null || echo "No data_filtered directory found"
    @echo ""
    @echo "Environment variables:"
    @echo "ANTHROPIC_API_KEY: $${ANTHROPIC_API_KEY:+✅ Set}" 
    @echo "VOYAGE_API_KEY: $${VOYAGE_API_KEY:+✅ Set}"