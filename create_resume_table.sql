-- Create resume_data table based on ResumeData model
CREATE TABLE resume_data (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || substr(lower(hex(randomblob(2))),2) || '-' || substr('ab89',abs(random()) % 4 + 1, 1) || substr(lower(hex(randomblob(2))),2) || '-' || lower(hex(randomblob(6)))),
    
    -- Simple fields
    professional_summary TEXT,
    category TEXT,
    years_of_experience INTEGER,
    seniority_level TEXT,
    pdf_path TEXT,
    
    -- Complex fields stored as JSON
    personal_info TEXT CHECK (json_valid(personal_info)),
    work_experience TEXT CHECK (json_valid(work_experience)),
    education TEXT CHECK (json_valid(education)),
    skills TEXT CHECK (json_valid(skills)),
    certifications TEXT CHECK (json_valid(certifications)),
    languages TEXT CHECK (json_valid(languages)),
    keywords TEXT CHECK (json_valid(keywords)),
    
    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create trigger to update updated_at timestamp
CREATE TRIGGER update_resume_data_updated_at 
    AFTER UPDATE ON resume_data
    FOR EACH ROW
BEGIN
    UPDATE resume_data 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.id;
END;

-- Create indexes for common queries
CREATE INDEX idx_resume_category ON resume_data(category);
CREATE INDEX idx_resume_seniority ON resume_data(seniority_level);
CREATE INDEX idx_resume_experience ON resume_data(years_of_experience);