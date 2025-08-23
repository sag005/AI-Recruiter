# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GMAIL_CREDENTIALS = os.getenv("GMAIL_CREDENTIALS_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")