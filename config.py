import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.environ.get('OPENAI_KEY', '')
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID', '')
MODEL_NAME = os.environ.get('MODEL_NAME', '')
MODEL = os.environ.get('MODEL', '')
