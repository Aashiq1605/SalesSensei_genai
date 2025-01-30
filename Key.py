from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env file

OPENAI_API_KEY = os.getenv("API_KEY")