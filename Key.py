from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access API key
OPENAI_API_KEY = os.getenv("API_KEY")


# Access MongoDB connection string
MONGODB_URI = os.getenv("MONGODB_URI")