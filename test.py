import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()  # loads your .env file

print("Token loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))  # debug
