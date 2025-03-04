from mangum import Mangum
from streamlit.web.cli import main as st_main
import os
import sys
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    original_argv = sys.argv
    try:
        sys.argv = ["streamlit", "run", "app.py", "--server.port=8000", "--server.headless=true"]
        st_main()
    finally:
        sys.argv = original_argv
    return {"message": "Streamlit app initialized"}

handler = Mangum(app)
