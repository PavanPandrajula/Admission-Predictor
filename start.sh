#!/bin/bash
# Start FastAPI backend with Uvicorn
uvicorn main:app --host 0.0.0.0 --port $PORT

