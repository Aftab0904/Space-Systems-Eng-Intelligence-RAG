import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from rag_engine import get_query_engine, ingest_document
import asyncio

app = FastAPI()

# Global query engine
query_engine = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    citations: str

@app.on_event("startup")
async def startup_event():
    global query_engine
    try:
        # We don't ingest on startup to keep it fast, 
        # but we try to load the index if it exists
        query_engine = get_query_engine()
    except Exception as e:
        print(f"Error during startup: {e}")

@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    global query_engine
    if not query_engine:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized. Please deploy dataset first.")
    
    try:
        print(f"Processing query: {request.query}")
        # Run the query in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response_obj = await loop.run_in_executor(None, query_engine.query, request.query)
        
        # Extract citations
        source_nodes = getattr(response_obj, "source_nodes", [])
        citations = []
        for node in source_nodes:
            page = node.metadata.get('page_label', 'N/A')
            file_name = node.metadata.get('file_name', 'NASA Handbook')
            citations.append(f"Source: {file_name}, Page: {page}")
        
        return QueryResponse(
            response=str(response_obj),
            citations="\n".join(list(set(citations))) if citations else "No specific citations found."
        )
    except Exception as e:
        import traceback
        traceback.print_exc() # This will show the error in your terminal
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest():
    global query_engine
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ingest_document)
        query_engine = get_query_engine()
        return {"status": "success", "message": "Dataset ingested and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve Frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
