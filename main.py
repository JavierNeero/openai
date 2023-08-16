import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import asyncio
from handler.query import QueryData, Embed
from handler.run import run_agent
from handler.agent import MegaAgent
from langchain import PromptTemplate, LLMChain
from mangum import Mangum
from loguru import logger

app = FastAPI()
handler = Mangum(app)
ss = Embed()

ada_agent = MegaAgent.initialize()

@app.on_event("startup")
async def load_data():
    #ss.load_data()
    pass

def semantic_search(query: str, top_n: int = 5):
    search_results = ss.search(query, top_n)
    results_str = ", ".join([f"{row.get('marca', 'N/A')} - {row.get('producto', 'N/A')}" for _, row in search_results.iterrows()])
    return results_str

@app.get("/")
async def root():
    return {"message": "Neero AI"}
    
@app.post("/chat")
async def query_and_chat(msg: dict = None):
    txt = msg.get('message', {}).get('text')
    if not txt:
        return JSONResponse(status_code=200, content={'action': 'reply', 'replies': ["Hola!, Soy Tulia, ¿En qué te puedo ayudar?"], "suggestions": ["Comprar Material","Comprar a en la App"]})
    try:
        qd = QueryData()
        result = qd.chat_data(question=txt, semantic_search=semantic_search)
        return JSONResponse(status_code=200, content={'action': 'reply', 'replies': [result['answer']]})
    except Exception as e:
        logger.error(f'[-] Chat failed: {e}')
        return JSONResponse(status_code=400, content={'action': 'reply', 'replies': 'Chat failed!'})

@app.post("/search")
async def embed_search(chat: dict):
    try:
        result = ss.search(query=chat['question'])
        if result is None:
            return {'answer': 'No match found!'}
        result_json = result.to_dict(orient='records')
        return {'answer': result_json}
    except Exception as e:
        logger.error(f'[-] Search failed: {e}')
        raise HTTPException(status_code=400, detail='Search failed!')

@app.post("/ada")
async def agente_ada(msg: dict = None):
    txt = msg.get('message', {}).get('text')
    if not txt:
        return JSONResponse(status_code=200, content={'action': 'reply', 'replies': ["Hola!, soy Ada, ¿En qué te puedo ayudar?"], "suggestions": ["Quiero un Crédito","Quiero comprar carro"]})
    try:
        prompt = PromptTemplate(
            template="""Plan: {input}
            History: {chat_history}
            Let's think about answer step by step.
            If it's information retrieval task, solve it like a professor in particular field.""",
            input_variables=["input", "chat_history"],
        )
        
        result = ada_agent.run(txt)
        print("Res: ", result)
        return JSONResponse(status_code=200, content={'action': 'reply', 'replies': [result]})
    except Exception as e:
        logger.error(f'[-] Chat failed: {e}')
        return JSONResponse(status_code=400, content={'action': 'reply', 'replies': 'Chat failed!'})