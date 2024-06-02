from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# load model
import torch
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
chat = ChatTTS.Chat()
chat.load_models()


class TextsInput(BaseModel):
    texts: List[str]
    params_infer_code: Dict
    params_refine_text: Dict

@app.post("/tts", response_class=JSONResponse)
async def tts(input: TextsInput):
    if not input.texts:
        raise HTTPException(status_code=400, detail="List of texts is required")
    
    try:
        # Use chatTTS to generate speech
        wavs = chat.infer(input.texts, 
                          params_refine_text=input.params_refine_text, 
                          params_infer_code=input.params_infer_code)
        
        return {"wavs": wavs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)