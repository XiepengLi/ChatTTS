from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

# load model
import torch
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
chat = ChatTTS.Chat()
chat.load_models()
chat.infer("hello")

class TextsInput(BaseModel):
    texts: list[str]
    params_infer_code: dict
    params_refine_text: dict

@app.post("/tts", response_class=JSONResponse)
async def tts(input: TextsInput):
    if not input.texts:
        raise HTTPException(status_code=400, detail="List of texts is required")
    
    try:
        # Use chatTTS to generate speech
        wavs = chat.infer(input.texts, 
                          params_refine_text=input.params_refine_text, 
                          params_infer_code=input.params_infer_code)
        
        return {"wavs": [base64.b64encode(wav.tobytes()).decode('utf-8') for wav in wavs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# client
# import base64
# import numpy as np

# def decode_array(array_base64):
#     array_bytes = base64.b64decode(array_base64)
#     array_data = np.frombuffer(array_bytes, dtype=np.float32).reshape(1, -1)
#     return array_data

# # Example usage
# encoded_array = "..."  # The Base64 string from the response
# decoded_array = decode_array(encoded_array)