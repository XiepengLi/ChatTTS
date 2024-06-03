from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import pickle
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

def generate_audio(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, spk_emb=None):

    torch.manual_seed(audio_seed_input)
    if spk_emb is None:
        spk_emb = chat.sample_random_speaker()

    params_infer_code = {
        'spk_emb': spk_emb, 
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    
    torch.manual_seed(text_seed_input)

    if refine_text_flag:
        text = chat.infer(text, 
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
    
    wav = chat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    sample_rate = 24000

    return sample_rate, spk_emb, wav

class TTSInput(BaseModel):
    text: list[str]
    temperature: float=0.3
    top_P: float=0.7
    top_K: int=20
    audio_seed_input: int=2
    text_seed_input: int=2
    refine_text_flag: bool=True
    spk_emb: str=""

@app.post("/tts", response_class=JSONResponse)
async def tts(input: TTSInput):
    if not input.text:
        raise HTTPException(status_code=400, detail="List of texts is required")
    
    try:
        # Use chatTTS to generate speech
        if input.spk_emb:
            input.spk_emb = pickle.loads(base64.b64decode(input.spk_emb))
        sample_rate, spk_emb, wavs = generate_audio(**dict(input))
        
        return {"wavs": [base64.b64encode(wav.tobytes()).decode('utf-8') for wav in wavs],
                "spk_emb": base64.b64encode(pickle.dumps(spk_emb)).decode('utf-8'), #.cpu().detach().numpy().tolist(),
                "sample_rate": sample_rate}
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