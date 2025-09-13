from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = FastAPI()

# Load model and tokenizer once at startup
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

class GenerationResponse:
    def __init__(self, output: str):
        self.output = output

@app.post("/generate/")
async def generate(prompt: str = Query("Hello, are you well?")):
    input_ids = tokenizer("translate English to French: " + prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Wrap result in a response object
    response = GenerationResponse(result_text)
    return JSONResponse(content=jsonable_encoder(response))