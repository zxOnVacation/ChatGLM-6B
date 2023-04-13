from typing import List, Union, Dict
import logging
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModel
import uvicorn
import torch
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class Message(BaseModel):
    role: str
    content: str
    history: list = []


class Item(BaseModel):
    messages: List[Message]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Starting ChatGLM-6B service"}


@app.post("/llm/stream", response_model=None)
async def llm_stream(item: Item):
    contents = item.messages[0]
    prompt = contents.content
    history = contents.history
    max_length = item.max_tokens
    top_p = item.top_p
    temperature = item.temperature

    async def chat_generator():
        try:
            initial_string = ""
            for response, his in model.stream_chat(tokenizer, prompt, history, max_length=max_length,
                                                   top_p=top_p, temperature=temperature):
                text = response[len(initial_string):]
                initial_string = response
                print(repr(text))
                yield text
            torch_gc()
            yield '[DONE]'
        except Exception as e:
            logging.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    return EventSourceResponse(chat_generator(), sep="")


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=7860, workers=1)