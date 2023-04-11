import asyncio
import logging
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from sse_starlette.sse import EventSourceResponse


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
STREAM_DELAY = 1  # second
RETRY_TIMEOUT = 15000  # milisecond


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Starting ChatGLM-6B service"}


@app.post("/llm/stream")
async def llm_stream(request: Request):
    request_body = await request.json()
    prompt = request_body.get('prompt')
    history = request_body.get('history', [])
    max_length = request_body.get('max_length', 2048)
    top_p = request_body.get('top_p', 0.7)
    temperature = request_body.get('temperature', 0.95)

    def get_stream(prompt, history, max_length, top_p, temperature):
        for response, history in model.stream_chat(tokenizer, prompt, history, max_length=max_length,
                                                   top_p=top_p,temperature=temperature):
            yield response

    async def chat_generator():
        while True:
            # initial_string = ""
            response = get_stream(prompt, history, max_length, top_p, temperature)
            logging.error(response)
            yield list(response)
            # get_string = "".join(list(response))
            # logging.error(get_string)
            # if get_string == initial_string:
            #     break
            # else:
            #     text = get_string[len(initial_string):]
            #     yield {"text": text}

            await asyncio.sleep(STREAM_DELAY)
        torch_gc()
    return EventSourceResponse(chat_generator())


# @app.post("/")
# async def create_item(request: Request):
#     global model, tokenizer
#     json_post_raw = await request.json()
#     json_post = json.dumps(json_post_raw)
#     json_post_list = json.loads(json_post)
#     prompt = json_post_list.get('prompt')
#     history = json_post_list.get('history')
#     max_length = json_post_list.get('max_length')
#     top_p = json_post_list.get('top_p')
#     temperature = json_post_list.get('temperature')
#     response, history = model.chat(tokenizer,
#                                    prompt,
#                                    history=history,
#                                    max_length=max_length if max_length else 2048,
#                                    top_p=top_p if top_p else 0.7,
#                                    temperature=temperature if temperature else 0.95)
#     now = datetime.datetime.now()
#     time = now.strftime("%Y-%m-%d %H:%M:%S")
#     answer = {
#         "response": response,
#         "history": history,
#         "status": 200,
#         "time": time
#     }
#     log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
#     print(log)
#     torch_gc()
#     return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=7860, workers=1)