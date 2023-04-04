
import logging
from flask import Flask, jsonify, request
import datetime
from transformers import AutoTokenizer, AutoModel
import torch


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')


app = Flask(__name__)

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model.eval()


@app.route('/hz')
def hz():
    return 'ok'


@app.route('/chatglm/infer', methods=['POST'])
def infer():
    if 1:
        print(request.args)
        print(dir(request))
        request_body = request.get_json()
        prompt = request_body.get('prompt')
        history = request_body.get('history', [])
        max_length = request_body.get('max_length', 2048)
        top_p = request_body.get('top_p', 0.7)
        temperature = request_body.get('temperature', 0.95)
        return 'ok'

        for response, history in model.stream_chat(tokenizer, prompt, history, max_length=max_length, top_p=top_p,
                                                   temperature=temperature):
            logging.error(response)
            logging.error(history)
            now = datetime.datetime.now()
            time = now.strftime("%Y-%m-%d %H:%M:%S")
            answer = {
                "response": response,
                "history": history,
                "status": 200,
                "time": time}
            yield jsonify(answer)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
