from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # بنغيّرها حسب الدومين بعدين
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

@app.post("/llm")
async def process_command(request: Request):
    data = await request.json()
    user_input = data.get("command", "")

    prompt = f"""
    You are an educational AI assistant for the GoAlgo project.
    The user may ask either to run or visualize an algorithm or to explain one.
    Respond clearly and concisely in educational style.
    User input: {user_input}
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=250)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response.strip()}
