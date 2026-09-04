from huggingface_hub import InferenceClient

client = InferenceClient()

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "user", "content": "What is Agentic AI?"}
    ],
)

print(response.choices[0].message.content)