import ollama

response = ollama.chat(
    model='llama3.2',
    messages=[
        {
            'role': 'user',
            'content': 'Explain LLM evaluation in simple terms.'
        }
    ]
)

print("\nAI Response:\n")
print(response['message']['content'])