from huggingface_hub import InferenceClient

client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token="your api key"  # Replace with your actual API key
)

def chat_with_phi(prompt):
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=False
    )

    return response['choices'][0]['message']['content'].strip()

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        response = chat_with_phi(user_input)
        print("chatbot:", response)
