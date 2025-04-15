from huggingface_hub import InferenceClient

client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token="hf api key"  
)

import requests
from bs4 import BeautifulSoup

def crawl_website(url):
    """Crawl the specified website and extract relevant information."""
    try:
        response = requests.get(url)
        response.raise_for_status()  

        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')  
        return [p.get_text() for p in paragraphs]
    except requests.RequestException as e:
        print(f"Error during crawling: {e}")
        return None
    
content = crawl_website("https://HakiKenya.netlify.app")

def chat_with_phi(prompt):
    context = f"Document content: {content}\n\nUser question: {prompt}"
    
    response = client.chat_completion(
        messages=[{"role": "user", "content": context}],
        max_tokens=500,
        stream=False
    )

    return response['choices'][0]['message']['content'].strip()

if __name__ == "__main__":
    print("Chatbot is ready to answer questions about the document!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        response = chat_with_phi(user_input)
        print("Chatbot:", response)
