import json
import os
from datetime import datetime
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

HISTORY_FILE = "history.json"

llm = ChatGroq(model_name="Gemma2-9b-It")

template = """
You are a smart AI assistant.

User ID: {user_id}
Product: {product_name}
User Purchase History: {history}

Based on the above, predict if the user will return the product. Respond only with "YES" or "NO" and give a short reason.
"""

prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

def save_to_history(user_id, product_name):
    new_entry = {
        "user_id": user_id,
        "product_name": product_name,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(new_entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_user_history(user_id):
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    user_entries = [entry for entry in history if entry["user_id"] == user_id]
    return user_entries[-5:]

def predict_return(user_id, product_name):
    history = get_user_history(user_id)
    formatted_history = json.dumps(history, indent=2)
    result = chain.invoke({
        "user_id": user_id,
        "product_name": product_name,
        "history": formatted_history
    })
    return result['text']

if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    product_name = input("Enter product name: ")

    save_to_history(user_id, product_name)
    prediction = predict_return(user_id, product_name)
    print("\nPrediction:", prediction)
