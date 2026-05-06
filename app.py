import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

app = Flask(__name__)
API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)

# Store chat history per session (in-memory for now)
chat_sessions = {}

SYSTEM_PROMPT = """You are NutriMind — a smart, no-fluff diet and nutrition agent built for people who take their fitness seriously.

Your job:
- First ask for the user's fitness goal if not provided (weight loss, muscle gain, maintenance, recomp)
- Ask for their weight, height, age if needed to estimate calorie targets
- Track everything the user has eaten today as they tell you
- Calculate approximate calories and macros (protein, carbs, fats) for each food mentioned
- After each food log, tell the user:
  1. Macros of what they just ate (cal / P / C / F)
  2. Running daily total
  3. What they should eat next based on remaining targets — suggest specific Indian or common foods
- Be direct, specific, and practical. No fluff.
- Keep responses concise and well structured. Use emojis sparingly for readability.
- If the user asks what to eat, suggest actual meal options with approximate macros.
- Remember everything said in the conversation."""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id", "default")
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Init session if new
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history = chat_sessions[session_id]
    history.append(HumanMessage(content=user_message))

    try:
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + history)
        ai_message = response.content
        history.append(AIMessage(content=ai_message))
        chat_sessions[session_id] = history
        return jsonify({"response": ai_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    data = request.json
    session_id = data.get("session_id", "default")
    chat_sessions[session_id] = []
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)