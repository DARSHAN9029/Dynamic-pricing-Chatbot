from flask import Flask, render_template, request, jsonify
from chatbot.query_bot import get_bot_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

@app.route("/")
def index():
    return render_template("ui.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    bot_reply = get_bot_response(user_input)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)