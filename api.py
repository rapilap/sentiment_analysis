from flask import Flask, request, jsonify
from sentiment import predict_sentiment

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text'}), 400
    
    result = predict_sentiment(data['text'])
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)