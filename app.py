from flask import Flask, request, render_template
from sentiment import predict_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        input = request.form['text']
        prediction = predict_sentiment(input)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)