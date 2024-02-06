from flask import Flask, request, render_template, redirect, url_for


app = Flask(__name__)

tracks = []


@app.route('/')
def index():
    return render_template('index.html', tracks=tracks)


@app.route('/predict')
def predict():
    pass


if __name__ == '__main__':
    app.run(debug=True)
