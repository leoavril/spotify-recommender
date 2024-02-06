from flask import Flask, request, render_template, redirect, url_for


app = Flask(__name__)

tracks = ['5xmaFH9oMYg8SMMTndypEh', '4EnwhEyuVrC1CgvSur5YL4', '4k9EAtZdZrzPlBUsFncXCZ', '6p6TjiJHc1kJQt5dXzkdrs', '6TRp2628QKH3kY6KrCnjqp',
          '5SWfocxhAkjf3eg4lrv8vZ', '6NqfslFfCCOyBQ0Xw8kN8E', '2XQ1tVtmHNKBOxaCbjYxdn', '1uHt85NYsb2JJ05FkAK6as', '0IzIW2MSFdllQ7zFLbv1uS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    pass


if __name__ == '__main__':
    app.run(debug=True)
