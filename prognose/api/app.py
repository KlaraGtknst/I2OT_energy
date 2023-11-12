from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/forecast', methods=['POST'])
def forecast():
    return jsonify(
        {
            'timestamp': 'blabla',
            'consumption': 0.24
        }
    )



