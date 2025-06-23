from flask import Flask, request, jsonify

app = Flask(__name__)
latest_output_json = {}  # shared state

@app.route('/detections', methods=['GET'])
def get_detections():
    return jsonify(latest_output_json)

@app.route('/update', methods=['POST'])
def update_detections():
    global latest_output_json
    latest_output_json = request.get_json(force=True)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
