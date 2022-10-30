from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from pandas import infer_freq
from streamer import Streamer
import json
import warnings

app = Flask(__name__)
# http://127.0.0.1:8000
def gen():
  streamer = Streamer('127.0.0.1', 8000)
  streamer.start()

  while True:
    if streamer.streaming:
      yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + streamer.get_jpeg() + b'\r\n\r\n')

@app.route('/')
def index():
  return render_template('view.html')

@app.route('/test', methods=['GET'])
def result():
  with open("data.json", "r+") as jsonFile:
    data = json.load(jsonFile)

  Inference_value = data
  
  return jsonify(result = "success", result2= Inference_value)

@app.route('/video_feed')
def video_feed():
  return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
  app.run(host='127.0.0.1', threaded=True)
