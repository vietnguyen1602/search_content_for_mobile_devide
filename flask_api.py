import json

from flask import Flask, request
from test_api import *
app = Flask(__name__)
from flask import jsonify

@app.route('/get_name_product', methods=['POST'])
def demo():
    try:
        # lấy dữ liệu từ BE gửi về
        data = request.json
        # Lấy url
        url = data["mess_text"]
        url_respone = search_image(url)

        return jsonify(url_respone)
    except:
        pass



if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 1998)