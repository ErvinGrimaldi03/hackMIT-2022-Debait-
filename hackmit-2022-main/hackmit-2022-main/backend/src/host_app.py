import flask
from flask import request, jsonify
from news_scraping import politics_list, sports_list

app = flask.Flask(__name__)
app.config["DEBUG"] = True

_links = {
    "Politics": politics_list,
    "Sports": sports_list
}


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''


@app.route('/News', methods=['GET'])
def api_all():
    return jsonify(_links)

app.run()