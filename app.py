from flask import Flask, render_template, request
from flask.ext.sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
db = SQLAlchemy(app)

from models import Temperatures
from datacenter import datacenter, datacenter_layout, gmrf, build_vector

import json

@app.route("/", methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == 'POST':
        data = json.loads(request.data.decode())
        name = data["url"].lower()

        if name not in datacenter.columns.values:
            errors.append("Name is not recognised")
        else:
            results = datacenter[name].values
            return str(results[0])

    return render_template('index.html', errors=errors, results=results)

@app.route("/data", methods=['POST'])
def get_layout():
    data = {}
    params = json.loads(request.data.decode())
    t = int(params["t"])
    keys = list(datacenter_layout.keys())
    for k in keys:
        x, y = datacenter_layout[k]
        v = datacenter[k][t]
        data[k] = (x, y, v)
    return json.dumps(data)

@app.route("/simulate", methods=['POST'])
def simulate():
    params = json.loads(request.data.decode())
    t = int(params["t"])
    ahus = params["ahus"]
    data_layout = params["state"]

    names = list(data_layout.keys())
    temps = [data_layout[n][2] for n in names]
    data = dict(zip(['l1_' + n for n in names], temps))
    for k in ahus.keys():
        data['l1_' + k] = ahus[k]

    for i in range(t):
        x = build_vector(data)
        state = gmrf.predict(x, names)
        data = dict(zip(['l1_' + n for n in names], state.ravel()))
        for k in ahus.keys():
            data['l1_' + k] = ahus[k]

    results = dict()
    for k in data_layout.keys():
        x, y, _ = data_layout[k]
        v = data['l1_' + k]
        results.update({k: (x, y, v)})
    return json.dumps(results)

if __name__ == "__main__":
    app.run()
