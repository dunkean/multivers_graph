import flask
# Serve the file over http to allow for cross origin requests
app = flask.Flask(__name__, static_folder="viz")
@app.route("/")
def static_proxy():
    return app.send_static_file("3dforce.html")
    # return app.send_static_file("2dforce.html")


app.run(port=8000)
