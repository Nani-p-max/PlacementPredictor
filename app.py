from flask import Flask, render_template

from load_data import (
    get_data_summary
)

app = Flask(__name__)



@app.route("/")
def index():
    return render_template(
        "index.html",
        active="none"
    )
@app.route("/data-loading")
def data_loading():
    error = None

    summary = None

    try:

        # Function 1:
        # Get basic dataset information
        summary = get_data_summary()


    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",

        # Tell HTML which section is active
        active="data-loading",

        # Send summary to HTML
        summary=summary,


        # Send error message to HTML
        error=error
    )



if __name__ == "__main__":
    app.run(debug=True)