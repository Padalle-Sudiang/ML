from flask import Flask, render_template_string

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Flask Page</title>
</head>
<body>
    <h1>Halo dari Flask!</h1>
    <p>Ini adalah halaman HTML sederhana yang dikirim dari Flask.</p>
    <button onclick="alert('Tombol ditekan!')">Klik Saya</button>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
