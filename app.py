from flask import Flask, render_template, request
from combined import predict_career_path, translate_text, language_options

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_interest = user_skill = user_academic = ''
    selected_language = 'en'
    if request.method == "POST":
        user_interest = request.form.get("interest", "")
        user_skill = request.form.get("skill", "")
        user_academic = request.form.get("academic", "")
        selected_language = request.form.get("language", "en")
        # Translate input to English if needed
        user_interest_en = translate_text(user_interest, "en") if selected_language != "en" else user_interest
        user_skill_en = translate_text(user_skill, "en") if selected_language != "en" else user_skill
        user_academic_en = translate_text(user_academic, "en") if selected_language != "en" else user_academic
        # Get prediction in English
        result_en = predict_career_path(user_interest_en, user_skill_en, user_academic_en)
        # Translate result back to selected language if needed
        result = translate_text(result_en, selected_language) if selected_language != "en" else result_en
    return render_template("index.html", result=result, user_interest=user_interest, user_skill=user_skill, user_academic=user_academic, selected_language=selected_language, language_options=language_options)

@app.route("/products")
def products():
    return "<p>this is product page</p>"

if __name__ == "__main__":
    app.run(debug=True, port=8000)