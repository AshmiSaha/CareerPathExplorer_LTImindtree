import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from difflib import get_close_matches
import requests
import pickle
import google.generativeai as genai  # Gemini API

# Load environment variables from .env file
load_dotenv()

# Set your Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY is not set in the environment.")
genai.configure(api_key=GEMINI_API_KEY)

# Find a free Gemini model
available_models = [m.name for m in genai.list_models()]
gemini_model_name = next((m for m in available_models if m.startswith('models/gemini')), None)
if not gemini_model_name:
    raise Exception("No Gemini model available for your API key.")

# Load the dataset
df = pd.read_csv('./career_path_dataset.csv')
le_interest = LabelEncoder()
le_skill = LabelEncoder()
le_academic = LabelEncoder()
le_career = LabelEncoder()

df['Interest_enc'] = le_interest.fit_transform(df['Interest'])
df['Skill_enc'] = le_skill.fit_transform(df['Skill'])
df['Academic_enc'] = le_academic.fit_transform(df['Academic Background'])
df['Career_enc'] = le_career.fit_transform(df['Career Path'])

X = df[['Interest_enc', 'Skill_enc', 'Academic_enc']]
y = df['Career_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Pickle dump the trained model
with open('career_path_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Pickle dump the label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'le_interest': le_interest,
        'le_skill': le_skill,
        'le_academic': le_academic,
        'le_career': le_career
    }, f)

def get_closest_match(user_input, choices):
    matches = get_close_matches(user_input, choices, n=1, cutoff=0.6)
    return matches[0] if matches else None

def query_gemini(interest, skill, academic_background, career=None):
    if career:
        prompt = (
            f"You are a career guidance assistant. "
            f"Given the following user profile:\n"
            f"- Interest: {interest}\n"
            f"- Skill: {skill}\n"
            f"- Academic Background: {academic_background}\n"
            f"- Predicted Career Path: {career}\n\n"
            f"1. Generate a detailed career roadmap for this career.\n"
            f"2. Recommend top online learning resources (courses, books, websites).\n"
            f"3. Simulate a 'day-in-the-life' scenario for this profession.\n"
            f"Be specific and actionable."
        )
    else:
        prompt = (
            f"You are a career guidance assistant. "
            f"Given the following user profile:\n"
            f"- Interest: {interest}\n"
            f"- Skill: {skill}\n"
            f"- Academic Background: {academic_background}\n\n"
            f"1. Suggest the most suitable career path(s) for this user.\n"
            f"2. Generate a detailed career roadmap for the suggested career(s).\n"
            f"3. Recommend top online learning resources (courses, books, websites).\n"
            f"4. Simulate a 'day-in-the-life' scenario for the recommended profession(s).\n"
            f"Be specific and actionable."
        )
    return query_gemini_http(prompt, GEMINI_API_KEY)

def query_gemini_http(prompt, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 512
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
    except requests.exceptions.RequestException as e:
        return f"Network error occurred: {str(e)}"
    if response.status_code == 200:
        result = response.json()
        try:
            candidates = result.get('candidates', [])
            if candidates and 'content' in candidates[0] and 'parts' in candidates[0]['content']:
                return candidates[0]['content']['parts'][0]['text']
            else:
                return "Error: Unexpected response format."
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return f"Error: {response.status_code} {response.text}"

def translate_text(text, target_language="en"):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "auto",  # auto-detect source language
        "tl": target_language,
        "dt": "t",
        "q": text,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        result = response.json()
        return ''.join([t[0] for t in result[0]])
    else:
        return text  # fallback to original

def predict_career_path(interest, skill, academic_background):
    """
    Takes user input, matches it to known categories using fuzzy matching,
    encodes it, and predicts the career path.
    If no close match is found, or the prediction is uncertain, uses Gemini GenAI.
    """
    gemini_cache = {}

    # Lowercase and strip inputs for robustness
    interest = interest.lower().strip()
    skill = skill.lower().strip()
    academic_background = academic_background.lower().strip()

    # Find closest matches from known categories
    interest_match = get_closest_match(interest, le_interest.classes_)
    skill_match = get_closest_match(skill, le_skill.classes_)
    academic_match = get_closest_match(academic_background, le_academic.classes_)

    if not interest_match or not skill_match or not academic_match:
        cache_key = (interest, skill, academic_background)
        if cache_key not in gemini_cache:
            gemini_cache[cache_key] = query_gemini(interest, skill, academic_background)
        return gemini_cache[cache_key]

    # Encode matched values
    interest_enc = le_interest.transform([interest_match])[0]
    skill_enc = le_skill.transform([skill_match])[0]
    academic_enc = le_academic.transform([academic_match])[0]

    # Predict and return career
    input_features = pd.DataFrame(
        [[interest_enc, skill_enc, academic_enc]],
        columns=['Interest_enc', 'Skill_enc', 'Academic_enc']
    )
    predicted_career_enc = model.predict(input_features)[0]
    predicted_career = le_career.inverse_transform([predicted_career_enc])[0]

    # If the predicted career is rare in the dataset, use Gemini GenAI
    if (df['Career Path'] == predicted_career).sum() < 10:
        cache_key = (interest, skill, academic_background)
        if cache_key not in gemini_cache:
            gemini_cache[cache_key] = query_gemini(interest, skill, academic_background)
        return gemini_cache[cache_key]
    else:
        cache_key = (interest, skill, academic_background, predicted_career)
        if cache_key not in gemini_cache:
            gemini_cache[cache_key] = query_gemini(interest, skill, academic_background, predicted_career)
        ai_guidance = gemini_cache[cache_key]
        return f"Predicted Career Path: {predicted_career}\n\nAI Guidance:\n{ai_guidance}"

# Supported languages (add more as needed)
language_options = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh-CN": "Chinese (Simplified)",
    "ar": "Arabic"
}

def prompt_in_language(prompt_text, lang_code):
    if lang_code == "en":
        return input(prompt_text)
    else:
        translated_prompt = translate_text(prompt_text, lang_code)
        return input(translated_prompt)

if __name__ == "__main__":
    print("Available Gemini models:", available_models)
    print("Columns in our dataset: ", df.columns)
    print(df.head())
    print(df.isnull().sum(axis=0))
    categorical_col = df[['Interest', 'Skill', 'Academic Background', 'Career Path']]
    for i in categorical_col:
        print(df[i].value_counts(), end="\n\n")
    sns.set(rc={'figure.figsize':(20,6)})
    sns.countplot(x = df["Career Path"])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    df_encoded = pd.get_dummies(df)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Supported languages:")
    for code, name in language_options.items():
        print(f"{code}: {name}")
    user_lang = input("Enter your preferred language code from the above options: ").strip().lower()
    if user_lang not in language_options:
        print("Invalid input. Please enter a valid language code (e.g., 'ml' for Malayalam).")
        exit()
    user_interest = prompt_in_language("Enter your interest: ", user_lang)
    user_skill = prompt_in_language("Enter your skill: ", user_lang)
    user_academic = prompt_in_language("Enter your academic background: ", user_lang)
    user_interest_en = translate_text(user_interest, "en")
    user_skill_en = translate_text(user_skill, "en")
    user_academic_en = translate_text(user_academic, "en")
    predicted_career_path = predict_career_path(user_interest_en, user_skill_en, user_academic_en)
    output_translated = translate_text(predicted_career_path, user_lang)
    print(output_translated)