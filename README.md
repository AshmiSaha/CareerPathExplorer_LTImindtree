**Note-  We have used virtual environment. Please use that, also use your own api key.**
# 🎓 Career Path Explorer

Career Path Explorer is an AI-powered web application that helps students discover personalized career paths based on their interests, skills, and academic background. It combines Machine Learning and Generative AI to suggest suitable careers, provide learning roadmaps, recommend resources, and simulate real-life experiences of various professions.

## 🚀 Features

- 🔍 *Career Prediction* using ML model (Random Forest)
- 🤖 *Generative AI fallback* using Gemini API for custom suggestions
- 🧭 *Personalized Roadmaps* with skills, courses, and tools
- 📚 *Recommended Resources* (books, online courses, platforms)
- 🎥 *Day-in-the-Life Simulation* generated via GenAI
- 🌐 *Multilingual Support* (English, Hindi, Bengali, Spanish)
- ⚡ *Modern UI* built with React and Tailwind CSS

---

## 🛠 Tech Stack

### 🎯 Frontend
- Flask API
- HTML, CSS

### 🧠 Backend
- Python
- Scikit-learn (RandomForestClassifier)
- Gemini API (Generative AI)

### 🗃 Dataset
- Custom dataset with academic scores, skills, extracurriculars
- Fields include: interests, skills, scores, suggested_career

---

## 🧪 How It Works

1. *User Input*  
   The user enters their academic background, skills, and interests via the frontend form.

2. *ML Prediction*  
   The backend model predicts a suitable career path using a trained RandomForestClassifier.

3. *Fallback to GenAI*  
   If no match is found, the app queries Gemini API to generate a relevant career suggestion and detailed response.

4. *Response Generation*  
   Based on the output, the app generates:
   - A career roadmap
   - Resource links
   - A simulated "day in the life" scenario for that profession
