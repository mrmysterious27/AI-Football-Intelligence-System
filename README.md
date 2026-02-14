URL:https://ai-football-intelligence-system-e9pujfx63cpzebvsuexbiu.streamlit.app/
# AI-Football-Intelligence-System
Built an Expected Goals (xG) model using StatsBomb CSV shot data to predict goal probability. Engineered features like shot distance and angle, trained a Logistic Regression model, and deployed a Streamlit app for real-time xG predictions, demonstrating sports analytics and machine learning skills.

⚽ AI Football Intelligence System – Expected Goals (xG) Model
📌 Overview

This project develops an Expected Goals (xG) prediction system using predefined StatsBomb CSV shot data. The model estimates the probability that a shot results in a goal based on spatial and contextual features.

🎯 Objective

To analyze shot quality and calculate scoring probability using machine learning techniques.

📊 Dataset

The system uses a predefined StatsBomb CSV dataset containing shot coordinates, shot outcome, body part, shot type, and pressure information.

🧠 Feature Engineering

Shot distance calculated from goal center

Shooting angle derived using geometric relationships

Encoding of categorical features

🤖 Model

A Logistic Regression model is trained to predict goal probability (xG).
The output represents the likelihood of a shot being scored.

📈 Evaluation

Model performance is evaluated using accuracy and ROC-AUC metrics.

💻 Web Application

Built with Streamlit to display predictions and analytical insights interactively.

Run Locally
pip install -r requirements.txt
streamlit run app.py

📌 Author
Surya Chakraborty

🧩 Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit
