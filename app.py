import streamlit as st
import pickle
import re
import pandas as pd


def predict_batch(texts):
    clean_texts = [preprocess(t) for t in texts]
    labels = model.predict(clean_texts)
    confidences = model.predict_proba(clean_texts).max(axis=1)

    return labels, confidences.round(2)


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"\s+", " ", text)
    return text

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

def predict(text: str):
    clean_text = preprocess(text)
    label = model.predict([clean_text])[0]
    confidence = max(model.predict_proba([clean_text])[0])
    return label, round(confidence, 2)

st.title("🧠 Sentiment Prediction App")

st.write("Type a sentence and click Predict.")

user_text = st.text_area("Input text")

if st.button("Predict"):
    if user_text.strip():
        label, confidence = predict(user_text)

        st.subheader("Prediction Result")
        st.json({
            "label": label,
            "confidence": confidence
        })
    else:
        st.warning("Please enter some text.")


st.divider()
st.header("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a 'text' column",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a column named 'text'")
    else:
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            labels, confidences = predict_batch(df["text"].astype(str))

            df["predicted_label"] = labels
            df["confidence"] = confidences

            st.subheader("Prediction Results")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download results as CSV",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

