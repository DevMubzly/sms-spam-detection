import os, json, random
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="Group 3 • SMS Spam Detection", page_icon="📨", layout="centered")
st.title("SMS Spam Detection — Group 3")
st.caption("AI Mini‑Project")

MODEL_PATH = "best_sms_spam_model.joblib"
META_PATH = "best_sms_spam_model.meta.json"
COMPARISON_CSV = "model_comparison.csv"

@st.cache_resource
def load_pipeline_and_meta():
    if not os.path.exists(MODEL_PATH):
        st.error("Model missing. Run notebook to generate best_sms_spam_model.joblib.")
        st.stop()
    pipe = load(MODEL_PATH)
    meta = None
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return pipe, meta

pipe, meta = load_pipeline_and_meta()
supports_proba = hasattr(pipe, "predict_proba")
supports_decision = hasattr(pipe, "decision_function")

with st.sidebar:
    st.subheader("MODEL SUMMARY ")
    st.info("Loaded saved TF‑IDF + classifier pipeline.")
    if meta:
        m = meta.get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1", f"{m.get('f1', 0):.3f}")
        c2.metric("Acc", f"{m.get('accuracy', 0):.3f}")
        c3.metric("Prec", f"{m.get('precision', 0):.3f}")
        c4.metric("Rec", f"{m.get('recall', 0):.3f}")
    else:
        st.warning("Metadata not found.")

    st.divider()
    st.subheader("Try examples")
    examples = [
        "WINNER!! As a valued customer you have been selected to receive a £900 prize! Claim at the link.",
        "Hey, are we still meeting for lunch? I can do 1:30.",
        "Your bank account is on hold. Verify now at http://secure-login.example.com",
        "Running 10 mins late—parking is crazy today.",
        "Limited time offer! Get 50% off your next purchase. Reply STOP to unsubscribe.",
        "Could you email me the slides from today's lecture?",
        "We couldn't deliver your parcel. Reschedule at http://delivery-fix.example",
        "Happy birthday! Hope you’re having a great day 🎉",
        "FINAL NOTICE: You’re owed a tax refund. Submit details within 24 hours.",
        "Thanks for covering my shift yesterday—lifesaver.",
        "Urgent: We've been trying to reach you about your car warranty. Call now.",
        "I reached home safely. Text me when you get back.",
        "Get instant loan approval. No credit check. Apply today.",
        "Reminder: Group meeting moved to Room 204 at 3pm.",
        "Congrats! You won 2 free tickets. Reply YES to claim.",
        "Where are you? I'm at the library near the entrance."
    ]
    if "example_pool" not in st.session_state:
        st.session_state.example_pool = examples.copy()
        random.shuffle(st.session_state.example_pool)
    if st.button("Shuffle examples"):
        random.shuffle(st.session_state.example_pool)
    pick = st.selectbox("Pick one", st.session_state.example_pool, index=0)
    if st.button("Use example"):
        st.session_state.msg_input = pick

msg = st.text_area("Enter an SMS message:", height=120, placeholder="Type or paste a message...", key="msg_input")

col1, col2 = st.columns(2)
with col1:
    run = st.button("Classify")
with col2:
    if st.button("Clear"):
        st.session_state.msg_input = ""
        st.experimental_rerun()

if run:
    if not msg.strip():
        st.warning("Please enter a message.")
    else:
        pred = int(pipe.predict([msg])[0])
        label = "SPAM" if pred == 1 else "Not Spam"
        detail = ""
        if supports_proba:
            try:
                p = pipe.predict_proba([msg])[0][1]
                detail = f" (confidence: {p:.2%})"
            except Exception:
                pass
        if not detail and supports_decision:
            try:
                s = pipe.decision_function([msg])[0]
                detail = f" (score: {s:.3f})"
            except Exception:
                pass
        (st.error if label == "SPAM" else st.success)(label + detail)

st.divider()
if os.path.exists(COMPARISON_CSV):
    try:
        df = pd.read_csv(COMPARISON_CSV).sort_values("F1", ascending=False).head(5)
        st.subheader("Top training models")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not read comparison file: {e}")
