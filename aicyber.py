# aicyber.py
"""
AI Cyber Safety - single-file Streamlit app
- Safe startup (no automatic heavy training)
- Optional manual ML training via button (if scikit-learn available)
- Fallback rule-only classifier if sklearn not present or no model provided
- Saves a pickle model to 'scam_detector.pkl' if you train locally (optional)
"""

import re
import pickle
from pathlib import Path
import streamlit as st
import numpy as np
from datetime import datetime

MODEL_PATH = Path("scam_detector.pkl")

# Try sklearn imports but don't fail if not installed
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    SKLEARN = True
except Exception:
    SKLEARN = False

# --- Tiny demo dataset (safe & scam examples) ---
DATA = [
    ("Your account has been suspended. Verify now at http://fake-bank-login.com", "scam"),
    ("URGENT: Update payment details or your subscription will be cancelled", "scam"),
    ("You've won a $1000 gift card! Click the link to claim", "scam"),
    ("Transfer 5000 INR immediately to avoid legal action", "scam"),
    ("Congrats! You are selected for a work-from-home job. Send bank details", "scam"),
    ("Hey, are we meeting at 6 pm today at the cafe?", "safe"),
    ("Reminder: project deadline is next Friday. Please push your code", "safe"),
    ("Happy birthday! Hope you have an amazing day 🎉", "safe"),
    ("I sent the assignment PDF to your email, check and confirm", "safe"),
    ("Movie night plan: let's vote for a movie by Friday", "safe"),
]

SCAM_KEYWORDS = [
    "verify", "suspended", "urgent", "transfer", "click", "claim", "won", "pay",
    "payment", "bank", "loan", "otp", "processing fee", "selected for", "tinyurl",
    "bit.ly", "send money", "final notice", "legal action", "training materials"
]
URL_REGEX = re.compile(r"https?://|www\.|tinyurl\.|bit\.ly", re.IGNORECASE)
UPI_REGEX = re.compile(r"@\w+$")

# --- Rule-only fallback model class ---
class RuleOnlyModel:
    def predict_proba(self, texts):
        out = []
        for t in texts:
            t_l = t.lower()
            score = 0.05
            if any(k in t_l for k in ["transfer", "pay", "payment", "bank", "send money"]):
                score += 0.4
            if "urgent" in t_l or "immediately" in t_l or "final notice" in t_l:
                score += 0.2
            if URL_REGEX.search(t):
                score += 0.2
            if any(k in t_l for k in ["verify", "verify now", "verify your", "suspended", "otp", "code"]):
                score += 0.25
            score = min(score, 0.99)
            out.append([1 - score, score])  # [prob_safe, prob_scam]
        return np.array(out)

# --- Helper functions ---
def find_keywords(text):
    found = []
    lower = text.lower()
    for kw in SCAM_KEYWORDS:
        if kw in lower:
            found.append(kw)
    if URL_REGEX.search(text):
        found.append("url/link")
    if UPI_REGEX.search(text):
        found.append("upi-handle")
    if re.search(r"\botp\b", lower) or re.search(r"\b(code|passcode)\b", lower):
        found.append("otp/code mention")
    if re.search(r"\b\d{4,6}\b", text):
        if "otp/code mention" not in found:
            found.append("number (possible OTP)")
    return found

def rule_based_score(text):
    score = 0
    lower = text.lower()
    if any(word in lower for word in ["transfer", "pay", "payment", "send money", "bank", "penalty", "processing fee"]):
        score += 0.35
    if "urgent" in lower or "immediately" in lower or "final notice" in lower:
        score += 0.25
    if URL_REGEX.search(text):
        score += 0.2
    if "verify" in lower and ("account" in lower or "identity" in lower or "login" in lower):
        score += 0.2
    return min(score, 1.0)

def train_model(data=DATA, max_features=500):
    if not SKLEARN:
        raise RuntimeError("scikit-learn not available in the environment.")
    texts = [t for t, _ in data]
    y = np.array([1 if label == "scam" else 0 for _, label in data])
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
    clf = LogisticRegression(max_iter=500)
    pipe = make_pipeline(vec, clf)
    pipe.fit(texts, y)
    return pipe

def save_model(pipe, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(pipe, f)

def load_pickled_model(path=MODEL_PATH):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="AI Cyber Safety", page_icon="🛡️", layout="centered")
st.title("🛡️ AI Cyber Safety — Quick Demo")
st.write("Paste a message (email/DM/WhatsApp) and click Analyze. The app uses a safe rule-based fallback; ML is optional and manual.")

# show environment info
with st.expander("App / env info (click)"):
    st.write(f"scikit-learn available: **{SKLEARN}**")
    st.write(f"Pretrained pickle found: **{MODEL_PATH.exists()}**")
    st.write("Tip: To avoid deploy-time delays, do not enable auto-train. Train manually when ready.")

# Model load / control area
st.sidebar.header("Model control")
model = None
pick = load_pickled_model()
if pick is not None:
    st.sidebar.write("Loaded saved model (scikit-learn pipeline).")
    model = pick
else:
    st.sidebar.write("No saved model found.")
    st.sidebar.write("ML training is optional (does not run at startup).")

# Manual train controls (user triggers training)
if SKLEARN:
    do_train = st.sidebar.checkbox("Enable ML training (manual)", value=False)
    maxf = st.sidebar.slider("TF-IDF max features (smaller = faster)", min_value=100, max_value=2000, value=500, step=100)
    if do_train:
        st.sidebar.warning("Training will run now in the server — keep dataset small. Press Train to start.")
        if st.sidebar.button("Train model now"):
            with st.spinner("Training model (small demo)..."):
                try:
                    pipe = train_model(DATA, max_features=maxf)
                    save_model(pipe)
                    model = pipe
                    st.sidebar.success("Model trained and saved to scam_detector.pkl ✅")
                except Exception as e:
                    st.sidebar.error(f"Training failed: {e}")
else:
    st.sidebar.info("scikit-learn not installed — using rule-only classifier.")

# User input area
text = st.text_area("Paste message to analyze", height=170, value="Hi, your account has been suspended. Verify now at http://fake-bank-login.com")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please paste a message first.")
        else:
            # ML prob
            if model is not None:
                try:
                    prob = float(model.predict_proba([text])[0][1])
                except Exception:
                    prob = None
            else:
                if SKLEARN and st.sidebar.checkbox("Force try loading saved model", value=False):
                    # try load again
                    pick2 = load_pickled_model()
                    if pick2:
                        model = pick2
                        prob = float(model.predict_proba([text])[0][1])
                    else:
                        prob = None
                else:
                    prob = None

            rb = rule_based_score(text)
            if prob is None:
                # fallback to rule-only
                fallback = RuleOnlyModel()
                ml_prob = float(fallback.predict_proba([text])[0][1])
                combined = min(1.0, 0.5 * ml_prob + 0.5 * rb)
                used = "rule-only (no ML model)"
            else:
                combined = min(1.0, 0.6 * prob + 0.4 * rb)
                used = "ML + rules"

            if combined >= 0.66:
                label = "High Risk (Likely Scam)"
                color = "red"
            elif combined >= 0.33:
                label = "Medium Risk (Suspicious)"
                color = "orange"
            else:
                label = "Low Risk (Likely Safe)"
                color = "green"

            st.markdown(f"### 🔎 {label}")
            st.write(f"**Combined risk score:** {combined:.2f}  ( ml_prob: {prob if prob is not None else 'N/A'} | rules: {rb:.2f} )")
            st.write(f"**Model used:** {used}")

            st.markdown("**Why:**")
            reasons = []
            kws = find_keywords(text)
            if kws:
                reasons.append("Detected suspicious keywords / patterns: " + ", ".join(kws))
            # model tokens (best-effort)
            if model is not None and SKLEARN:
                try:
                    vec = model.named_steps['tfidfvectorizer']
                    clf = model.named_steps['logisticregression']
                    x = vec.transform([text])
                    nz = x.nonzero()[1]
                    if len(nz) > 0:
                        feat_names = np.array(vec.get_feature_names_out())
                        coefs = clf.coef_[0]
                        feat_scores = list(zip(feat_names[nz], coefs[nz]))
                        feat_scores = sorted(feat_scores, key=lambda t: -abs(t[1]))[:6]
                        reasons.append("Model-relevant tokens: " + ", ".join([f"{f} ({s:.2f})" for f, s in feat_scores]))
                except Exception:
                    pass

            if not reasons:
                reasons = ["No obvious scam indicators detected by rules or model."]
            for r in reasons:
                st.write("- " + r)

            st.markdown("**Quick suggestions:**")
            if combined >= 0.66:
                st.write("- Do NOT click links or share OTPs/passwords or bank details.")
                st.write("- Verify sender via a known official channel.")
                st.write("- Report/block if it's an untrusted DM/email.")
            elif combined >= 0.33:
                st.write("- Be cautious: double-check links and money requests.")
                st.write("- Ask sender via trusted channel before acting.")
            else:
                st.write("- Looks safe, but never share OTPs/passwords/bank info.")

with col2:
    st.markdown("## Demo features")
    st.write("- App starts instantly (no auto training).")
    st.write("- You can train small ML model manually from sidebar (if sklearn exists).")
    st.write("- If you push `scam_detector.pkl` to repo, the app will auto-load it.")
    st.write("- Suitable for quick resume demo / screenshots.")

st.markdown("---")
st.caption(f"Demo app — small dataset. For production, use a large labeled dataset, cross-validation, and secure deployment. {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
