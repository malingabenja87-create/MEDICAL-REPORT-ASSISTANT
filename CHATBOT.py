import base64
import os
import re
from io import BytesIO

import streamlit as st
import pytesseract
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="Medical Report Assistant", page_icon="🩺", layout="wide")

st.title("🩺 Medical Report Assistant")
st.write(
    "Upload a PDF or image of lab results and medical reports. The app extracts content, highlights important findings, and supports an assistant chat that stays grounded in the uploaded report."
)

DISCLAIMER = "This tool is for educational information only and is not a medical diagnosis. Please consult a qualified clinician for medical decisions."

LAB_RANGES = {
    "glucose": (70, 100),
    "hemoglobin": (12.0, 16.0),
    "hgb": (12.0, 16.0),
    "a1c": (4.0, 5.6),
    "hba1c": (4.0, 5.6),
    "wbc": (4.0, 11.0),
    "white blood cells": (4.0, 11.0),
    "platelets": (150.0, 450.0),
    "cholesterol": (0, 200),
    "ldl": (0, 100),
    "hdl": (40, 100),
    "triglycerides": (0, 150),
    "sodium": (135, 145),
    "potassium": (3.5, 5.0),
    "creatinine": (0.6, 1.2),
    "ast": (5, 40),
    "alt": (7, 56),
    "bilirubin": (0.1, 1.2),
    "crp": (0, 5),
}

SYMPTOM_TERMS = [
    "headache",
    "fever",
    "cough",
    "fatigue",
    "dizziness",
    "shortness of breath",
    "nausea",
    "vomiting",
    "chest pain",
    "rash",
    "swelling",
    "pain",
    "diarrhea",
    "constipation",
    "weight loss",
    "insomnia",
]


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text


def extract_text_from_image(uploaded_file, api_key=None):
    uploaded_file.seek(0)
    image_bytes = uploaded_file.getvalue()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    try:
        text = pytesseract.image_to_string(image).strip()
        if text:
            return text
    except Exception:
        pass

    client = get_openai_client(api_key)
    if client:
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = uploaded_file.type or "image/png"
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Extract all readable text from this medical report image. Return plain text only.",
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{image_b64}",
                            },
                        ],
                    }
                ],
                temperature=0.1,
            )
            text = response.output_text.strip()
            if text:
                return text
        except Exception as exc:
            return f"OCR unavailable: {exc}"

    return "OCR unavailable: Tesseract is not installed and no OpenAI API key was provided."


def extract_report_text(uploaded_file, api_key=None):
    file_type = uploaded_file.type or ""
    if file_type.startswith("image/"):
        return extract_text_from_image(uploaded_file, api_key=api_key)
    return extract_text_from_pdf(uploaded_file)


def clean_value(value_str):
    value_str = value_str.replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)", value_str)
    if not match:
        return None
    return float(match.group(1))


def parse_lab_values(text):
    findings = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        lower = line.lower()
        for key, range_ in LAB_RANGES.items():
            if key in lower:
                value = clean_value(line)
                if value is not None:
                    findings.append((key, value, range_))
                    break
    return findings


def detect_symptoms(text):
    found = []
    lowered = text.lower()
    for symptom in SYMPTOM_TERMS:
        if symptom in lowered:
            found.append(symptom)
    return found


def interpret_findings(findings):
    if not findings:
        return {
            "summary": "No clear lab values were detected in the uploaded report.",
            "highlights": ["Please upload a clearer PDF or image with readable lab values."],
            "next_steps": ["Ask for a summary of a specific test, or consult your clinician for interpretation."],
        }

    highlights = []
    next_steps = []
    for name, value, range_ in findings:
        low, high = range_
        if value < low:
            highlights.append(f"{name.title()} is below the usual range ({value} vs expected around {low}-{high}).")
            next_steps.append(f"Discuss the low {name.title()} result with a clinician.")
        elif value > high:
            highlights.append(f"{name.title()} is above the usual range ({value} vs expected around {low}-{high}).")
            next_steps.append(f"Discuss the high {name.title()} result with a clinician.")
        else:
            highlights.append(f"{name.title()} appears within the usual range ({value}).")

    summary = (
        "I reviewed the detected values and found a mix of results. "
        "This app can flag possible concerns, but it cannot diagnose disease."
    )
    return {"summary": summary, "highlights": highlights, "next_steps": next_steps}


def build_lab_trends(findings):
    rows = []
    for name, value, range_ in findings:
        low, high = range_
        status = "normal"
        if value < low:
            status = "low"
        elif value > high:
            status = "high"
        rows.append((name.title(), value, f"{low}-{high}", status))
    return rows


def build_follow_up_questions(findings, symptoms):
    questions = []
    if findings:
        questions.append("Which result is most concerning, and has it been repeated on a follow-up test?")
    if symptoms:
        questions.append("Do these symptoms come and go, or are they persistent?")
    questions.append("Would you like me to summarize the report in plain language or explain one lab value in more detail?")
    return questions


def get_openai_client(api_key):
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def get_ai_response(question, report_text, findings, symptoms, chat_history, api_key):
    client = get_openai_client(api_key)
    if not client:
        return None

    found_names = [name for name, _, _ in findings[:5]]
    found_text = ", ".join(found_names) if found_names else "no obvious lab values"
    symptom_text = ", ".join(symptoms) if symptoms else "none mentioned"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful medical education assistant. Explain results in plain English, avoid diagnosing disease, "
                "and remind the user to discuss findings with a qualified clinician."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Here is the uploaded report context. Report text excerpt: {report_text[:5000]}. "
                f"Detected lab names: {found_text}. Detected symptoms: {symptom_text}."
            ),
        },
    ]

    for role, content in chat_history:
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"AI response could not be generated: {exc}"


def get_rule_based_response(question, findings, symptoms):
    q = question.lower()
    if "diagnosis" in q:
        return "I cannot provide a diagnosis. I can help summarize results and suggest that you discuss abnormal findings with a qualified clinician."
    if "normal" in q or "range" in q:
        return "I can compare detected values with common reference ranges. If you share the test name, I can explain whether it appears high, low, or within range."
    if findings:
        names = [f[0].title() for f in findings[:3]]
        return f"The uploaded report appears to include {', '.join(names)}. I can explain these values and suggest follow-up questions for a clinician."
    if symptoms:
        return f"I detected possible symptoms such as {', '.join(symptoms[:3])}. I can help you organize them for a clinician discussion."
    return "Please upload a PDF or image with readable results so I can help interpret them."


st.sidebar.header("OpenAI API key")
st.sidebar.caption("Paste your OpenAI API key here to enable richer AI chat responses.")
api_key = st.sidebar.text_input("API key", type="password", key="openai_api_key", help="Used only for the current session")
st.sidebar.info("Leave it blank to use the built-in guidance without AI.")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or medical image",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded_file is not None:
    with st.spinner("Reading report..."):
        text = extract_report_text(uploaded_file, api_key=api_key)

    if text.startswith("OCR unavailable"):
        st.warning(text)

    findings = parse_lab_values(text)
    symptoms = detect_symptoms(text)
    analysis = interpret_findings(findings)
    lab_trends = build_lab_trends(findings)
    follow_up_questions = build_follow_up_questions(findings, symptoms)

    if "current_report_text" not in st.session_state or st.session_state.current_report_text != text:
        st.session_state.current_report_text = text
        st.session_state.chat_history = []

    st.subheader("Clinical summary")
    st.write(analysis["summary"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detected labs", len(findings))
    with col2:
        st.metric("Detected symptoms", len(symptoms))
    with col3:
        st.metric("Upload type", uploaded_file.type.split("/")[0].title())

    tab_overview, tab_chat = st.tabs(["Overview", "Assistant chat"])

    with tab_overview:
        st.subheader("Symptoms")
        if symptoms:
            for symptom in symptoms:
                st.write(f"- {symptom}")
        else:
            st.info("No obvious symptoms were detected in the uploaded report.")

        st.subheader("Lab trends")
        if lab_trends:
            for name, value, expected, status in lab_trends:
                st.write(f"- {name}: {value} (expected {expected}, status: {status})")
        else:
            st.info("No clear lab values were detected.")

        st.subheader("Follow-up questions")
        for question in follow_up_questions:
            st.write(f"- {question}")

    with tab_chat:
        st.subheader("Ask about the report")
        st.write("This chat remembers the uploaded report context while you ask questions.")

        for role, content in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Assistant:** {content}")

        user_question = st.chat_input("Ask about the results, symptoms, or next steps")
        if user_question:
            ai_reply = get_ai_response(user_question, text, findings, symptoms, st.session_state.chat_history, api_key)
            if ai_reply is None:
                ai_reply = get_rule_based_response(user_question, findings, symptoms)
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("assistant", ai_reply))
            st.rerun()

    st.subheader("Extracted text")
    st.text_area("Report content", text[:7000], height=250)

    st.subheader("Interpretation")
    st.write(analysis["summary"])
    st.write("### Key findings")
    for item in analysis["highlights"]:
        st.write("- " + item)
    st.write("### Suggested next steps")
    for item in analysis["next_steps"]:
        st.write("- " + item)

    if api_key:
        st.success("AI chat is enabled.")
    else:
        st.info("Add an OpenAI API key in the sidebar for richer AI chat responses.")

    st.warning(DISCLAIMER)
else:
    st.info("Upload a PDF or image file to get started.")
