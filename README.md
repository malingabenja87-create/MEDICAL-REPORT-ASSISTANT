# Medical Report Assistant

This app lets users upload a PDF or image of medical results and receive a simple educational interpretation.

## Features
- Upload PDF or image files
- Extract text from uploaded reports
- Detect common lab values and symptoms
- Provide a report-aware chat assistant
- Optional OpenAI API key for richer AI responses

## Run locally
```bash
pip install -r requirements.txt
streamlit run CHATBOT.py
```

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file to `CHATBOT.py`.
5. Add the OpenAI API key in Streamlit secrets:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

## Notes
This app is for educational use only and is not a medical diagnosis.
