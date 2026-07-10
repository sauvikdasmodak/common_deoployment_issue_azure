# Azure Deployment Issue Analyzer

This project is a Streamlit app that analyzes Azure deployment logs with a Groq or OpenAI LLM and returns structured JSON findings.

## What it does

- Accepts logs by paste, file upload, or built-in sample scenarios
- Sends logs to a selected Groq or OpenAI model for analysis
- Detects and categorizes deployment issues (critical/warning/info/resolved)
- Shows root cause, recommended fix, and relevant log snippets per issue
- Provides overall recommendations and a deployment health score
- Supports exporting the full analysis as JSON

## Requirements

- Python 3.9+
- A Groq API key or OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file (or export env vars) with:

```env
GROQ_API_KEY=your_groq_api_key
# or
OPENAI_API_KEY=your_openai_api_key
```

You can copy from `.env.example`.

## Run

Quick start:

```bash
./start.sh
```

Or run directly:

```bash
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.
