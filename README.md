# Patient Health Records

A small Flask + SQLite web app for storing patient blood-test results (glucose, haemoglobin, cholesterol) with an AI-generated risk remark for each record.

## Features
- Create, read, update and delete patient records
- Input validation: name characters, `dd/mm/yyyy` date of birth (real calendar date, not in the future), email format, duplicate email check, numeric blood values within plausible ranges
- Persistent storage in SQLite (`patients.db`, created on first run)
- AI integration: after a valid save, the Anthropic Messages API is called and the short risk assessment is stored in **Remarks**. If the call fails or no key is set, the record is still saved and the row shows a **Retry AI** button.
- Remarks are regenerated on edit only when the date of birth or a blood value changed

## Run
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here             # Windows: set ANTHROPIC_API_KEY=your-key-here
python app.py
```
Open http://127.0.0.1:5000

Optional environment variables: `ANTHROPIC_MODEL`, `FLASK_SECRET_KEY`, `DB_PATH`.
The API key is read from the environment only and is never stored in the code or repository.

## Structure
- `app.py`: routes, validation, database access, AI call
- `templates/index.html`: Bootstrap UI (form + records table)

## Note
Remarks are a screening aid produced by an AI model, not a medical diagnosis.
