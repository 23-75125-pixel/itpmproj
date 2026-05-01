# SEMCDS

Smart Exam Management & Cheating Detection System prototype built with Flask.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The app uses SQLite locally by default. The database file is `database/semcds.db` unless `DB_PATH` is overridden.

To run it with the Flask CLI instead, use:

```powershell
flask --app wsgi run --debug
```

## Optional Supabase Setup

1. Set `DATABASE_BACKEND=supabase` in your `.env` file.
2. Create a Supabase project.
3. Run [database/supabase_schema.sql](database/supabase_schema.sql) in the SQL editor.
4. Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in your `.env` file.
5. Restart the Flask app.

## Render Deploy

1. Push the repo to GitHub.
2. In Render, create a new Blueprint service from this repo.
3. Render will pick up [render.yaml](render.yaml) and create a Python web service.
4. In Render, fill in the missing environment variables:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`
   - `GOOGLE_REDIRECT_URI`
   - `SMTP_HOST`
   - `SMTP_USER`
   - `SMTP_PASSWORD`
   - `SMTP_FROM`
   - `EMAIL_DELIVERY_PROVIDER`
   - `EMAIL_FROM`
   - `EMAIL_FROM_NAME`
   - `BREVO_API_KEY` and/or `RESEND_API_KEY`
   - `OPENAI_API_KEY` and/or `GEMINI_API_KEY` if you use AI quiz generation
5. Restart the service after saving the environment variables.

The Render blueprint is configured for a single instance with `eventlet` because live quiz monitoring uses Socket.IO and in-memory room state.
If you deploy on a free Render instance, outbound SMTP can be blocked. In that case, set `EMAIL_DELIVERY_PROVIDER=brevo`, `BREVO_API_KEY`, `EMAIL_FROM`, and optional `EMAIL_FROM_NAME` to send invitations and password reset links over the Brevo HTTPS API instead of SMTP. Resend is still supported as an alternative with `EMAIL_DELIVERY_PROVIDER=resend`, `RESEND_API_KEY`, and `EMAIL_FROM`.

## Login

- Instructor: `/signin/admin`
- Student: `/signin/user`

## Notes

- The app uses a Flask application factory in `src/app.py`.
- `DATABASE_BACKEND=sqlite` keeps data local in SQLite and still supports the app's live Socket.IO monitoring.
- `DATABASE_BACKEND=supabase` switches quiz data reads and writes to Supabase.
- New databases no longer seed default users. Create your first instructor account manually.
- Base44 login remains a placeholder; the AI question generation and quiz workflow are wired into the Flask routes.
