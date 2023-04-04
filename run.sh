export FLASK_APP=app.py

gunicorn --workers 1 -k sync --timeout 300 --graceful-timeout 600 --bind 0.0.0.0:7860 app:app