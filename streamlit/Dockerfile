# this needs to be run from the parent directory
# cmd: docker build -f streamlit/Dockerfile .

# currently latest stable small python image
FROM python:3.9.5-slim-buster

# streamlit runs on this port by default
EXPOSE 8501

WORKDIR /usr/src/app

# install everything needed for the app
COPY streamlit/requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# make logging to console realtime
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py"]