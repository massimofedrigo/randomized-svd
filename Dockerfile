FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir -e .[dev]

COPY . .

CMD ["pytest"]