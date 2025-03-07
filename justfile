# List all available recipes
default:
    @just --list

API_URL := "http://localhost:8000"

# Read API_KEY from .env file
API_KEY := `grep '^API_KEY=' .env | cut -d'=' -f2 | tr -d '"'`

healthcheck:
    @echo "Checking API health..."
    @curl -s {{API_URL}}/healthcheck | jq

train:
    @echo "Training the model..."
    @curl -s -X POST {{API_URL}}/train \
        -H "X-API-Key: {{API_KEY}}" | jq

predict steps:
    @if [ -z "{{steps}}" ]; then \
        echo "Error: steps argument is required"; \
        exit 1; \
    fi
    @echo "Making predictions for {{steps}} steps..."
    @curl -s -X POST {{API_URL}}/predict \
        -H "Content-Type: application/json" \
        -H "X-API-Key: {{API_KEY}}" \
        -d "{\"steps\": {{steps}}}" | jq
