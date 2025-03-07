default:

API_KEY := env_var("API_KEY")
API_URL := "http://localhost:8000"

healthcheck:
    @echo "Checking API health..."
    @curl $(API_URL)/healthcheck

train:
    @echo "Training the model..."
    @curl -X POST $(API_URL)/train \
        -H "X-API-Key: $(API_KEY)"

predict steps:
    @if [ -z "{{steps}}" ]; then \
        echo "Error: steps argument is required"; \
        exit 1; \
    fi
    @echo "Making predictions for {{steps}} steps..."
    @curl -X POST $(API_URL)/predict \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $(API_KEY)" \
        -d "{\"steps\": {{steps}}}"
