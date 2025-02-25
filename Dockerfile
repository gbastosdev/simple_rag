# Use the official Ollama image
FROM ollama/ollama:latest

COPY ./shell-ollama.sh /tmp/shell-ollama.sh

WORKDIR /tmp

RUN chmod +x shell-ollama.sh \
    && ./shell-ollama.sh

# Expose the Ollama API port
EXPOSE 11434


