# Use the official Ollama image as the base image
FROM ollama/ollama:latest

# Create a volume for /root/.ollama
VOLUME /root/.ollama

# Expose the Ollama API port (default is 11434)
EXPOSE 11434

# Pull the tinyllama model
RUN ollama pull tinyllama

# Command to run Ollama
CMD ["ollama", "serve"]