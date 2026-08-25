FROM ubuntu:22.04

# Avoid interactive prompts during apt package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    openjdk-17-jdk-headless \
    g++ \
    nodejs \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory and non-root sandbox user
WORKDIR /sandbox
RUN useradd -m -u 1000 sandboxuser && chown -R sandboxuser:sandboxuser /sandbox

USER sandboxuser

CMD ["/bin/bash"]
