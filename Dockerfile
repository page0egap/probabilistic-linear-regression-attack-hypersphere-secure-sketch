FROM ubuntu:22.04

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

COPY . /app 

WORKDIR /app
RUN uv sync

# CMD RUN uv jupyter notebook
EXPOSE 8888
CMD ["uv", "run", "jupyter", "notebook" , "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]