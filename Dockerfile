FROM python:3.10-bookworm

# Set the maintainer label
LABEL maintainer="phiedulxp@gmail.com"

# create a working directory
RUN mkdir -p /app
WORKDIR /app

COPY ChatTTS /app/ChatTTS
COPY server.py requirements.txt /app/

# Install packages with pip
RUN pip install -r requirements.txt

# make the paths of the nvidia libs installed as wheels visible. equivalent to:
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib"

# Set the default command to activate the environment
CMD ["python", "server.py"]