# Base image
FROM python:3.8.1

# Put all files in working directory
WORKDIR /simulations-neural

# Copy requirements into working directory
COPY ./data ./data
COPY requirements.txt .

# Create a virtual enviroment, activate and run the requirements
RUN python3 -m venv env
RUN . env/bin/activate
RUN pip install -r requirements.txt

# Copy all the absolute files into the working directory
COPY prototype_nn.py .
COPY prototype.py .
COPY simulations.py .
COPY help.txt .

# Copy relative files into the working directory
COPY ./plots ./plots
COPY ./models ./models
COPY ./average ./average


# RUN apt-get update
CMD ["/bin/bash"]
