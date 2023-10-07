# Use an official Python 3.11 image as a parent image
FROM python:3.11

# Set the working directory to /app
WORKDIR /app

# Install system-level dependencies for OpenGL support and other libraries
RUN apt-get update && apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install TensorFlow
RUN pip install tensorflow

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run your application
CMD ["streamlit", "run", "application.py"]
