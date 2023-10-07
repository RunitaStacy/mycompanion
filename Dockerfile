# Use an official Windows-based Python 3.11 image
FROM python:3.11

# Set the working directory to C:\app
WORKDIR C:\app

# Copy the current directory contents into the container at C:\app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow
RUN pip install tensorflow

# Expose port 8501
EXPOSE 8501

# Define the command to run your application
CMD ["streamlit", "run", "application.py"]
