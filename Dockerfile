# Use TensorFlow GPU base image (supports both GPU and CPU)
FROM tensorflow/tensorflow:2.10.0-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the local files into the container
COPY . /app

# Install required Python packages (if needed)
# Add additional requirements here if necessary (e.g., specific version of TensorFlow, other libraries)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir tensorflow keras

# Expose a port if needed (not needed here but good practice for APIs or dashboards)
# EXPOSE 8888

# Define the command to run your TensorFlow script
CMD ["python", "cifar.py"]
