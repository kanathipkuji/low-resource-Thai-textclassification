# Use the official Miniconda3 base image
FROM continuumio/miniconda3:latest

# # Set environment variables for non-interactive and UTF-8 encoding
# ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

# Set the working directory in the container
WORKDIR /app

# Copy the Conda environment (including notebooks) into the container
COPY environment.yml /app/

# Create and activate the Conda environment
RUN conda env create -f environment.yml

# Expose no ports (since we won't run Jupyter Notebook inside the container)

# Start a shell (bash) upon container launch
CMD ["/bin/bash"]
