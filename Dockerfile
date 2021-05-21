# Base image as python
FROM python:3.8

# Set Environment
ENV PYTHONUNBUFFERED 1

# Copy contents into container
COPY . . /application

# Set working directory
WORKDIR /application

# Install necessary packages
RUN pip install Flask scikit-learn scipy pandas


# CMD commands for executing application
CMD ["python" , "./app.py"]