# Use the official Python image from Docker Hub
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    firefox-esr


# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Download and install geckodriver
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz && \
    tar -xvzf geckodriver-v0.34.0-linux64.tar.gz && \
    chmod +x geckodriver && \
    mv geckodriver /usr/local/bin/

# Clean up
RUN rm geckodriver-v0.34.0-linux64.tar.gz

# copy entrypoint.sh
# COPY ./entrypoint.sh .
# Ensure that the entrypoint script is executable
# RUN sed -i 's/\r$//g'  /app/entrypoint.sh 
# RUN chmod +x /app/entrypoint.sh

# Copy the rest of the application code into the container
COPY . .



# Expose the port that Streamlit will run on
EXPOSE 8501

# ENTRYPOINT ["./entrypoint.sh"]

# Run the Python script within the db directory
# RUN cd db && python3 vector_store.py --create_db true

# Command to run the Streamlit app
# CMD ["streamlit", "run", "start.py"]
CMD ["sh", "-c", "cd db && python3 vector_store.py --create_db true && cd .. && streamlit run start.py"]