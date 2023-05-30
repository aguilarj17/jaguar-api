#Load a Docker Imge for python to an InferenceAPI
FROM python:3.9
#create a work directory /code
WORKDIR /code
#COPY into file requirements.txt and load from the current folder the requeriments.txt
COPY requirements.txt requirements.txt
#Install all libraries
RUN pip install -r requirements.txt
#Copy from this folder all to the folder /code in docker container
COPY . .
#entry point to run python3 scripts
ENTRYPOINT [ "python3" ]
#Name of the script
CMD ["main.py"]