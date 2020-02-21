
FROM cytomineuliege/software-python3-base:latest


RUN apt-get update -y && apt-get install -y git

RUN apt-get install -y openslide-tools python-openslide python3-tk

RUN mkdir -p /CRLM

COPY . /CRLM/

RUN pip install -r /CRLM/requirements.txt

RUN mv /CRLM/settings.py.docker_image /CRLM/settings.py


ENTRYPOINT ["python", "/CRLM/cytomine_add_annotations.py"]
