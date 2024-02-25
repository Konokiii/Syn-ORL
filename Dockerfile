FROM cwatcherw/cql:0.1
WORKDIR /workspace

#COPY requirements/requirements_dzx.txt requirements_dzx.txt
#RUN pip install --no-cache-dir -r requirements_dzx.txt
RUN apt update && apt install tcl


