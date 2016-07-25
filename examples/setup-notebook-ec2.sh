#!/usr/bin/env bash

cd /root
mkdir certificates
cd certificates
openssl req -x509 -nodes -days 365 -subj "/C=XX/ST=XX/L=XX/O=XX/CN=XX" -newkey rsa:1024 -keyout mykey.key -out mycert.pem
jupyter notebook --generate-config
python -c "from notebook.auth import passwd; print passwd()" > /root/.jupyter/nbpasswd.txt
echo "# Configuration file for jupyter-notebook.
c = get_config()
# Notebook config
c.NotebookApp.certfile = u'/root/certificates/mycert.pem'
c.NotebookApp.keyfile = u'/root/certificates/mykey.key'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
# It is a good idea to put it on a known, fixed port
c.NotebookApp.port = 8888
PWDFILE='/root/.jupyter/nbpasswd.txt'
c.NotebookApp.password = open(PWDFILE).read().strip()" >> /root/.jupyter/jupyter_notebook_config.py


echo -e "\n\n\n"
echo "-----------------------------------------------------------------"
echo "Jupyter notebook successfully set up!"
echo "-----------------------------------------------------------------"
echo -e "\n\n\n"
