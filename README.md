# LISFLOOD2FIM <img src="doc/Logo_CWE.png" align="right" alt="lisflood2fim agency" height="80"> <br> <br>
## <i>Texas Bridge - Flood Warning Cross Section Web Server </i>

<img src="/doc/tx-bridge-logo-20220517.png" align="right"
     alt="tx-bridge logo" width="160" height="160">

**Description**:  A database of simplified bridge geometry was extracted from statewide LiDAR and DEMs for the State of Texas.  From the bridge database, it was necessary to produce interactive graphics estimating water depths for more than 19,000 bridges across Texas on an hourly basis, providing an eighteen-hour advance prediction.  These graphics are used to indicate instances when a bridge's superstructure, including beams and deck, may be at risk of flooding. <br><br>
Each cross section is cross-referenced with both the National Bridge Inventory database and the National Weather Service’s National Water Model (NWM) stream reach. Anticipated stream flows from the National Water Model are then converted into predictions of flood depths.

<img src="/doc/sample_xs.png" align="center"
     alt="sample cross section" width="100%">

  - **Technology stack**: Scripts were all developed in Python 3.11<br><br>
  - **Status**:  Version 0.1- Preliminary release. (2026.02.09)<br><br>
  - **Related Projects**: Bridge database was created using the TX-Bridge repository.  https://github.com/andycarter-pe/ras2fim-2d<br>
  
## Dockerfile
To build a container from this repository, clone to your local drive and build with the following command
```
docker build -t lisflood2fim .
```

## Docker Container
For convience, a container has been pre-built and pushed to DockerHub.  To pull this container to your machine...
```
docker pull civileng127/lisflood2fim:20260209
```
To run this container, use the following command
```
docker run -e AWS_ACCESS_KEY_ID=<your_access_key> -e AWS_SECRET_ACCESS_KEY=<your_secret_access-key> -p 5000:5000 civileng127/tx-bridge-xs:20240223
```
Note that this container will need to have your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as command line environment inputs.  Additionally, the Dockerfile/Docker Container
has an environmental variable for the bucket containing the bridge jsons ... ENV PATH_TO_BRIDGE_JSONS="s3://tx-bridge-xs-json/"

## Misc
Before initializing the web server, a JSON file is generated for each bridge and stored in an S3 bucket. 
This process is facilitated by a script called 'create_bridge_json_files.py', located in the misc folder. 
The script converts the bridge SQLite database, generated using 'TX-Bridge' for the State of Texas, into separate JSON files for each UUID.
Currently, the database is hosted on S3 at the location s3://txbridge-data/test-upload/tx-bridge-geom.sqlite.
