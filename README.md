# Lisflood2FIM <img src="doc/Logo_CWE.png" align="right" alt="lisflood2fim agency" height="80"> <br> <br>
## <i>Lisflood2FIM - Flood Inundation Mapping using LISFLOOD-FP</i>

<img src="/doc/lisflood2fim-logo-20260209.png" align="right"
     alt="lisflood2fim logo" width="160" height="160">

**Description**:  LISFLOOD2FIM creates flood inundation maps (FIM) indexed to various excess rainfall rates for a given catchment (watershed).  The inital routine determines the stream network on which to apply the excess precipitation.  This requires hydroenforecment buring through dams and roadways.  Then input data needed for LISFLOOD-FP is created including paramters, terrain, boundary conditions and stream centeric rainfall.<br>
The runs are simulated with LISFLOOD-FP v8.1.0 to a point of 'stability' where the inflow rainfall is equal to the outflow rate.  This is run for various excess rainfall initensities.  The 'stable' flood depths for each intensity and aggregated into a single datacube (netCDF) respresenting the depth of flooding for multiple intensities over a given watershed. <br>
These scripts were created in support of the National Weather Service (Research Project NA22NWS4320003 / A25-0366-S018) research project.


<img src="/doc/lisflood2fim.gif" align="center"
     alt="sample cross section" width="60%">

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
