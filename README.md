# Lisflood2FIM <img src="doc/Logo_CWE.png" align="right" alt="lisflood2fim agency" height="80"> <br> <br>
## <i>Lisflood2FIM - Flood Inundation Mapping using LISFLOOD-FP</i>

<img src="/doc/lisflood2fim-logo-20260209.png" align="right"
     alt="lisflood2fim logo" width="160" height="160">

**Description**:  LISFLOOD2FIM creates Flood Inundation Maps (FIMs) indexed to various excess rainfall rates for a given catchment (watershed). The initial routine determines the stream network on which to apply the excess precipitation. This requires hydro-enforcement through dams and roadways. Input data needed for LISFLOOD-FP is then created, including parameters, terrain, boundary conditions, and stream-centric rainfall..<br><br>
Simulations are run using LISFLOOD-FP v8.1.0 to a point of “stability,” where inflow rainfall equals the outflow rate. This process is repeated for various excess rainfall intensities. The resulting “stable” flood depths for each intensity are aggregated into a single data cube (NetCDF), representing flood depth across multiple intensities over the watershed. <br><br>
These scripts were developed in support of the National Weather Service (Research Project NA22NWS4320003 / A25-0366-S018).

<img src="/doc/lisflood2fim.gif" align="center"
     alt="sample cross section" width="55%">

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
