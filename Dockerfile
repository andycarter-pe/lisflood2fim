################################################################################
# LISFLOOD-FP v8.1 + WhiteboxTools v2.4.0 Dockerfile (Mamba + TOS fix)
#
# Features:
#   - LISFLOOD-FP compiled with OpenMP, NetCDF, and NUMA support
#   - LISFLOOD-FP executable in /usr/local/bin
#   - WhiteboxTools installed as /opt/whitebox_tools and symlink wbt
#   - Miniconda3 + mamba for fast geospatial Python environment
################################################################################

# Base image
FROM ubuntu:22.04

# Non-interactive for apt
ENV DEBIAN_FRONTEND=noninteractive

# Compiler flags
ENV CXXFLAGS="-O2 -fopenmp"
ENV LDFLAGS="-fopenmp -lnuma -lnetcdf"

# -------------------------------
# Install system dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    build-essential gfortran make wget unzip ca-certificates cmake \
    libomp-dev libnetcdf-dev libnetcdff-dev libnuma-dev libglib2.0-0 curl \
    gdal-bin netcdf-bin nano git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# -------------------------------
# Build LISFLOOD-FP
# -------------------------------
RUN wget -O lisflood-fp.zip \
    "https://zenodo.org/records/6912932/files/LISFLOOD-FP%20v8.1.zip?download=1" \
    && unzip lisflood-fp.zip && rm lisflood-fp.zip

WORKDIR /opt/LISFLOOD-FP-trunk
RUN rm -rf build && mkdir build && cd build \
    && cmake .. -DCMAKE_Fortran_FLAGS="-fopenmp" \
               -DCMAKE_EXE_LINKER_FLAGS="-fopenmp -lnuma -lnetcdf" \
    && make -j$(nproc)

RUN cp build/lisflood /usr/local/bin/lisflood \
    && chmod +x /usr/local/bin/lisflood \
    && ln -s /usr/local/bin/lisflood /usr/local/bin/lisflood_fp

# -------------------------------
# Install WhiteboxTools CLI
# -------------------------------
WORKDIR /opt
RUN curl -L -o whitebox.zip https://www.whiteboxgeo.com/WBT_Linux/WhiteboxTools_linux_amd64.zip \
    && unzip whitebox.zip -d whitebox && rm whitebox.zip \
    && mv /opt/whitebox/WhiteboxTools_linux_amd64/WBT /opt/whitebox_tools \
    && chmod -R +x /opt/whitebox_tools/whitebox_tools \
    && printf '#!/bin/bash\n/opt/whitebox_tools/whitebox_tools "$@"\n' > /usr/local/bin/whitebox \
    && chmod +x /usr/local/bin/whitebox \
    && ln -sf /usr/local/bin/whitebox /usr/local/bin/wbt

# -------------------------------
# Install Miniconda 3.11
# -------------------------------
WORKDIR /opt
RUN curl -L -o Miniconda3-py311.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_25.11.1-1-Linux-x86_64.sh \
    && bash Miniconda3-py311.sh -b -p /opt/miniconda \
    && rm Miniconda3-py311.sh

ENV PATH="/opt/miniconda/bin:$PATH"

# -------------------------------
# Accept Conda TOS and install Mamba
# -------------------------------
RUN conda config --set always_yes yes --set changeps1 no \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -n base -c conda-forge mamba \
    && conda clean -afy

# -------------------------------
# Create 'geo' environment with Mamba (fast!)
# -------------------------------
RUN mamba create -y -n geo -c conda-forge python=3.11 \
    numpy pandas geopandas rasterio rioxarray shapely fiona pyogrio \
    netCDF4 gdal proj pyproj \
    && mamba clean -afy

# Activate geo for all subsequent RUN commands
SHELL ["conda", "run", "-n", "geo", "/bin/bash", "-c"]

# Optional: Install Whitebox Python package
RUN pip install --no-cache-dir whitebox

# Trigger WhiteboxTools initialization
RUN python -c "import whitebox; whitebox.WhiteboxTools(); print('WhiteboxTools initialized')"

# Verify Python geospatial stack
RUN python -c "import numpy, pandas, geopandas, rasterio, rioxarray, shapely, fiona, pyogrio, netCDF4; import whitebox; print('Geo environment OK')"

# Initialize conda for interactive bash shells
RUN /opt/miniconda/bin/conda init bash

# -------------------------------
# Revert to system shell for system commands
# -------------------------------
SHELL ["/bin/bash", "-c"]

# Clone lisflood2fim repository
RUN git clone https://github.com/andycarter-pe/lisflood2fim.git /app/lisflood2fim

WORKDIR /app/lisflood2fim/src

# Use geo environment by default for Python commands
SHELL ["conda", "run", "-n", "geo", "/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]

# Activate geo environment by default and set working directory
RUN echo "conda activate geo" >> /root/.bashrc
WORKDIR /mnt