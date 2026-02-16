################################################################################
# LISFLOOD-FP v8.1 + WhiteboxTools v2.4.0 Dockerfile
#
# Description:
#   Builds and configures LISFLOOD-FP v8.1.0 and WhiteboxTools v2.4.0 inside
#   an Ubuntu 22.04 container. Provides CLI access to both tools.
#
# Features:
#   - LISFLOOD-FP compiled with OpenMP, NetCDF, and NUMA support
#   - LISFLOOD-FP executable in /usr/local/bin as 'lisflood' and symlink 'lisflood_fp'
#   - WhiteboxTools installed as 'whitebox_tools' and symlink 'wbt'
#   - Default working directory: /data
################################################################################

# Base image
FROM ubuntu:22.04

# Non-interactive for apt
ENV DEBIAN_FRONTEND=noninteractive

# Compiler flags
ENV CXXFLAGS="-O2 -fopenmp"
ENV LDFLAGS="-fopenmp -lnuma -lnetcdf"

# Install build dependencies including CMake and Whitebox dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    make \
    wget \
    unzip \
    ca-certificates \
    cmake \
    libomp-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    libnuma-dev \
    libglib2.0-0 \
    curl \
    gdal-bin \
	netcdf-bin \
	nano \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /opt

# Download LISFLOOD-FP source
RUN wget -O lisflood-fp.zip \
    "https://zenodo.org/records/6912932/files/LISFLOOD-FP%20v8.1.zip?download=1" \
    && unzip lisflood-fp.zip \
    && rm lisflood-fp.zip

# Set LISFLOOD-FP source directory
WORKDIR /opt/LISFLOOD-FP-trunk

# Build LISFLOOD-FP with CMake
RUN rm -rf build && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_Fortran_FLAGS="-fopenmp" \
        -DCMAKE_EXE_LINKER_FLAGS="-fopenmp -lnuma -lnetcdf" \
    && make -j$(nproc)

# Copy LISFLOOD-FP executable to PATH
RUN cp build/lisflood /usr/local/bin/lisflood
RUN chmod +x /usr/local/bin/lisflood
RUN ln -s /usr/local/bin/lisflood /usr/local/bin/lisflood_fp

# -------------------------------
# Install WhiteboxTools CLI
# -------------------------------
WORKDIR /opt

# Download and unzip WhiteboxTools Linux AMD64
RUN curl -L -o whitebox.zip https://www.whiteboxgeo.com/WBT_Linux/WhiteboxTools_linux_amd64.zip \
    && unzip whitebox.zip -d whitebox \
    && rm whitebox.zip \
    && echo "Listing extracted files:" \
    && ls -lR whitebox

# Move the WBT folder to a permanent location
RUN mv /opt/whitebox/WhiteboxTools_linux_amd64/WBT /opt/whitebox_tools \
    && chmod -R +x /opt/whitebox_tools/whitebox_tools

# Create a launcher in /usr/local/bin
RUN printf '#!/bin/bash\n/opt/whitebox_tools/whitebox_tools "$@"\n' > /usr/local/bin/whitebox \
    && chmod +x /usr/local/bin/whitebox \
    && ln -sf /usr/local/bin/whitebox /usr/local/bin/wbt

# -------------------------------
# Install Miniconda 3.11 and set up default geospatial environment
# -------------------------------
WORKDIR /opt

# Install Miniconda with Python 3.11
RUN curl -L -o Miniconda3-py311.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_25.11.1-1-Linux-x86_64.sh \
    && bash Miniconda3-py311.sh -b -p /opt/miniconda \
    && rm Miniconda3-py311.sh

# Add conda to PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# Accept Anaconda repository Terms of Service non-interactively
RUN conda config --set always_yes yes --set changeps1 no \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create 'geo' environment with Python 3.11
RUN conda create -y -n geo python=3.11 \
    && conda clean -afy

# Activate geo as default for all subsequent RUN commands
SHELL ["conda", "run", "-n", "geo", "/bin/bash", "-c"]

# Install Python geospatial packages with pip inside 'geo' environment
RUN conda install -n geo -c conda-forge \
    python=3.11 \
    numpy pandas geopandas \
    rasterio rioxarray \
    shapely fiona pyogrio \
    netCDF4 \
    gdal proj proj-data pyproj \
    && conda clean -afy
	
# Then you can still install whitebox via pip if you want
RUN pip install --no-cache-dir whitebox

# Trigger WhiteboxTools first-run initialization during build
RUN python -c "import whitebox; whitebox.WhiteboxTools(); print('WhiteboxTools initialized')"

# Verify installation
RUN python -c "import numpy, pandas, geopandas, rasterio, rioxarray, shapely, fiona, pyogrio, netCDF4; import whitebox; print('Geo environment OK')"

# Initialize conda for bash interactive shells
RUN /opt/miniconda/bin/conda init bash

# -------------------------------
# Revert to normal shell for system commands
# -------------------------------
SHELL ["/bin/bash", "-c"]

# Install git system-wide for cloning repositories
RUN apt-get update && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Clone the lisflood2fim repository
RUN git clone https://github.com/andycarter-pe/lisflood2fim.git /app/lisflood2fim

# -------------------------------
# Set default working directory for simulations
# -------------------------------
WORKDIR /app/lisflood2fim/src

# -------------------------------
# Switch back to Conda environment shell for Python commands
# -------------------------------
SHELL ["conda", "run", "-n", "geo", "/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]