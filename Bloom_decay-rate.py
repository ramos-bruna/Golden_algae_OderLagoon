#!/usr/bin/env python

import sys
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import glob

import math

import numpy as np
from datetime import timedelta

import xarray as xr

from operator import attrgetter

import matplotlib.pyplot as plt

from parcels import (AdvectionRK4,
                     FieldSet,
                     Field,
                     Variable,
                     JITParticle,
                     ParticleSet,
                     ParcelsRandom,
                     ParticleFile)

datapath_in = '/silod7/ramos/Oder/MOM_ERGOM/'
datapath_out = '/silod7/ramos/Golden/'

velfilenames_U = [
    datapath_in + '20220101.ocean_month_2022_07.nc',
    datapath_in + '20220101.ocean_month_2022_08.nc'
]

velfilenames_V = [
    datapath_in + '20220101.ocean_month_2022_07.nc',
    datapath_in + '20220101.ocean_month_2022_08.nc'
]

trajfilename = os.path.join(datapath_out, 'MOM_Decay_2022-Cgrid-5days_mais_particulas.zarr')

gridtype = 'C' #MOM Arakawa-C grid

half_life_days = 5.0
half_life_seconds = half_life_days * 24 * 3600  # convert days to seconds
decay_rate = np.log(2) / half_life_seconds

def create_fieldset(gridtype):

    if gridtype == 'C':
        filenames = {
            'U': velfilenames_U,
            'V': velfilenames_V
        }

        variables = {'U': 'u', 'V': 'v'}

        dimensions = {
            'U': {'lon': 'xq', 'lat': 'yh', 'depth': 'zl', 'time': 'time'},
            'V': {'lon': 'xh', 'lat': 'yq', 'depth': 'zl', 'time': 'time'}
        }

        fieldset = FieldSet.from_netcdf(
            filenames, variables, dimensions,
            mesh='spherical',
            allow_time_extrapolation=False,
            deferred_load=True
        )

    else:
        raise ValueError("Only 'C' grid supported in this script.")

    # Add decay constant
    fieldset.add_constant('decay_rate', decay_rate)

    return fieldset


fieldset = create_fieldset(gridtype)

# =====================================================================================
# DECAY KERNEL
# =====================================================================================

def decay_kernel(particle, fieldset, time):
    dt_sec = math.fabs(particle.dt)
    particle.concentration *= math.exp(-fieldset.decay_rate * dt_sec)
    if particle.concentration < 1e-6:
        particle.concentration = 0.0

# =====================================================================================
# DISTANCE KERNEL
# =====================================================================================

def TotalDistance(particle, fieldset, time):
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2

    lon_dist = (
        (particle.lon - particle.prev_lon)
        * 1.11e2
        * math.cos(particle.lat * math.pi / 180.0)
    )

    particle.distance += math.sqrt(lon_dist**2 + lat_dist**2)

    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat

x = fieldset.U.grid.lon
y = fieldset.U.grid.lat

cell_areas = Field(name="cell_areas", data=fieldset.U.cell_areas(), lon=x, lat=y)
fieldset.add_field(cell_areas)

fieldset.add_constant("Cs", 0.1)

# =====================================================================================
# PARTICLE CLASS
# =====================================================================================

initial_concentration = 150000000 #150 million nr/L

extra_vars = [
    Variable("distance", initial=0.0, dtype=np.float32),
    Variable("prev_lon", dtype=np.float32, to_write=False, initial=attrgetter("lon")),
    Variable("prev_lat", dtype=np.float32, to_write=False, initial=attrgetter("lat")),
    Variable("concentration", initial=initial_concentration, dtype=np.float32),
]

AlgaeParticle = JITParticle.add_variables(extra_vars)

# ============================================================
# LOAD RELEASE TIMES FROM .dat FILE
# ============================================================
n_releases = 240
# # Model start date (known)
t0 = np.datetime64("2022-07-01T00:00")

# Interval between releases
release_interval = np.timedelta64(3, "h")  

# Release start date
release_start = np.datetime64("2022-07-20T00:00")
release_times_datetime = release_start + np.arange(n_releases) * release_interval
release_times_seconds = (release_times_datetime - t0) / np.timedelta64(1, "s")
release_times_seconds = release_times_seconds.astype(int)

release_file = "/silod7/ramos/Golden/release_times2022.dat"  # <-- change this path!

with open(release_file, "r") as f:
    RELEASE_TIMES_SECONDS = [int(line.strip()) for line in f if line.strip()]

# ============================================================
# RELEASE AREA
# ============================================================

release_lonmin = 14.56
release_lonmax = 14.60

release_latmin = 53.60
release_latmax = 53.65

release_dx = 0.004 

# Create grid coordinates
lon_vals = np.arange(
    release_lonmin,
    release_lonmax + release_dx,
    release_dx
)

lat_vals = np.arange(
    release_latmin,
    release_latmax + release_dx,
    release_dx
)

# 2D grid
lons, lats = np.meshgrid(lon_vals, lat_vals)

# Flatten to 1D particle positions
release_lons = lons.flatten()
release_lats = lats.flatten()

# Number of spatial release points
release_npos = release_lons.size

# ============================================================
# COMBINE SPACE + TIME RELEASES
# ============================================================

particle_lons = np.repeat(release_lons, n_releases)
particle_lats = np.repeat(release_lats, n_releases)

particle_times = np.tile(
    release_times_seconds,
    release_npos
)


# ============================================================
# PARTICLES PER RELEASE
# ============================================================

N_TOTAL = len(particle_times)

pset = ParticleSet.from_list(
    fieldset=fieldset,
    pclass=AlgaeParticle,
    lon=particle_lons,
    lat=particle_lats,
    time=particle_times
)

# total integration time in hours
tint = timedelta(hours=753) #(run for 1 month) # simulation runtime

# set frequency with which to output particle data in minutes
outdt = timedelta(hours=1) # write output every 

# integration time step in minutes
intdt = timedelta(minutes=3)  #seconds timestep

kernels = [AdvectionRK4, decay_kernel, TotalDistance]

pfile = ParticleFile(trajfilename, pset, outputdt=outdt)

ParcelsRandom.seed(1636)  # Random seed for reproducibility

pset.execute(
    kernels,
    runtime=tint,
    dt=intdt,
    output_file=pfile
)

