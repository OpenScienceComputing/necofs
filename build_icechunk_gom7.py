#!/usr/bin/env python
"""
Build Icechunk store for NECOFS GOM7 forecast (single NetCDF3 file).

No subchunking, no Dask, no Coiled — one virtual dataset written to Icechunk.

Run:
    export $(grep -v '^#' ~/dotenv/gom3_forecast.env | xargs) && \
    conda run -n protocoast-notebook python build_icechunk_gom7.py
"""
import os

import icechunk
import numpy as np
import xarray as xr
from dotenv import load_dotenv
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import S3Store
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import NetCDF3Parser

load_dotenv(os.path.expanduser('~/dotenv/gom3_forecast.env'))

BUCKET     = 'neracoos-necofs-forecast'
REGION     = 'us-east-1'
SOURCE_KEY = 'GOM7/NECOFS_GOM7_2026_05_20.nc'
IC_PREFIX  = 'GOM7/icechunk/gom7_forecast.icechunk'
SOURCE_URL = f's3://{BUCKET}/{SOURCE_KEY}'

# ── UGRID / CF metadata ───────────────────────────────────────────────────────
CF_VAR_ATTRS = {
    'time':     {'standard_name': 'time'},
    'h':        {'standard_name': 'sea_floor_depth_below_geoid', 'units': 'm',
                 'coordinates': 'lat lon'},
    'zeta':     {'standard_name': 'sea_surface_height_above_geoid', 'units': 'meters',
                 'coordinates': 'time lat lon', 'coverage_content_type': 'modelResult'},
    'temp':     {'standard_name': 'sea_water_potential_temperature',
                 'coordinates': 'time siglay lat lon', 'coverage_content_type': 'modelResult'},
    'salinity': {'standard_name': 'sea_water_salinity', 'units': '0.001',
                 'coordinates': 'time siglay lat lon', 'coverage_content_type': 'modelResult'},
    'u':        {'standard_name': 'eastward_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time siglay latc lonc', 'coverage_content_type': 'modelResult'},
    'v':        {'standard_name': 'northward_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time siglay latc lonc', 'coverage_content_type': 'modelResult'},
    'ww':       {'standard_name': 'upward_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time siglay latc lonc', 'coverage_content_type': 'modelResult'},
    'ua':       {'standard_name': 'barotropic_eastward_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time latc lonc', 'coverage_content_type': 'modelResult'},
    'va':       {'standard_name': 'northward_barotropic_sea_water_velocity', 'units': 'meters s-1',
                 'coordinates': 'time latc lonc', 'coverage_content_type': 'modelResult'},
    'siglay':   {'standard_name': 'ocean_sigma_coordinate', 'positive': 'up',
                 'valid_min': -1.0, 'valid_max': 0.0,
                 'formula_terms': 'sigma: siglay eta: zeta depth: h'},
    'nv':       {'long_name': 'nodes surrounding element',
                 'cf_role': 'face_node_connectivity', 'start_index': 1},
}

mesh_topology_ds = xr.Dataset({'mesh_topology': xr.Variable((), np.int32(0), attrs={
    'cf_role':               'mesh_topology',
    'topology_dimension':    2,
    'node_coordinates':      'lon lat',
    'face_coordinates':      'lonc latc',
    'face_node_connectivity': 'nv',
    'face_dimension':        'nele',
})})


def add_ugrid_metadata(ds):
    ds.attrs['Conventions'] = 'CF-1.11, UGRID-1.0'
    for var, attrs in CF_VAR_ATTRS.items():
        if var in ds:
            ds[var].attrs.update(attrs)
    for var in ds.data_vars:
        dims = ds[var].dims
        if 'node' in dims or 'nele' in dims:
            ds[var].attrs.setdefault('mesh', 'mesh_topology')
            ds[var].attrs.setdefault('location', 'face' if 'nele' in dims else 'node')
    return xr.merge([ds, mesh_topology_ds], compat='override', combine_attrs='no_conflicts')


# ── Build virtual dataset ─────────────────────────────────────────────────────
print(f'Opening virtual dataset from {SOURCE_URL} ...')
obstore = S3Store.from_url(
    f's3://{BUCKET}',
    config={
        'access_key_id':     os.environ['AWS_ACCESS_KEY_ID'],
        'secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY'],
        'region':            REGION,
    },
)
registry = ObjectStoreRegistry({f's3://{BUCKET}': obstore})
parser   = NetCDF3Parser(skip_variables=['Itime', 'Itime2', 'Times', 'file_date', 'iint', 'nprocs'])
vds = open_virtual_dataset(
    SOURCE_URL,
    registry=registry,
    parser=parser,
    loadable_variables=['time'],
)
vds = add_ugrid_metadata(vds)
print(vds)

# ── Create Icechunk repo ──────────────────────────────────────────────────────
print(f'\nCreating Icechunk repo at s3://{BUCKET}/{IC_PREFIX} ...')
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        url_prefix=f's3://{BUCKET}/',
        store=icechunk.s3_store(region=REGION),
    ),
)
storage = icechunk.s3_storage(
    bucket=BUCKET, prefix=IC_PREFIX, region=REGION,
    access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
)
repo    = icechunk.Repository.open_or_create(storage, config)
session = repo.writable_session('main')

# ── Write and commit ──────────────────────────────────────────────────────────
print('Writing virtual dataset to Icechunk ...')
vds.vz.to_icechunk(session.store)
snap = session.commit('NECOFS GOM7 forecast — single virtual dataset, UGRID/CF-1.11')
print(f'Done!  snapshot={snap}')
print(f'Store: s3://{BUCKET}/{IC_PREFIX}')
