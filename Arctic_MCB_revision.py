#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings 
warnings.filterwarnings('ignore')
import os
import iris
import iris.analysis.cartography as iac
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.util as cutil
import cftime
import nc_time_axis
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.path as mpath
from scipy import stats
from aprp import APRP
data_dir = '/data/users/matthew.henry/ArcticMCB_data/'
fname = data_dir+'ukesm_land.pp' 
aod_ = iris.load(fname)
land = xr.DataArray.from_iris(aod_[0])
land = land.rename({'latitude':'lat','longitude':'lon'})
land_cesm = xr.open_mfdataset(data_dir+'cesm_lfrac.nc')
refdata_cesm = xr.open_dataset(data_dir+'cesm/refdata.nc')
DSA = xr.open_mfdataset(data_dir+'ne30pg2.nc').rename({'grid_size':'ncol'})
weights_cesm = np.cos(np.deg2rad(refdata_cesm.lat))
weights_cesm.name = "weights"
grid_areas = xr.open_dataarray(data_dir+'grid_areas.nc')
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)


# ##### For UKESM1, Arctic = North of 70N
# 50 Tg/year (nominal) in bin 10 in NO gives -3.8K in coupled simulation\
# Years 2035-44 = 3.1K of cooling → 40 Tg/yr (nominal injection rates)\
# Years 2045-54 = 4.6K of cooling → 70 Tg/yr (nominal injection rates)\
# Years 2055-64 = 5.7K of cooling → 100 Tg/yr (nominal injection rates)\
# Years 2065-74 = 6.5K of cooling → 150 Tg/yr (nominal injection rates)

# In[2]:


#Getting CESM2 data
refdata = xr.open_mfdataset(data_dir+'cesm/cam_gridinfo.nc')

def load_files(path, variables,freq = 'h0', shiftdate = True,ind_var=-2):
    d_files = {var:[] for var in variables}
    # Scans directory for files with specified variables
    for fname in os.listdir(path):
        if not '.nc' in fname:
            continue;
        var = fname.split('.')[ind_var]
        if var in variables and freq in fname:
            d_files[var].append(path + fname)
    # print(d_files)
    # loads datasets for each variable
    l_datasets = [xr.open_mfdataset(d_files[var]) for var in variables]
    # merge dataset (need to do this b/c of issues with the PRECT file time_bnds,
    # which causes an error for some reason)
    out_ds = xr.merge(l_datasets, compat='override')
    # shift date if the data back for CESM2 data
    if shiftdate:
        out_ds['time'] = out_ds.indexes['time'].shift(-15,"D")
    # Remove duplicate time steps
    out_ds = out_ds.sel(time=~out_ds.get_index("time").duplicated())
    out_ds = out_ds.assign_coords(lat=refdata.lat)
    return out_ds

variables = ['TREFHT','ICEFRAC','PRECT','FSNT','FLNT','ncl_a1SF','FSNS','FLNS','SHFLX','LHFLX']
cesm_ssp245_ens = load_files(data_dir+'cesm/ssp245_ensemble_mean/',variables,ind_var=-3)
cesm_ssp245_ens['time'] = cesm_ssp245_ens.indexes['time'].to_datetimeindex()

cesm_mcb_exp = [f'b.e21.BSSP245smbb_MCBsse.Arctic_from2035.manual_ctrl.Arc_3.67K.f09_g17.{ens}'
                  for ens in ['LE2-1011.001','LE2-1031.002','LE2-1051.003']]
cesm_mcb_data = {}
variables = ['TREFHT','ICEFRAC','PRECT','FSNT','FLNT','ncl_a1SF','FSNS','FLNS','SHFLX','LHFLX']
for exp in cesm_mcb_exp:
    cesm_mcb_data[exp] = load_files(data_dir+f'cesm/{exp}/atm/',variables,ind_var=-2)
    cesm_mcb_data[exp]['time'] = cesm_mcb_data[exp].indexes['time'].to_datetimeindex()

cesm_mcb_data_ave=(cesm_mcb_data['b.e21.BSSP245smbb_MCBsse.Arctic_from2035.manual_ctrl.Arc_3.67K.f09_g17.LE2-1011.001']+\
                   cesm_mcb_data['b.e21.BSSP245smbb_MCBsse.Arctic_from2035.manual_ctrl.Arc_3.67K.f09_g17.LE2-1031.002']+\
                   cesm_mcb_data['b.e21.BSSP245smbb_MCBsse.Arctic_from2035.manual_ctrl.Arc_3.67K.f09_g17.LE2-1051.003'])/3


# ### Figure 1

# In[3]:


mcb_emiss_ukesm_amcb = xr.open_dataarray(data_dir+'ukesm/amcb/mcb_emiss_ukesm_amcb_ave.nc')
mcb_emiss_cesm_amcb_1 = xr.open_dataarray(data_dir+'cesm/amcb/mcb_emiss_cesm_amcb_1.nc')
mcb_emiss_cesm_amcb_2 = xr.open_dataarray(data_dir+'cesm/amcb/mcb_emiss_cesm_amcb_2.nc')
mcb_emiss_cesm_amcb_3 = xr.open_dataarray(data_dir+'cesm/amcb/mcb_emiss_cesm_amcb_3.nc')
mcb_emiss_cesm_amcb = (mcb_emiss_cesm_amcb_1+mcb_emiss_cesm_amcb_2+mcb_emiss_cesm_amcb_3)/3
mcb_emiss_e3sm_amcb_1 = xr.open_dataset(data_dir+'e3sm/amcb/mcb_emiss_e3sm_amcb_1.nc').ncl_a1SF
mcb_emiss_e3sm_amcb_2 = xr.open_dataset(data_dir+'e3sm/amcb/mcb_emiss_e3sm_amcb_2.nc').ncl_a1SF
mcb_emiss_e3sm_amcb_3 = xr.open_dataset(data_dir+'e3sm/amcb/mcb_emiss_e3sm_amcb_3.nc').ncl_a1SF
mcb_emiss_e3sm_amcb = (mcb_emiss_e3sm_amcb_1+mcb_emiss_e3sm_amcb_2+mcb_emiss_e3sm_amcb_3)/3

# Don't need ctl emission as done differently in UKESM1. The AMCB value is the value injected
mcb_emiss_cesm_ctl = xr.open_dataarray(data_dir+'cesm/ctl/mcb_emiss_cesm_ctl_ave.nc')
mcb_emiss_e3sm_ctl_1 = xr.open_dataset(data_dir+'e3sm/ctl/mcb_emiss_e3sm_ctl_1.nc').ncl_a1SF
mcb_emiss_e3sm_ctl_2 = xr.open_dataset(data_dir+'e3sm/ctl/mcb_emiss_e3sm_ctl_2.nc').ncl_a1SF
mcb_emiss_e3sm_ctl_3 = xr.open_dataset(data_dir+'e3sm/ctl/mcb_emiss_e3sm_ctl_3.nc').ncl_a1SF
mcb_emiss_e3sm_ctl = (mcb_emiss_e3sm_ctl_1+mcb_emiss_e3sm_ctl_2+mcb_emiss_e3sm_ctl_3)/3


# In[4]:


ukesm_tas_hist_1 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_1_ref.nc').air_temperature.drop(['forecast_period','forecast_reference_time']).rename({'latitude':'lat','longitude':'lon'})
ukesm_tas_hist_2 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_2_ref.nc').air_temperature.drop(['forecast_period','forecast_reference_time']).rename({'latitude':'lat','longitude':'lon'})
ukesm_tas_hist_3 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_3_ref.nc').air_temperature.drop(['forecast_period','forecast_reference_time']).rename({'latitude':'lat','longitude':'lon'})


# In[5]:


tas_ukesm_amcb_1 = xr.open_dataarray(data_dir+'ukesm/amcb/tas_ukesm_amcb_1.nc')
tas_ukesm_amcb_2 = xr.open_dataarray(data_dir+'ukesm/amcb/tas_ukesm_amcb_2.nc')
tas_ukesm_amcb_3 = xr.open_dataarray(data_dir+'ukesm/amcb/tas_ukesm_amcb_3.nc')
tas_ukesm_amcb = (tas_ukesm_amcb_1+tas_ukesm_amcb_2+tas_ukesm_amcb_3)/3
tas_cesm_amcb_1 = xr.open_dataarray(data_dir+'cesm/amcb/tas_cesm_amcb_1.nc')
tas_cesm_amcb_2 = xr.open_dataarray(data_dir+'cesm/amcb/tas_cesm_amcb_2.nc')
tas_cesm_amcb_3 = xr.open_dataarray(data_dir+'cesm/amcb/tas_cesm_amcb_3.nc')
tas_cesm_amcb = (tas_cesm_amcb_1+tas_cesm_amcb_2+tas_cesm_amcb_3)/3
tas_e3sm_amcb_1 = xr.open_dataarray(data_dir+'e3sm/amcb/tas_e3sm_amcb_1.nc')
tas_e3sm_amcb_2 = xr.open_dataarray(data_dir+'e3sm/amcb/tas_e3sm_amcb_2.nc')
tas_e3sm_amcb_3 = xr.open_dataarray(data_dir+'e3sm/amcb/tas_e3sm_amcb_3.nc')
tas_e3sm_amcb = (tas_e3sm_amcb_1+tas_e3sm_amcb_2+tas_e3sm_amcb_3)/3

tas_ukesm_ctl_1 = xr.open_mfdataset('/data/users/managecmip/champ/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp245/r1i1p1f2/Amon/tas/gn/v20190507/*.nc').tas
tas_ukesm_ctl_2 = xr.open_mfdataset('/data/users/managecmip/champ/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp245/r2i1p1f2/Amon/tas/gn/v20190507/*.nc').tas
tas_ukesm_ctl_3 = xr.open_mfdataset('/data/users/managecmip/champ/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp245/r3i1p1f2/Amon/tas/gn/v20190507/*.nc').tas
#Add the year 2014 to the UKESM data
tas_ukesm_ctl_1 = xr.concat([ukesm_tas_hist_1,tas_ukesm_ctl_1],dim='time')
tas_ukesm_ctl_2 = xr.concat([ukesm_tas_hist_2,tas_ukesm_ctl_2],dim='time')
tas_ukesm_ctl_3 = xr.concat([ukesm_tas_hist_3,tas_ukesm_ctl_3],dim='time')
tas_ukesm_ctl = (tas_ukesm_ctl_1+tas_ukesm_ctl_2+tas_ukesm_ctl_3)/3
tas_cesm_ctl = cesm_ssp245_ens.TREFHT
tas_cesm_ctl_1 = xr.open_mfdataset(data_dir+'/cesm/*.001.*.TREFHT.*.nc').TREFHT
tas_cesm_ctl_2 = xr.open_mfdataset(data_dir+'/cesm/*.002.*.TREFHT.*.nc').TREFHT
tas_cesm_ctl_3 = xr.open_mfdataset(data_dir+'/cesm/*.003.*.TREFHT.*.nc').TREFHT
tas_cesm_ctl_ave = (tas_cesm_ctl_1+tas_cesm_ctl_2+tas_cesm_ctl_3)/3
tas_e3sm_ctl_1 = xr.open_dataarray(data_dir+'e3sm/ctl/tas_e3sm_ctl_1.nc')
tas_e3sm_ctl_2 = xr.open_dataarray(data_dir+'e3sm/ctl/tas_e3sm_ctl_2.nc')
tas_e3sm_ctl_3 = xr.open_dataarray(data_dir+'e3sm/ctl/tas_e3sm_ctl_3.nc')
tas_e3sm_ctl = (tas_e3sm_ctl_1+tas_e3sm_ctl_2+tas_e3sm_ctl_3)/3


# In[6]:


weights_ukesm_ssp = np.cos(np.deg2rad(tas_ukesm_ctl.lat))
weights_ukesm_ssp.name = "weights"
weights_ukesm_amcb = np.cos(np.deg2rad(tas_ukesm_amcb.latitude))
weights_ukesm_amcb.name = "weights"
weights_e3sm = np.cos(np.deg2rad(tas_e3sm_amcb.lat))
weights_e3sm.name = "weights"
weights_cesm = np.cos(np.deg2rad(tas_cesm_ctl.lat))
weights_cesm.name = "weights"


# In[7]:


print('Reference values')
print('UKESM1 (global) 1850-80 : 286.53 K\nCESM2 (global) 1850-80 : 287.25 K\nE3SM (global) 1850-80 : 286.83 K')
print('UKESM global-mean T (2014-2033) : ',tas_ukesm_ctl.sel(time=slice('2014','2033')).weighted(weights_ukesm_ssp).mean(('time','lat','lon')).values)
print('CESM global-mean T (2020-2039) : ',tas_cesm_ctl.sel(time=slice('2020','2039')).weighted(weights_cesm).mean(('time','lat','lon')).values)
print('E3SM global-mean T (2029-2048) : ',tas_e3sm_ctl.sel(time=slice('2029','2048')).weighted(weights_e3sm).mean(('time','lat','lon')).values)
print('UKESM Arctic T (2014-2033) : ',tas_ukesm_ctl.where(tas_ukesm_ctl.lat>70).sel(time=slice('2014','2033')).weighted(weights_ukesm_ssp).mean(('time','lat','lon')).values)
print('CESM Arctic T (2020-2039) : ',tas_cesm_ctl.where(tas_cesm_ctl.lat>70).sel(time=slice('2020','2039')).weighted(weights_cesm).mean(('time','lat','lon')).values)
print('E3SM Arctic T (2029-2048) : ',tas_e3sm_ctl.where(tas_e3sm_ctl.lat>70).sel(time=slice('2029','2048')).weighted(weights_e3sm).mean(('time','lat','lon')).values)


# In[88]:


ukesm_seaice_hist_1 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_1_ref.nc').sea_ice_area_fraction
ukesm_seaice_hist_2 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_2_ref.nc').sea_ice_area_fraction
ukesm_seaice_hist_3 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_3_ref.nc').sea_ice_area_fraction

DSA = xr.open_mfdataset(data_dir+'/ne30pg2.nc').rename({'grid_size':'ncol'})

seaice_ssp_ukesm_1 = xr.open_dataarray(data_dir+'/ukesm/seaice_ssp_1.nc')
seaice_ssp_ukesm_2 = xr.open_dataarray(data_dir+'/ukesm/seaice_ssp_2.nc')
seaice_ssp_ukesm_3 = xr.open_dataarray(data_dir+'/ukesm/seaice_ssp_3.nc')
#add the year 2014 to the UKESM data
seaice_ssp_ukesm_1 = xr.concat([ukesm_seaice_hist_1,seaice_ssp_ukesm_1],dim='time')
seaice_ssp_ukesm_2 = xr.concat([ukesm_seaice_hist_2,seaice_ssp_ukesm_2],dim='time')
seaice_ssp_ukesm_3 = xr.concat([ukesm_seaice_hist_3,seaice_ssp_ukesm_3],dim='time')
seaice_ssp_ukesm_1 = seaice_ssp_ukesm_1.where(seaice_ssp_ukesm_1 < 1e30)
seaice_ssp_ukesm_2 = seaice_ssp_ukesm_2.where(seaice_ssp_ukesm_2 < 1e30)
seaice_ssp_ukesm_3 = seaice_ssp_ukesm_3.where(seaice_ssp_ukesm_3 < 1e30)
seaice_ssp_ukesm_ave = (seaice_ssp_ukesm_1+seaice_ssp_ukesm_2+seaice_ssp_ukesm_3)/3

seaice_ssp_cesm_1 = xr.open_mfdataset(data_dir+'/cesm/*.001.*ICEFRAC*.nc').ICEFRAC
seaice_ssp_cesm_2 = xr.open_mfdataset(data_dir+'/cesm/*.002.*ICEFRAC*.nc').ICEFRAC
seaice_ssp_cesm_3 = xr.open_mfdataset(data_dir+'/cesm/*.003.*ICEFRAC*.nc').ICEFRAC
seaice_ssp_cesm = (seaice_ssp_cesm_1+seaice_ssp_cesm_2+seaice_ssp_cesm_3)/3

e3sm_ICEFRAC_ctl_1 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_01_ICEFRAC.nc')
e3sm_ICEFRAC_ctl_2 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_02_ICEFRAC.nc')
e3sm_ICEFRAC_ctl_3 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_03_ICEFRAC.nc')
seaice_ssp_e3sm = (e3sm_ICEFRAC_ctl_1+e3sm_ICEFRAC_ctl_2+e3sm_ICEFRAC_ctl_3)/3
land_e3sm = land_cesm.LANDFRAC.interp(
    lat=seaice_ssp_e3sm.lat,
    lon=seaice_ssp_e3sm.lon,
    method="nearest" # Use nearest-neighbor to preserve mask values (0 or 1)
)


# In[8]:


fs=16


# In[89]:


fs = 16
fig = plt.figure(figsize=(18,12),dpi=300)
plt.subplot(331)
mcb_emiss_ukesm_amcb.plot(c='r',lw=1)
mcb_emiss_ukesm_amcb.rolling(time=12, center=True).mean().plot(c='k',lw=3)
plt.ylabel('Tg/yr',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.title('(a) UKESM1 SSA emissions (60-80N)',fontweight='bold', fontsize=fs)
plt.xlim(cftime.Datetime360Day(2020, 1, 1),cftime.Datetime360Day(2075, 1, 1))
plt.xticks([cftime.Datetime360Day(i, 1, 1) for i in np.linspace(2020,2070,6)], np.linspace(2020,2070,6).astype(int),fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,140])
plt.grid()
plt.subplot(332)
(mcb_emiss_cesm_amcb.where(cesm_ssp245_ens.lat > 60).weighted(refdata.AREA[0]).sum(('lon','lat'))/1e9*(60*60*24*365)).plot(c='r',lw=1)
(mcb_emiss_cesm_amcb.where(cesm_ssp245_ens.lat > 60).weighted(refdata.AREA[0]).sum(('lon','lat'))/1e9*(60*60*24*365)).rolling(time=12, center=True).mean().plot(c='k',lw=3)
plt.ylabel('Tg/yr',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.title('(b) CESM2 SSA emissions (60-80N)',fontweight='bold', fontsize=fs)
plt.xlim(cftime.Datetime360Day(2020, 1, 1),cftime.Datetime360Day(2075, 1, 1))
plt.xticks([cftime.Datetime360Day(i, 1, 1) for i in np.linspace(2020,2070,6)], np.linspace(2020,2070,6).astype(int),fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,14])
plt.grid()
plt.subplot(333)
((mcb_emiss_e3sm_amcb-mcb_emiss_e3sm_ctl).where(DSA.grid_center_lat>60)\
 .weighted(DSA.grid_area).sum('ncol')*6.371e6**2/1e9*(365*24*60*60)).plot(c='r',lw=1)
((mcb_emiss_e3sm_amcb-mcb_emiss_e3sm_ctl).where(DSA.grid_center_lat>60)\
 .weighted(DSA.grid_area).sum('ncol')*6.371e6**2/1e9*(365*24*60*60)).rolling(time=12, center=True).mean().plot(c='k',lw=3)
plt.ylabel('Tg/yr',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.title('(c) E3SM SSA emissions (60-80N)',fontweight='bold', fontsize=fs)
plt.xlim(cftime.Datetime360Day(2020, 1, 1),cftime.Datetime360Day(2075, 1, 1))
plt.xticks([cftime.Datetime360Day(i, 1, 1) for i in np.linspace(2020,2070,6)], np.linspace(2020,2070,6).astype(int),fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,30])
plt.grid()
plt.subplot(334)
plt.axhline(y=0, color='k', linestyle='--',lw=2)
(tas_ukesm_ctl-260.4).sel(lat=slice(70,90)).weighted(weights_ukesm_ssp).mean(('lat','lon'))\
  .groupby('time.year').mean('time').plot(c='r',lw=3,label='SSP2-4.5')
(tas_ukesm_amcb-260.4).sel(latitude=slice(70,90)).weighted(weights_ukesm_amcb).mean(('latitude','longitude'))\
  .groupby('time.year').mean('time').plot(c='b',lw=3,label='Arctic MCB')
for sim in [tas_ukesm_ctl_1,tas_ukesm_ctl_2,tas_ukesm_ctl_3]:
    (sim-260.4).sel(lat=slice(70,90))\
      .weighted(weights_ukesm_ssp).mean(('lat','lon')).groupby('time.year')\
      .mean('time').plot(c='r',lw=1,alpha=0.5)
for sim in [tas_ukesm_amcb_1,tas_ukesm_amcb_2,tas_ukesm_amcb_3]:
    (sim-260.4).sel(latitude=slice(70,90))\
      .weighted(weights_ukesm_amcb).mean(('latitude','longitude')).groupby('time.year')\
      .mean('time').plot(c='b',lw=1,alpha=0.5)
plt.title('(d) UKESM1 Arctic T$_S$ (70N-90N)',fontweight='bold', fontsize=fs)
# plt.ylim([0.5,4])
plt.ylabel('K',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
plt.xlim([2020,2075])
plt.ylim([-2,8])
plt.legend()
plt.subplot(335)
plt.axhline(y=0, color='k', linestyle='--',lw=2)
(tas_cesm_ctl_ave-263.6).sel(lat=slice(70,90)).weighted(weights_cesm).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='r',lw=3,label='SSP2-4.5')
(tas_cesm_amcb-263.6).sel(lat=slice(70,90)).weighted(cesm_ssp245_ens.gw).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='b',lw=3,label='Arctic MCB')
for sim in [tas_cesm_ctl_1,tas_cesm_ctl_2,tas_cesm_ctl_3]:
    (sim-263.6).sel(lat=slice(70,90)).weighted(weights_cesm).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='r',lw=1,alpha=0.5)
for sim in [tas_cesm_amcb_1,tas_cesm_amcb_2,tas_cesm_amcb_3]:
    (sim-263.6).sel(lat=slice(70,90)).weighted(cesm_ssp245_ens.gw).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='b',lw=1,alpha=0.5)
plt.title('(e) CESM2 Arctic T$_S$ (70N-90N)',fontweight='bold', fontsize=fs)
# plt.ylim([0.5,4])
plt.ylabel('K',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
plt.xlim([2020,2075])
plt.ylim([-2,8])
plt.legend()
plt.subplot(336)
plt.axhline(y=0, color='k', linestyle='--',lw=2)
(tas_e3sm_ctl-266.25).sel(lat=slice(70,90)).weighted(weights_e3sm).mean(('lat','lon'))\
  .groupby('time.year').mean('time').plot(c='r',lw=3,label='SSP2-4.5')
(tas_e3sm_amcb-266.25).sel(lat=slice(70,90)).weighted(weights_e3sm).mean(('lat','lon'))\
  .groupby('time.year').mean('time').plot(c='b',lw=3,label='Arctic MCB')
for sim in [tas_e3sm_ctl_1,tas_e3sm_ctl_2,tas_e3sm_ctl_3]:
    (sim-266.25).sel(lat=slice(70,90)).weighted(weights_e3sm).mean(('lat','lon'))\
      .groupby('time.year').mean('time').plot(c='r',lw=1,alpha=0.5)
for sim in [tas_e3sm_amcb_1,tas_e3sm_amcb_2,tas_e3sm_amcb_3]:
    (sim-266.25).sel(lat=slice(70,90)).weighted(weights_e3sm).mean(('lat','lon'))\
      .groupby('time.year').mean('time').plot(c='b',lw=1,alpha=0.5)
plt.title('(f) E3SM Arctic T$_S$ (70N-90N)',fontweight='bold', fontsize=fs)
# plt.ylim([0.5,4])
plt.ylabel('K',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
plt.xlim([2020,2075])
plt.ylim([-2,8])
plt.legend()

ax1 = fig.add_subplot(3, 3, 7, projection=ccrs.NorthPolarStereo())
data_diff = (tas_ukesm_amcb.rename({'latitude':'lat','longitude':'lon'}).sel(time=slice('2055','2074')).mean('time')- \
             tas_ukesm_ctl.sel(time=slice('2055','2074')).mean('time'))
data_diff_cyclic = xr.concat([data_diff,data_diff.isel(lon=0).assign_coords(lon=361)],dim='lon')
p = data_diff_cyclic.plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
ax1.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax1.set_boundary(circle, transform=ax1.transAxes)
(seaice_ssp_ukesm_ave*100).where(seaice_ssp_ukesm_ave<0.5).sel(time=(seaice_ssp_ukesm_ave.time.dt.month==9)).sel(time=slice('2014','2033'))\
    .where(seaice_ssp_ukesm_ave.latitude>=60).where(seaice_ssp_ukesm_ave.latitude<=80).mean('time').plot\
    .contourf(levels=[0.5,1], colors='None',hatches=['////'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
(seaice_ssp_ukesm_ave*100).where(seaice_ssp_ukesm_ave<0.5).sel(time=(seaice_ssp_ukesm_ave.time.dt.month==3)).sel(time=slice('2014','2033'))\
    .where(seaice_ssp_ukesm_ave.latitude>=60).where(seaice_ssp_ukesm_ave.latitude<=80).mean('time').plot\
    .contourf(levels=[0.5,1], colors='None',hatches=['\\\\'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
ax1.set_title("(g) UKESM1 $\Delta$T$_S$ (2055-74)",fontweight='bold', fontsize=fs)
ax1.coastlines()
ax2 = fig.add_subplot(3, 3, 8, projection=ccrs.NorthPolarStereo())
data_diff = (tas_cesm_amcb-tas_cesm_ctl).sel(time=slice('2055','2074')).mean('time')
data_diff_cyclic = xr.concat([data_diff,data_diff.isel(lon=0).assign_coords(lon=361)],dim='lon')
p = data_diff_cyclic.plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
ax2.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax2.set_boundary(circle, transform=ax2.transAxes)
ax2.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
(seaice_ssp_cesm*100).where(seaice_ssp_cesm<0.5).where(land_cesm.LANDFRAC<1).sel(time=(seaice_ssp_cesm.time.dt.month==9)).sel(time=slice('2020','2039'))\
    .where(seaice_ssp_cesm.lat>=60).where(seaice_ssp_cesm.lat<=80).mean('time').plot\
    .contourf(levels=[0.5,1], colors='None',hatches=['////'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
(seaice_ssp_cesm*100).where(seaice_ssp_cesm<0.5).where(land_cesm.LANDFRAC<1).sel(time=(seaice_ssp_cesm.time.dt.month==3)).sel(time=slice('2020','2039'))\
    .where(seaice_ssp_cesm.lat>=60).where(seaice_ssp_cesm.lat<=80).mean('time').plot\
    .contourf(levels=[0.5,1], colors='None',hatches=['\\\\'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
ax2.set_title("(h) CESM2 $\Delta$T$_S$ (2055-74)",fontweight='bold', fontsize=fs)
ax2.coastlines()
ax3 = fig.add_subplot(3, 3, 9, projection=ccrs.NorthPolarStereo())
data_diff = ((tas_e3sm_amcb-tas_e3sm_ctl).sel(time=slice('2055','2074')).mean('time'))
data_diff_cyclic = xr.concat([data_diff,data_diff.isel(lon=0).assign_coords(lon=361)],dim='lon')
p = data_diff_cyclic.plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
ax3.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax3.set_boundary(circle, transform=ax3.transAxes)
ax3.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
(seaice_ssp_e3sm*100).where(seaice_ssp_e3sm>0).where(seaice_ssp_e3sm<0.5).where(land_e3sm<1).sel(time=(seaice_ssp_e3sm.time.dt.month==9)).sel(time=slice('2029','2048'))\
    .where(seaice_ssp_e3sm.lat>=60).where(seaice_ssp_e3sm.lat<=80).mean('time').plot\
    .contourf(levels=[0.5,1], colors='None',hatches=['////'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
(seaice_ssp_e3sm*100).where(seaice_ssp_e3sm>0).where(seaice_ssp_e3sm<0.5).where(land_e3sm<1).sel(time=(seaice_ssp_e3sm.time.dt.month==3)).sel(time=slice('2029','2048'))\
    .where(seaice_ssp_e3sm.lat>=60).where(seaice_ssp_e3sm.lat<=80).mean('time').plot\
    .contourf(levels=[0.5,1], colors='None',hatches=['\\\\'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
ax3.set_title("(i) E3SM $\Delta$T$_S$ (2055-74)",fontweight='bold', fontsize=fs)
ax3.coastlines()
plt.tight_layout()
plt.savefig('./figs/fig_amcb_1.pdf',bbox_inches='tight')
plt.show()


# In[10]:


fig = plt.figure(figsize=(18,5),dpi=300)
ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.Robinson())
p = (tas_ukesm_amcb.rename({'latitude':'lat','longitude':'lon'}).sel(time=slice('2055','2074')).mean('time')- \
     tas_ukesm_ctl.sel(time=slice('2055','2074')).mean('time')).plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
# ax1.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
# ax1.set_boundary(circle, transform=ax1.transAxes)
ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
ax1.set_title("(a) UKESM1 $\Delta$T$_S$ (2055-74)",fontweight='bold', fontsize=fs)
ax1.coastlines()

ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.Robinson())
p = (tas_cesm_amcb-tas_cesm_ctl).sel(time=slice('2055','2074')).mean('time').plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
# ax2.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
# ax2.set_boundary(circle, transform=ax2.transAxes)
ax2.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
ax2.set_title("(b) CESM2 $\Delta$T$_S$ (2055-74)",fontweight='bold', fontsize=fs)
ax2.coastlines()

ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.Robinson())
p = ((tas_e3sm_amcb-tas_e3sm_ctl).sel(time=slice('2055','2074')).mean('time')).plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
# ax3.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
# ax3.set_boundary(circle, transform=ax3.transAxes)
ax3.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
ax3.set_title("(c) E3SM $\Delta$T$_S$ (2055-74)",fontweight='bold', fontsize=fs)
ax3.coastlines()
plt.tight_layout()
plt.savefig('./figs/fig_amcb_A10.pdf',bbox_inches='tight')
plt.show()


# In[11]:


# fname = '/data/users/mhenry/Arctic_MCB/u-ct629/sw_dn_toa/*.pp' 
# aod_ = iris.load(fname)
# sw_dn_toa_ctl = xr.DataArray.from_iris(aod_[0])
# var_tris=['sw_dn_sfc','sw_dn_net_sfc']
# var_tris_names=['surface_downwelling_shortwave_flux_in_air','surface_net_downward_shortwave_flux']
# var_bis=['clt','sw_up_toa','sw_up_toa_cs','sw_dn_sfc','sw_dn_cs_sfc','sw_up_cs_sfc']
# var_bis_names=['cloud_area_fraction','toa_outgoing_shortwave_flux','toa_outgoing_shortwave_flux_assuming_clear_sky',
#                'surface_downwelling_shortwave_flux_in_air','surface_downwelling_shortwave_flux_in_air_assuming_clear_sky',
#                'surface_upwelling_shortwave_flux_in_air_assuming_clear_sky']
# var = ['clt','rsut','rsutcs','rsds','rsdscs','rsuscs']
# #first is ctl and second is perturbation experiment
# exps = ['u-ct629','u-cx412']

# prefix = '/data/users/mhenry/Arctic_MCB/proc_data/'
# DATA={}
# for exp in exps:
#     DATA[exp]={}
#     DATA[exp]['rsdt']=sw_dn_toa_ctl.rename({'latitude':'lat','longitude':'lon'}).groupby('time.month').mean('time').rename({'month':'time'})

#     for i in range(len(var_tris)):
#         ds = xr.open_dataset(prefix+exp+'_'+var_tris[i]+'.nc')
#         climo = ds.groupby('time.month').mean('time').rename({'month':'time'})
#         DATA[exp][var_tris[i]] = climo[var_tris_names[i]].rename({'latitude':'lat','longitude':'lon'})

#     for i in range(len(var)):
#         ds = xr.open_dataset(prefix+exp+'_'+var_bis[i]+'.nc')
#         climo = ds.groupby('time.month').mean('time').rename({'month':'time'})
#         DATA[exp][var[i]] = climo[var_bis_names[i]].rename({'latitude':'lat','longitude':'lon'})

#     DATA[exp]['rsus']=DATA[exp]['sw_dn_sfc']-DATA[exp]['sw_dn_net_sfc']

# output_ukesm = APRP(DATA['u-ct629'],DATA['u-cx412'])
# output_ukesm.to_netcdf('/data/users/mhenry/ArcticMCB_data/ukesm/aprp_output_ukesm_50Tg_bis.nc')


# In[8]:


output_ukesm=xr.open_dataset(data_dir+'/ukesm/aprp_output_ukesm_50Tg_bis.nc')
output_e3sm=xr.open_dataset(data_dir+'/e3sm/aprp_output_e3sm.nc')
output_cesm=xr.open_dataset(data_dir+'/cesm/aprp_output_cesm.nc')
weights_aprp = np.cos(np.deg2rad(output_ukesm.lat))
weights_aprp.name = "weights"
weights_aprp_cesm = np.cos(np.deg2rad(output_cesm.lat))
weights_aprp_cesm.name = "weights"
weights_aprp_e3sm = np.cos(np.deg2rad(output_e3sm.lat))
weights_aprp_e3sm.name = "weights"

olr_erf_ctl_ukesm=xr.open_dataset(data_dir+'/ukesm/olr_erf_ctl_ukesm.nc')
olr_erf_amcb_ukesm=xr.open_dataset(data_dir+'/ukesm/u-cx412_olr.nc')
olr_erf_ctl_cesm=xr.open_dataset(data_dir+'/cesm/olr_erf_ctl_cesm.nc')
olr_erf_amcb_cesm=xr.open_dataset(data_dir+'/cesm/olr_erf_amcb_cesm.nc')
olr_erf_ctl_e3sm=xr.open_dataset(data_dir+'/e3sm/olr_erf_ctl_e3sm.nc')
olr_erf_amcb_e3sm=xr.open_dataset(data_dir+'/e3sm/olr_erf_amcb_e3sm.nc')
ukesm_erf_ctl_surf_lw=xr.open_dataarray(data_dir+'/ukesm/u-ct629_lw_net_sfc.nc')
ukesm_erf_amcb_surf_lw=xr.open_dataarray(data_dir+'/ukesm/u-cx412_lw_net_sfc.nc')
ukesm_erf_ctl_surf_sw=xr.open_dataarray(data_dir+'/ukesm/u-ct629_sw_dn_surf.nc')
ukesm_erf_amcb_surf_sw=xr.open_dataarray(data_dir+'/ukesm/u-cx412_sw_dn_sfc.nc')


# In[9]:


# cdnc_erf_ctl_ukesm=xr.open_dataarray('/data/users/mhenry/MCB/proc_data/u-ct629_cdnc.nc')
# cdnc_erf_amcb_ukesm=xr.open_dataarray('/data/users/mhenry/MCB/proc_data/u-cx412_cdnc.nc')


# In[9]:


cesm_erf_ctl_surf_lw = xr.open_dataset(data_dir+'/cesm/F2010climo.cam.h0.FLNS.1-19.zarr', engine = 'zarr')
cesm_erf_ctl_toa_lw = xr.open_dataset(data_dir+'/cesm/F2010climo.cam.h0.FLNT.1-19.zarr', engine = 'zarr')
cesm_erf_ctl_surf_sw = xr.open_dataset(data_dir+'/cesm/F2010climo.cam.h0.FSNS.1-19.zarr', engine = 'zarr')
cesm_erf_ctl_toa_sw = xr.open_dataset(data_dir+'/cesm/F2010climo.cam.h0.FSNT.1-19.zarr', engine = 'zarr')
cesm_erf_amcb_surf_lw = xr.open_mfdataset(data_dir+'/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.FLNS.1-11.zarr', engine = 'zarr')
cesm_erf_amcb_toa_lw = xr.open_mfdataset(data_dir+'/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.FLNT.1-11.zarr', engine = 'zarr')
cesm_erf_amcb_surf_sw = xr.open_mfdataset(data_dir+'/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.FSNS.1-11.zarr', engine = 'zarr')
cesm_erf_amcb_toa_sw = xr.open_mfdataset(data_dir+'/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.FSNT.1-11.zarr', engine = 'zarr')

e3sm_erf_ctl_surf_lw = xr.open_dataarray(data_dir+'/e3sm/ctl/e3sm_erf_ctl_surf_lw.nc')
e3sm_erf_ctl_toa_lw = xr.open_dataarray(data_dir+'/e3sm/ctl/e3sm_erf_ctl_toa_lw.nc')
e3sm_erf_ctl_surf_sw = xr.open_dataarray(data_dir+'/e3sm/ctl/e3sm_erf_ctl_surf_sw.nc')
e3sm_erf_ctl_toa_sw = xr.open_dataarray(data_dir+'/e3sm/ctl/e3sm_erf_ctl_toa_sw.nc')
e3sm_erf_amcb_surf_lw = xr.open_dataarray(data_dir+'/e3sm/amcb/e3sm_erf_amcb_surf_lw.nc')
e3sm_erf_amcb_toa_lw = xr.open_dataarray(data_dir+'/e3sm/amcb/e3sm_erf_amcb_toa_lw.nc')
e3sm_erf_amcb_surf_sw = xr.open_dataarray(data_dir+'/e3sm/amcb/e3sm_erf_amcb_surf_sw.nc')
e3sm_erf_amcb_toa_sw = xr.open_dataarray(data_dir+'/e3sm/amcb/e3sm_erf_amcb_toa_sw.nc')

ssemiss_ukesm = xr.open_dataarray(data_dir+'/ukesm/ss_emiss_erf_ukesm_50Tg.nc')
ssemiss_e3sm = xr.open_dataset(data_dir+'/e3sm/20240418.v2.LR.F2010.MCB-SSLT-EM.ARCTIC_16.33Tga.eam.h0.ncl_a1SF.zarr', engine = 'zarr')
ssemiss_cesm = xr.open_dataset(data_dir+'/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.ncl_a1SF.1-11.zarr', engine = 'zarr')
ssemiss_cesm_erf_clim = xr.open_dataset(data_dir+'/cesm/F2010climo.cam.h0.ncl_a1SF.1-19.zarr', engine = 'zarr')
ssemiss_e3sm_erf_clim = xr.open_dataset(data_dir+'/e3sm/20220930.v2.LR.F2010.E1_CNTL.eam.h0.ncl_a1SF.zarr', engine = 'zarr')


# In[15]:


print('UKESM emissions : ',ssemiss_ukesm.mean('month').values)
print('CESM emissions : ',((ssemiss_cesm.ncl_a1SF-ssemiss_cesm_erf_clim.ncl_a1SF).where(cesm_ssp245_ens.lat > 60).weighted(refdata.AREA[0])\
 .sum(('lon','lat')).mean('time')/1e9*(60*60*24*365)).values)
print('E3SM emissions : ',((ssemiss_e3sm.ncl_a1SF-ssemiss_e3sm_erf_clim.ncl_a1SF).where(DSA.grid_center_lat>=60).where(DSA.grid_center_lat<=90)\
 .weighted(DSA.grid_area).sum('ncol')*6.371e6**2/1e9*(365*24*60*60)).mean('time').values)


# In[16]:


print('UKESM ERF : ',((output_ukesm.noncld+output_ukesm.cld).sel(lat=slice(60,90)).weighted(weights_aprp).mean(('lat','lon','time'))+\
                      (-olr_erf_amcb_ukesm.toa_outgoing_longwave_flux+olr_erf_ctl_ukesm.toa_outgoing_longwave_flux)\
                      .sel(latitude=slice(60,90)).weighted(weights_ukesm_amcb).mean(('time','longitude','latitude'))).values,' W/m2 (annual mean)')
print('CESM ERF : ',((output_cesm.noncld+output_cesm.cld).sel(lat=slice(60,90)).weighted(weights_aprp_cesm).mean(('lat','lon','time'))+\
                    (-olr_erf_amcb_cesm.FLNTC-olr_erf_amcb_cesm.LWCF+olr_erf_ctl_cesm.FLNTC+olr_erf_ctl_cesm.LWCF)\
                    .sel(lat=slice(60,90)).weighted(weights_cesm).mean(('time','lon','lat'))).values,' W/m2 (annual mean)')
print('E3SM ERF : ',((output_e3sm.noncld+output_e3sm.cld).sel(lat=slice(60,90)).weighted(weights_aprp_e3sm).mean(('lat','lon','time'))+\
                    (-olr_erf_amcb_e3sm.FLNT+olr_erf_ctl_e3sm.FLNT)\
                    .sel(lat=slice(60,90)).weighted(weights_e3sm).mean(('time','lon','lat'))).values,' W/m2 (annual mean)')


# In[17]:


print('ARCTIC ERF/mass (mass as given by Haruki)')
print('UKESM : -2.26 / 26.74 = ', -2.26/26.74, ' W/m2/Tg/yr (annual-mean)')
print('CESM : -4.5 / 2.48 = ', -4.5/2.48, ' W/m2/Tg/yr (annual-mean)')
print('E3SM : -3.1 / 13.1 = ', -3.1/13.1, ' W/m2/Tg/yr (annual-mean)')


# In[18]:


print('Arctic cooling per mass for UKESM is : -3.8K / 26.74 = ',-3.8/26.74,' K/Tg/yr')


# In[32]:


dns_aer_sst = 2.2e3 #seasalt density
print('Aerosol radii:')
print('UKESM1 : 86nm')
print('CESM2 and E3SMv2 : split 50/50 between 41nm and 52nm')
print('')

print('Mass emission rates (mean over emission area)')
ssemiss_ukesm_erf = xr.open_dataarray(data_dir+'/ukesm/ss_emiss_erf_ukesm_50Tg_unit.nc')
ssemiss_mean_area = ssemiss_ukesm_erf.sel(latitude=slice(60,80)).weighted(weights_ukesm_amcb).mean(('time','latitude','longitude')).values
# above in kg/m2/s
print('UKESM : ',np.mean(ssemiss_mean_area/(4./3.*np.pi*((86e-9)**3)*dns_aer_sst))/1e6,' 1e6/s/m2') #1e6/s/m2

cesm_arcticmcb_emis = ssemiss_cesm.ncl_a1SF.where(ssemiss_cesm.lat >= 60).where(ssemiss_cesm.lat <= 80).weighted(weights_aprp_cesm).mean(('lat','lon','time')) - \
                      ssemiss_cesm_erf_clim.ncl_a1SF.where(ssemiss_cesm.lat >= 60).where(ssemiss_cesm.lat <= 80).weighted(weights_aprp_cesm).mean(('lat','lon','time'))
print('CESM : ',(np.mean(cesm_arcticmcb_emis/(4./6.*np.pi*((41e-9)**3 + (52e-9)**3)*dns_aer_sst))/1e6).values,' 1e6/s/m2') #1e6/s/m2

e3sm_arcticmcb_emis = ssemiss_e3sm.ncl_a1SF.where(DSA.grid_center_lat >= 60).where(DSA.grid_center_lat <= 80).weighted(DSA.grid_area).mean(('ncol','time')) - \
                      ssemiss_e3sm_erf_clim.ncl_a1SF.where(DSA.grid_center_lat >= 60).where(DSA.grid_center_lat <= 80).weighted(DSA.grid_area).mean(('ncol','time'))
print('E3SM : ',(np.mean(e3sm_arcticmcb_emis/(4./6.*np.pi*((41e-9)**3 + (52e-9)**3)*dns_aer_sst))/1e6).values,' 1e6/s/m2') #1e6/s/m2


# In[20]:


fs=12
fig, axes = plt.subplots(3, 3, figsize=(14, 9), dpi=400, sharex=True)

# Define model data and parameters
models = {
    "(a) SW forcing (UKESM1, 26.7Tg/yr)": {
        "clear_sky": output_ukesm.noncld.sel(lat=slice(60, 90)).weighted(weights_aprp).mean(('lat', 'lon')),
        "cloud_forcing": output_ukesm.cld.sel(lat=slice(60, 90)).weighted(weights_aprp).mean(('lat', 'lon'))
    },
    "(b) SW forcing (CESM2, 2.5Tg/yr)": {
        "clear_sky": output_cesm.noncld.sel(lat=slice(60, 90)).weighted(weights_aprp_cesm).mean(('lat', 'lon')).roll(time=-1),
        "cloud_forcing": output_cesm.cld.sel(lat=slice(60, 90)).weighted(weights_aprp_cesm).mean(('lat', 'lon')).roll(time=-1)
    },
    "(c) SW forcing (E3SMv2, 13.1Tg/yr)": {
        "clear_sky": output_e3sm.noncld.sel(lat=slice(60, 90)).weighted(weights_aprp_e3sm).mean(('lat', 'lon')).roll(time=-1),
        "cloud_forcing": output_e3sm.cld.sel(lat=slice(60, 90)).weighted(weights_aprp_e3sm).mean(('lat', 'lon')).roll(time=-1)
    }
}

# Plot radiative forcing (top row)
for ax, (title, data) in zip(axes[0], models.items()):
    data["clear_sky"].plot(ax=ax, lw=3, c='y', ls='--', label='SW clear-sky forcing')
    data["cloud_forcing"].plot(ax=ax, lw=3, c='y', label='SW cloud forcing')

    ax.set_ylabel('W/m$^2$', fontweight='bold', fontsize=fs)
    ax.set_xlabel('Month', fontweight='bold', fontsize=fs)
    ax.set_xlim([1, 12])
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=fs)
    ax.set_ylim([-12.5, 0.5])
    ax.grid()

axes[0, -1].legend()

# Define sea-salt emissions data (bottom row)
emissions = {
    "(d) Sea-salt aerosol emissions (UKESM1)": ssemiss_ukesm,
    "(e) Sea-salt aerosol emissions (CESM2)": (
        (ssemiss_cesm.ncl_a1SF - ssemiss_cesm_erf_clim.ncl_a1SF)
        .where(cesm_ssp245_ens.lat > 60).weighted(refdata.AREA[0])
        .sum(('lon', 'lat')) / 1e9 * (60 * 60 * 24 * 365)
    ).groupby('time.month').mean('time').roll(month=-1),
    "(f) Sea-salt aerosol emissions (E3SMv2)": (
        (ssemiss_e3sm.ncl_a1SF - ssemiss_e3sm_erf_clim.ncl_a1SF)
        .where((DSA.grid_center_lat >= 60) & (DSA.grid_center_lat <= 80))
        .weighted(DSA.grid_area).sum('ncol') * 6.371e6**2 / 1e9 * (365 * 24 * 60 * 60)
    ).groupby('time.month').mean('time').roll(month=-1)
}

# Plot sea-salt emissions (bottom row)
y_limits = [(0, 40), (0, 5), (0, 25)]
for ax, (title, data), ylim in zip(axes[1], emissions.items(), y_limits):
    data.plot(ax=ax, lw=3, c='k')

    ax.set_ylabel('Tg/yr', fontweight='bold', fontsize=fs)
    ax.set_xlabel('Month', fontweight='bold', fontsize=fs)
    ax.set_xlim([1, 12])
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=fs)
    ax.set_ylim(ylim)
    ax.grid()

emissions = {
    "(g) Forcing / SSA emissions (UKESM1)": (output_ukesm.noncld+output_ukesm.cld).sel(lat=slice(60, 90))\
       .weighted(weights_aprp).mean(('lat', 'lon'))/ssemiss_ukesm.rename({'month':'time'}),
    "(h) Forcing / SSA emissions (CESM2)": (output_cesm.noncld+output_cesm.cld).sel(lat=slice(60, 90)).weighted(weights_aprp_cesm).mean(('lat', 'lon')).roll(time=-1).rename({'time':'month'})/((
        (ssemiss_cesm.ncl_a1SF - ssemiss_cesm_erf_clim.ncl_a1SF)
        .where(cesm_ssp245_ens.lat > 60).weighted(refdata.AREA[0])
        .sum(('lon', 'lat')) / 1e9 * (60 * 60 * 24 * 365)
    ).groupby('time.month').mean('time').roll(month=-1)),
    "(i) Forcing / SSA emissions (E3SMv2)": (output_e3sm.noncld+output_e3sm.cld).sel(lat=slice(60, 90)).weighted(weights_aprp_e3sm).mean(('lat', 'lon')).roll(time=-1).rename({'time':'month'})/((
        (ssemiss_e3sm.ncl_a1SF - ssemiss_e3sm_erf_clim.ncl_a1SF)
        .where((DSA.grid_center_lat >= 60) & (DSA.grid_center_lat <= 80))
        .weighted(DSA.grid_area).sum('ncol') * 6.371e6**2 / 1e9 * (365 * 24 * 60 * 60)
    ).groupby('time.month').mean('time').roll(month=-1))
}

# Plot sea-salt emissions (bottom row)
y_limits = [(-0.35,0), (-4.5,0), (-0.8,0)]
for ax, (title, data), ylim in zip(axes[2], emissions.items(), y_limits):
    data.plot(ax=ax, lw=3, c='k')

    ax.set_ylabel('W/m$^2$/(Tg/yr)', fontweight='bold', fontsize=fs)
    ax.set_xlabel('Month', fontweight='bold', fontsize=fs)
    ax.set_xlim([1, 12])
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=fs)
    ax.set_ylim(ylim)
    ax.grid()

plt.tight_layout()
plt.savefig('./figs/fig_amcb_2.pdf', bbox_inches='tight')
plt.show()


# In[21]:


import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=400, sharey=True)

models = {
    "(a) LW fluxes (fixed SST, UKESM1, 26.7Tg/yr)": {
        "LW TOA": (olr_erf_ctl_ukesm.toa_outgoing_longwave_flux -olr_erf_amcb_ukesm.toa_outgoing_longwave_flux)
            .sel(latitude=slice(60, 90)).weighted(weights_ukesm_amcb).mean('latitude')
            .groupby('time.month').mean(('time', 'longitude')),
        "LW surf": (-ukesm_erf_ctl_surf_lw + ukesm_erf_amcb_surf_lw)
            .sel(latitude=slice(60, 90)).weighted(weights_ukesm_amcb).mean('latitude')
            .groupby('time.month').mean(('time', 'longitude')),
    },

    "(b) LW fluxes (fixed SST, CESM2, 2.5Tg/yr)": {
        "LW TOA": (olr_erf_ctl_cesm.FLNTC + olr_erf_ctl_cesm.LWCF -olr_erf_amcb_cesm.FLNTC - olr_erf_amcb_cesm.LWCF)
            .sel(lat=slice(60, 90)).weighted(weights_cesm).mean('lat')
            .groupby('time.month').mean(('time', 'lon')).roll(month=-1),
        "LW surf": (cesm_erf_ctl_surf_lw.FLNS - cesm_erf_amcb_surf_lw.FLNS)
            .sel(lat=slice(60, 90)).weighted(weights_cesm).mean('lat')
            .groupby('time.month').mean(('time', 'lon')).roll(month=-1),
    },

    "(c) LW fluxes (fixed SST, E3SMv2, 13.1Tg/yr)": {
        "LW TOA": (olr_erf_ctl_e3sm.FLNT - olr_erf_amcb_e3sm.FLNT)
            .sel(lat=slice(60, 90)).weighted(weights_e3sm).mean('lat')
            .groupby('time.month').mean(('time', 'lon')).roll(month=-1),
        "LW surf": (e3sm_erf_ctl_surf_lw.isel(time=slice(0, 120)) - e3sm_erf_amcb_surf_lw.isel(time=slice(0, 120)))
            .sel(lat=slice(60, 90)).weighted(weights_e3sm).mean('lat')
            .groupby('time.month').mean(('time', 'lon')).roll(month=-1),
    },
}

months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

for ax, (title, data) in zip(axes, models.items()):
    data["LW TOA"].plot(ax=ax, lw=3, c='r', label='LW TOA')
    data["LW surf"].plot(ax=ax, lw=3, c='r', ls='--', label='LW surf')

    ax.set_ylabel('W/m$^2$', fontweight='bold')
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_xlim([1, 12])
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(months, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_ylim([-3,3])
    ax.grid()

axes[-1].legend()
plt.tight_layout()
plt.savefig('./figs/fig_amcb_A11.pdf', bbox_inches='tight')
plt.show()


# In[22]:


import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=400, sharey=True)

models = {
    "(a) LW fluxes (fixed SST, UKESM1, 26.7Tg/yr)": {
        "LW TOA": (olr_erf_ctl_ukesm.toa_outgoing_longwave_flux)
            .sel(latitude=slice(60, 90)).weighted(weights_ukesm_amcb).mean('latitude')
            .groupby('time.month').mean(('time', 'longitude')),
        "LW surf": (-ukesm_erf_ctl_surf_lw)
            .sel(latitude=slice(60, 90)).weighted(weights_ukesm_amcb).mean('latitude')
            .groupby('time.month').mean(('time', 'longitude')),
    },

    "(b) LW fluxes (fixed SST, CESM2, 2.5Tg/yr)": {
        "LW TOA": (olr_erf_ctl_cesm.FLNTC + olr_erf_ctl_cesm.LWCF)
            .sel(lat=slice(60, 90)).weighted(weights_cesm).mean('lat')
            .groupby('time.month').mean(('time', 'lon')).roll(month=-1),
        "LW surf": (cesm_erf_ctl_surf_lw.FLNS)
            .sel(lat=slice(60, 90)).weighted(weights_cesm).mean('lat')
            .groupby('time.month').mean(('time', 'lon')).roll(month=-1),
    },

    "(c) LW fluxes (fixed SST, E3SMv2, 13.1Tg/yr)": {
        "LW TOA": (olr_erf_ctl_e3sm.FLNT)
            .sel(lat=slice(60, 90)).weighted(weights_e3sm).mean('lat')
            .groupby('time.month').mean(('time', 'lon')).roll(month=-1),
        "LW surf": (e3sm_erf_ctl_surf_lw.isel(time=slice(0, 120)))
            .sel(lat=slice(60, 90)).weighted(weights_e3sm).mean('lat')
            .groupby('time.month').mean(('time', 'lon')).roll(month=-1),
    },
}

months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

for ax, (title, data) in zip(axes, models.items()):
    data["LW TOA"].plot(ax=ax, lw=3, c='r', label='LW TOA')
    data["LW surf"].plot(ax=ax, lw=3, c='r', ls='--', label='LW surf')

    ax.set_ylabel(' ')
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_xlim([1, 12])
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(months, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=9)
    # ax.set_ylim([-3,3])
    ax.grid()

axes[-3].set_ylabel('W/m$^2$', fontweight='bold')
axes[-1].legend()
plt.tight_layout()
# plt.savefig('./figs/fig_amcb_A11.pdf', bbox_inches='tight')
plt.show()


# Plot above confirms the signs are right, positive control for all except LW surf UKESM is positive upwards.

# In[23]:


print('UKESM CS forcing : ',output_ukesm.noncld.sel(lat=slice(60,90)).weighted(weights_aprp).mean(('lat','lon','time')).values)
print('UKESM cloud forcing : ',output_ukesm.cld.sel(lat=slice(60,90)).weighted(weights_aprp).mean(('lat','lon','time')).values)
print('CESM CS forcing : ',output_cesm.noncld.sel(lat=slice(60,90)).weighted(weights_aprp_cesm).mean(('lat','lon','time')).values)
print('CESM cloud forcing : ',output_cesm.cld.sel(lat=slice(60,90)).weighted(weights_aprp_cesm).mean(('lat','lon','time')).values)
print('E3SM CS forcing : ',output_e3sm.noncld.sel(lat=slice(60,90)).weighted(weights_aprp_e3sm).mean(('lat','lon','time')).values)
print('E3SM cloud forcing : ',output_e3sm.cld.sel(lat=slice(60,90)).weighted(weights_aprp_e3sm).mean(('lat','lon','time')).values)


# In[17]:


fs = 16
figsize = (18, 20)
dpi = 400
projection = ccrs.NorthPolarStereo()
extent = [-180, 180, 50, 90]
cmap = 'bwr'
levels = np.linspace(-15, 15, 31)
cbar_kwargs = {"shrink": 0.8, "label": "W/m$^2$"}
gridline_props = {"crs": ccrs.PlateCarree(), "linewidth": 1, "color": 'black', "alpha": 0.3}
titles = [
    "(a) UKESM1 SW TOA Cloud", "(b) CESM2 SW TOA Cloud", "(c) E3SMv2 SW TOA Cloud",
    "(d) UKESM1 SW TOA clear-sky", "(e) CESM2 SW TOA clear-sky", "(f) E3SMv2 SW TOA clear-sky",
    "(g) UKESM1 LW TOA", "(h) CESM2 LW TOA", "(i) E3SMv2 LW TOA",
    "(j) UKESM1 LW surf", "(k) CESM2 LW surf", "(l) E3SMv2 LW surf",
    "(m) UKESM1 SW surf", "(n) CESM2 SW surf", "(o) E3SMv2 SW surf"
]
data_sources = [
    output_ukesm.cld.weighted(weights_aprp).mean('time'),
    output_cesm.cld.weighted(weights_aprp_cesm).mean('time'),
    output_e3sm.cld.weighted(weights_aprp_e3sm).mean('time'),
    output_ukesm.noncld.weighted(weights_aprp).mean('time'),
    output_cesm.noncld.weighted(weights_aprp_cesm).mean('time'),
    output_e3sm.noncld.weighted(weights_aprp_e3sm).mean('time'),
    (-olr_erf_amcb_ukesm.toa_outgoing_longwave_flux + olr_erf_ctl_ukesm.toa_outgoing_longwave_flux).mean('time'),
    (-olr_erf_amcb_cesm.FLNTC - olr_erf_amcb_cesm.LWCF + olr_erf_ctl_cesm.FLNTC + olr_erf_ctl_cesm.LWCF).mean('time'),
    (-olr_erf_amcb_e3sm.FLNT + olr_erf_ctl_e3sm.FLNT).mean('time'),
    (-ukesm_erf_ctl_surf_lw + ukesm_erf_amcb_surf_lw).mean('time'),
    (cesm_erf_ctl_surf_lw.FLNS - cesm_erf_amcb_surf_lw.FLNS).mean('time'),
    (e3sm_erf_ctl_surf_lw.isel(time=slice(0, 120)) - e3sm_erf_amcb_surf_lw.isel(time=slice(0, 120))).mean('time'),
    (-ukesm_erf_ctl_surf_sw + ukesm_erf_amcb_surf_sw).mean('time'),
    (-cesm_erf_ctl_surf_sw.FSNS + cesm_erf_amcb_surf_sw.FSNS).mean('time'),
    (-e3sm_erf_ctl_surf_sw.isel(time=slice(0, 120)) + e3sm_erf_amcb_surf_sw.isel(time=slice(0, 120))).mean('time')
]

# Create figure
fig = plt.figure(figsize=figsize, dpi=dpi)

# Loop to create subplots
for i, (data, title) in enumerate(zip(data_sources, titles), start=1):
    ax = fig.add_subplot(5, 3, i, projection=projection)
    if 'lon' in data.coords:
        data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
    else:
        data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
    p = data_diff_cyclic.plot.contourf(
        levels=levels, cmap=cmap,
        subplot_kws={"projection": projection, "facecolor": "gray"},
        transform=ccrs.PlateCarree(), extend="both", cbar_kwargs=cbar_kwargs
    )
    cbar = p.colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("W/m$^2$", size=16)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.gridlines(**gridline_props)
    ax.set_title(title, fontsize=fs, fontweight='bold')
    ax.coastlines()

plt.tight_layout()
plt.savefig('./figs/fig_amcb_A4.pdf', bbox_inches='tight')
plt.show()


# The “net” flux is the difference between the upwelling and downwelling beams. We have to be careful about sign conventions! In the CESM data, “net” means net upward for longwave radiation, but net downward for shortwave radiation.
# 
# So less net upward LW flux when upwelling is fixed means more downwelling flux.
# 
# CAREFUL : In UKESM, ukesm_erf_ctl_surf_lw gives surface_net_downward_longwave_flux and it's negative, so opposite to CESM/E3SM.

# In[25]:


# cesm_erf_ctl_swp = xr.open_dataset('/data/users/mhenry/ArcticMCB_data/cesm/F2010climo.cam.h0.AQSNOW.1-19.zarr', engine = 'zarr').AQSNOW
# cesm_erf_amcb_swp = xr.open_dataset('/data/users/mhenry/ArcticMCB_data/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.AQSNOW.1-11.zarr', engine = 'zarr').AQSNOW
# DS1 = xr.open_dataset('/data/users/mhenry/ArcticMCB_data/cesm/F2010climo.cam.h0.AQSNOW.1-19.zarr', engine = 'zarr')
# gravity = 9.81
# PI = (DS1['PS']*DS1.hybi + DS1.hyai*DS1.P0) # pressure interfaces
# PI2 = PI.transpose('time','ilev',...)
# TMP = PI2.diff('ilev')/gravity
# DPOG = cesm_erf_ctl_swp.copy().rename('DPOG')
# DPOG.values = TMP.values
# DPOG.attrs['units'] = 'kg/m2'
# DPOG.attrs['long_name'] = 'Delta pressure over gravity'
# cesm_erf_ctl_swp_adj = (cesm_erf_ctl_swp*DPOG).sum(dim="lev")*1000.
# cesm_erf_amcb_swp_adj = (cesm_erf_amcb_swp*DPOG).sum(dim="lev")*1000.
# cesm_erf_ctl_swp_adj.to_netcdf('/data/users/mhenry/ArcticMCB_data/cesm/cesm_erf_ctl_swp.nc')
# cesm_erf_amcb_swp_adj.to_netcdf('/data/users/mhenry/ArcticMCB_data/cesm/cesm_erf_amcb_swp.nc')


# In[26]:


# e3sm_erf_ctl_swp = xr.open_dataset('/data/users/mhenry/ArcticMCB_data/e3sm/20220930.v2.LR.F2010.E1_CNTL.eam.h0.AQSNOW.zarr', engine = 'zarr').AQSNOW
# e3sm_erf_amcb_swp = xr.open_dataset('/data/users/mhenry/ArcticMCB_data/e3sm/20240418.v2.LR.F2010.MCB-SSLT-EM.ARCTIC_16.33Tga.eam.h0.AQSNOW.zarr', engine = 'zarr').AQSNOW
# e3sm_erf_ctl_swp = remap_3D(e3sm_erf_ctl_swp.transpose('ncol','time','lev'))
# e3sm_erf_amcb_swp = remap_3D(e3sm_erf_amcb_swp.transpose('ncol','time','lev'))
# DS1 = xr.open_dataset('/data/users/mhenry/ArcticMCB_data/e3sm/20220930.v2.LR.F2010.E1_CNTL.eam.h0.AQSNOW.zarr', engine = 'zarr')
# gravity = 9.81
# PI = (DS1['PS']*DS1.hybi + DS1.hyai*DS1.P0) # pressure interfaces
# PI2 = PI.transpose('time','ilev',...)
# TMP = PI2.diff('ilev')/gravity
# TMP = remap_3D(TMP.rename({'ilev':'lev'}).transpose('ncol','time','lev'))
# DPOG1 = e3sm_erf_ctl_swp.copy().rename('DPOG1')
# DPOG1.values = TMP.values
# DPOG1.attrs['units'] = 'kg/m2'
# DPOG1.attrs['long_name'] = 'Delta pressure over gravity'
# e3sm_erf_ctl_swp_adj = (e3sm_erf_ctl_swp*DPOG1).sum(dim="lev")*1000.
# e3sm_erf_amcb_swp_adj = (e3sm_erf_amcb_swp*DPOG1).sum(dim="lev")*1000.
# e3sm_erf_ctl_swp_adj.to_netcdf('/data/users/mhenry/ArcticMCB_data/e3sm/ctl/e3sm_erf_ctl_swp.nc')
# e3sm_erf_amcb_swp_adj.to_netcdf('/data/users/mhenry/ArcticMCB_data/e3sm/amcb/e3sm_erf_amcb_swp.nc')


# In[20]:


ukesm_erf_ctl_lwp=xr.open_dataarray(data_dir+'/ukesm/u-ct629_lwp.nc')
ukesm_erf_amcb_lwp=xr.open_dataarray(data_dir+'/ukesm/u-cx412_lwp.nc')
ukesm_erf_ctl_iwp=xr.open_dataarray(data_dir+'/ukesm/u-ct629_iwp.nc')
ukesm_erf_amcb_iwp=xr.open_dataarray(data_dir+'/ukesm/u-cx412_iwp.nc')
ukesm_erf_ctl_clt=xr.open_dataarray(data_dir+'/ukesm/u-ct629_clt.nc')
ukesm_erf_amcb_clt=xr.open_dataarray(data_dir+'/ukesm/u-cx412_clt.nc')

cesm_erf_ctl_clt = xr.open_dataset(data_dir+'/cesm/F2010climo.cam.h0.CLDTOT.1-19.zarr', engine = 'zarr').CLDTOT
cesm_erf_ctl_iwp = xr.open_dataset(data_dir+'/cesm/F2010climo.cam.h0.TGCLDIWP.1-19.zarr', engine = 'zarr').TGCLDIWP
cesm_erf_ctl_lwp = xr.open_dataset(data_dir+'/cesm/F2010climo.cam.h0.TGCLDLWP.1-19.zarr', engine = 'zarr').TGCLDLWP
cesm_erf_ctl_swp = xr.open_dataarray(data_dir+'/cesm/cesm_erf_ctl_swp.nc')/1000
cesm_erf_amcb_clt = xr.open_dataset(data_dir+'/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.CLDTOT.1-11.zarr', engine = 'zarr').CLDTOT
cesm_erf_amcb_iwp = xr.open_dataset(data_dir+'/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.TGCLDIWP.1-11.zarr', engine = 'zarr').TGCLDIWP
cesm_erf_amcb_lwp = xr.open_dataset(data_dir+'/cesm/F2010climo.ss_Arctic.scaledown_for_2.5Tg.60to90.cam.h0.TGCLDLWP.1-11.zarr', engine = 'zarr').TGCLDLWP
cesm_erf_amcb_swp = xr.open_dataarray(data_dir+'/cesm/cesm_erf_amcb_swp.nc')/1000

e3sm_erf_ctl_clt = xr.open_dataarray(data_dir+'/e3sm/ctl/e3sm_erf_ctl_cldtot.nc')
e3sm_erf_ctl_iwp = xr.open_dataarray(data_dir+'/e3sm/ctl/e3sm_erf_ctl_iwp.nc')
e3sm_erf_ctl_lwp = xr.open_dataarray(data_dir+'/e3sm/ctl/e3sm_erf_ctl_lwp.nc')
e3sm_erf_ctl_swp = xr.open_dataarray(data_dir+'/e3sm/ctl/e3sm_erf_ctl_swp.nc')/1000
e3sm_erf_amcb_clt = xr.open_dataarray(data_dir+'/e3sm/amcb/e3sm_erf_amcb_cldtot.nc')
e3sm_erf_amcb_iwp = xr.open_dataarray(data_dir+'/e3sm/amcb/e3sm_erf_amcb_iwp.nc')
e3sm_erf_amcb_lwp = xr.open_dataarray(data_dir+'/e3sm/amcb/e3sm_erf_amcb_lwp.nc')
e3sm_erf_amcb_swp = xr.open_dataarray(data_dir+'/e3sm/amcb/e3sm_erf_amcb_swp.nc')/1000


# In[23]:


cbarshrink = 0.8
figsize = (18, 12)
dpi = 400
projection = ccrs.NorthPolarStereo()
extent = [-180, 180, 50, 90]
gridline_props = {"crs": ccrs.PlateCarree(), "linewidth": 1, "color": 'black', "alpha": 0.3}

fig = plt.figure(figsize=figsize, dpi=dpi)

ax = fig.add_subplot(3, 3, 1, projection=projection)
data = (ukesm_erf_amcb_clt-ukesm_erf_ctl_clt).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.05, 0.05, 11), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Cloud Area Fraction"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label='Cloud Area Fraction',size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(a) UKESM1 Cloud Area Fraction', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 2, projection=projection)
data = (cesm_erf_amcb_clt-cesm_erf_ctl_clt).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.05, 0.05, 11), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Cloud Area Fraction"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label='Cloud Area Fraction',size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(b) CESM2 Cloud Area Fraction', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 3, projection=projection)
data = (e3sm_erf_amcb_clt-e3sm_erf_ctl_clt).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.05, 0.05, 11), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Cloud Area Fraction"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label='Cloud Area Fraction',size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(c) E3SMv2 Cloud Area Fraction', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 4, projection=projection)
data = (ukesm_erf_amcb_lwp-ukesm_erf_ctl_lwp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.04, 0.04, 9), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Kg/m$^2$"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(d) UKESM1 Liquid Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 5, projection=projection)
data = (cesm_erf_amcb_lwp-cesm_erf_ctl_lwp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.04, 0.04, 9), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Kg/m$^2$"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(e) CESM2 Liquid Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 6, projection=projection)
data = (e3sm_erf_amcb_lwp-e3sm_erf_ctl_lwp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.04, 0.04, 9), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Kg/m$^2$"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(f) E3SMv2 Liquid Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 7, projection=projection)
data = (ukesm_erf_amcb_iwp-ukesm_erf_ctl_iwp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.02, 0.02, 9), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Kg/m$^2$"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(g) UKESM1 Ice Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 8, projection=projection)
data = (cesm_erf_amcb_iwp+cesm_erf_amcb_swp-cesm_erf_ctl_iwp-cesm_erf_ctl_swp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.02, 0.02, 9), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Kg/m$^2$"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(h) CESM2 Ice Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 9, projection=projection)
data = (e3sm_erf_amcb_iwp+e3sm_erf_amcb_swp-e3sm_erf_ctl_iwp-e3sm_erf_ctl_swp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(-0.02, 0.02, 9), cmap='bwr_r',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Kg/m$^2$"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(i) E3SMv2 Ice Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

plt.tight_layout()
plt.savefig('./figs/fig_amcb_A12.pdf', bbox_inches='tight')
plt.show()


# In[26]:


figsize = (18, 12)
dpi = 400
projection = ccrs.NorthPolarStereo()
extent = [-180, 180, 50, 90]
gridline_props = {"crs": ccrs.PlateCarree(), "linewidth": 1, "color": 'black', "alpha": 0.3}

fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_subplot(3, 3, 1, projection=projection)
data = (ukesm_erf_ctl_clt).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(0.6,1,9),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink, "label": "Cloud Area Fraction"}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Cloud Area Fraction', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(a) UKESM1 Cloud Area Fraction', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 2, projection=projection)
data = (cesm_erf_ctl_clt).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(0.6,1,9),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Cloud Area Fraction', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(b) CESM2 Cloud Area Fraction', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 3, projection=projection)
p = (e3sm_erf_ctl_clt).mean('time').plot.contourf(
    levels=np.linspace(0.6,1,9),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Cloud Area Fraction', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(c) E3SMv2 Cloud Area Fraction', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 4, projection=projection)
data = (ukesm_erf_ctl_lwp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(0,0.15,16),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink}
)
ukesm_erf_ctl_lwp.mean('time').plot.contour(levels=[0.03],transform=ccrs.PlateCarree(),cmap='r',lw=3)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(d) UKESM1 Liquid Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 5, projection=projection)
data = (cesm_erf_ctl_lwp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(0,0.15,16),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink}
)
cesm_erf_ctl_lwp.mean('time').plot.contour(levels=[0.03],transform=ccrs.PlateCarree(),cmap='r',lw=3)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(e) CESM2 Liquid Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 6, projection=projection)
p = (e3sm_erf_ctl_lwp).mean('time').plot.contourf(
    levels=np.linspace(0,0.15,16),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink}
)
e3sm_erf_ctl_lwp.mean('time').plot.contour(levels=[0.03],transform=ccrs.PlateCarree(),cmap='r',lw=3)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(f) E3SMv2 Liquid Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 7, projection=projection)
data = (ukesm_erf_ctl_iwp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(0,0.15,16),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(g) UKESM1 Ice Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 8, projection=projection)
data = (cesm_erf_ctl_iwp+cesm_erf_ctl_swp).mean('time')
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
    levels=np.linspace(0,0.15,16),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(h) CESM2 Ice Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

ax = fig.add_subplot(3, 3, 9, projection=projection)
p = (e3sm_erf_ctl_iwp+e3sm_erf_ctl_swp).mean('time').plot.contourf(
    levels=np.linspace(0,0.15,16),cmap='Blues',
    subplot_kws={"projection": projection, "facecolor": "gray"},
    transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": cbarshrink}
)
cbar = p.colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Kg/m$^2$', size=16)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)
ax.gridlines(**gridline_props)
ax.set_title('(i) E3SMv2 Ice Water Path', fontsize=fs, fontweight='bold')
ax.coastlines()

plt.tight_layout()
plt.savefig('./figs/fig_amcb_A13.pdf', bbox_inches='tight')
plt.show()


# In[39]:


# figsize = (18, 8)
# dpi = 400
# projection = ccrs.NorthPolarStereo()
# extent = [-180, 180, 50, 90]
# gridline_props = {"crs": ccrs.PlateCarree(), "linewidth": 1, "color": 'black', "alpha": 0.3}

# fig = plt.figure(figsize=figsize, dpi=dpi)

# ax = fig.add_subplot(2, 3, 1, projection=projection)
# data = (ukesm_erf_amcb_lwp-ukesm_erf_ctl_lwp).sel(time=(ukesm_erf_ctl_lwp.time.dt.season=='DJF')).mean('time')
# if 'lon' in data.coords:
#     data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
# else:
#     data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
# data_diff_cyclic.plot.contourf(
#     levels=np.linspace(-0.04, 0.04, 9), cmap='bwr_r',
#     subplot_kws={"projection": projection, "facecolor": "gray"},
#     transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": 0.5, "label": "Kg/m$^2$"}
# )
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.gridlines(**gridline_props)
# ax.set_title('(a) UKESM1 Liquid Water Path (DJF)', fontsize=fs, fontweight='bold')
# ax.coastlines()

# ax = fig.add_subplot(2, 3, 2, projection=projection)
# data = (cesm_erf_amcb_lwp.sel(time=(cesm_erf_amcb_lwp.time.dt.season=='DJF'))-cesm_erf_ctl_lwp.sel(time=(cesm_erf_ctl_lwp.time.dt.season=='DJF'))).mean('time')
# if 'lon' in data.coords:
#     data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
# else:
#     data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
# data_diff_cyclic.plot.contourf(
#     levels=np.linspace(-0.04, 0.04, 9), cmap='bwr_r',
#     subplot_kws={"projection": projection, "facecolor": "gray"},
#     transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": 0.5, "label": "Kg/m$^2$"}
# )
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.gridlines(**gridline_props)
# ax.set_title('(b) CESM2 Liquid Water Path (DJF)', fontsize=fs, fontweight='bold')
# ax.coastlines()

# ax = fig.add_subplot(2, 3, 3, projection=projection)
# (e3sm_erf_amcb_lwp.sel(time=(e3sm_erf_amcb_lwp.time.dt.season=='DJF'))-e3sm_erf_ctl_lwp.sel(time=(e3sm_erf_ctl_lwp.time.dt.season=='DJF'))).mean('time').plot.contourf(
#     levels=np.linspace(-0.04, 0.04, 9), cmap='bwr_r',
#     subplot_kws={"projection": projection, "facecolor": "gray"},
#     transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": 0.5, "label": "Kg/m$^2$"}
# )
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.gridlines(**gridline_props)
# ax.set_title('(c) E3SMv2 Liquid Water Path (DJF)', fontsize=fs, fontweight='bold')
# ax.coastlines()

# ax = fig.add_subplot(2, 3, 4, projection=projection)
# data = (-olr_erf_amcb_ukesm.toa_outgoing_longwave_flux.sel(time=(olr_erf_amcb_ukesm.time.dt.season=='DJF')) + olr_erf_ctl_ukesm.toa_outgoing_longwave_flux.sel(time=(olr_erf_ctl_ukesm.time.dt.season=='DJF'))).mean('time')
# if 'lon' in data.coords:
#     data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
# else:
#     data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
# data_diff_cyclic.plot.contourf(
#     levels=np.linspace(-15, 15, 31), cmap='bwr',
#     subplot_kws={"projection": projection, "facecolor": "gray"},
#     transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": 0.5, "label": "Kg/m$^2$"}
# )
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.gridlines(**gridline_props)
# ax.set_title('(d) UKESM1 LW TOA (DJF)', fontsize=fs, fontweight='bold')
# ax.coastlines()

# ax = fig.add_subplot(2, 3, 5, projection=projection)
# data = (-olr_erf_amcb_cesm.FLNTC.sel(time=(olr_erf_amcb_cesm.time.dt.season=='DJF'))\
#  - olr_erf_amcb_cesm.LWCF.sel(time=(olr_erf_amcb_cesm.time.dt.season=='DJF'))\
#  + olr_erf_ctl_cesm.FLNTC.sel(time=(olr_erf_ctl_cesm.time.dt.season=='DJF'))\
#  + olr_erf_ctl_cesm.LWCF.sel(time=(olr_erf_ctl_cesm.time.dt.season=='DJF'))).mean('time')
# if 'lon' in data.coords:
#     data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
# else:
#     data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
# data_diff_cyclic.plot.contourf(
#     levels=np.linspace(-15, 15, 31), cmap='bwr',
#     subplot_kws={"projection": projection, "facecolor": "gray"},
#     transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": 0.5, "label": "Kg/m$^2$"}
# )
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.gridlines(**gridline_props)
# ax.set_title('(e) CESM2 LW TOA (DJF)', fontsize=fs, fontweight='bold')
# ax.coastlines()

# ax = fig.add_subplot(2, 3, 6, projection=projection)
# (-olr_erf_amcb_e3sm.FLNT.sel(time=(olr_erf_amcb_e3sm.time.dt.season=='DJF')) + \
#  olr_erf_ctl_e3sm.FLNT.sel(time=(olr_erf_ctl_e3sm.time.dt.season=='DJF'))).mean('time').plot.contourf(
#     levels=np.linspace(-15, 15, 31), cmap='bwr',
#     subplot_kws={"projection": projection, "facecolor": "gray"},
#     transform=ccrs.PlateCarree(), extend="both", cbar_kwargs={"shrink": 0.5, "label": "Kg/m$^2$"}
# )
# ax.set_extent(extent, crs=ccrs.PlateCarree())
# ax.set_boundary(circle, transform=ax.transAxes)
# ax.gridlines(**gridline_props)
# ax.set_title('(f) E3SMv2 LW TOA (DJF)', fontsize=fs, fontweight='bold')
# ax.coastlines()

# plt.tight_layout()
# # plt.savefig('./figs/fig_amcb_A13.pdf', bbox_inches='tight')
# plt.show()


# In[31]:


# seaice_ssp_ukesm_ave = xr.open_dataarray(data_dir+'/ukesm/seaice_ssp_ave.nc')
# seaice_ssp_ukesm_ave = seaice_ssp_ukesm_ave.where(seaice_ssp_ukesm_ave < 1e30)
seaice_amcb_ukesm_1 = xr.open_dataarray(data_dir+'/ukesm/seaice_amcb_ukesm_1.nc')
seaice_amcb_ukesm_2 = xr.open_dataarray(data_dir+'/ukesm/seaice_amcb_ukesm_2.nc')
seaice_amcb_ukesm_3 = xr.open_dataarray(data_dir+'/ukesm/seaice_amcb_ukesm_3.nc')
seaice_amcb_ukesm_ave = (seaice_amcb_ukesm_1+seaice_amcb_ukesm_2+seaice_amcb_ukesm_3)/3

seaice_ssp_cesm_ave = cesm_ssp245_ens.ICEFRAC
seaice_amcb_cesm = {}
for exp in cesm_mcb_data:
    seaice_amcb_cesm[exp] = cesm_mcb_data[exp].ICEFRAC
seaice_amcb_cesm_1 = seaice_amcb_cesm[cesm_mcb_exp[0]]
seaice_amcb_cesm_2 = seaice_amcb_cesm[cesm_mcb_exp[1]]
seaice_amcb_cesm_3 = seaice_amcb_cesm[cesm_mcb_exp[2]]
seaice_amcb_cesm_ave = (seaice_amcb_cesm_1+seaice_amcb_cesm_2+seaice_amcb_cesm_3)/3

e3sm_ICEFRAC_ctl_1 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_01_ICEFRAC.nc')
e3sm_ICEFRAC_ctl_2 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_02_ICEFRAC.nc')
e3sm_ICEFRAC_ctl_3 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_03_ICEFRAC.nc')
seaice_ssp_e3sm_ave = (e3sm_ICEFRAC_ctl_1+e3sm_ICEFRAC_ctl_2+e3sm_ICEFRAC_ctl_3)/3
seaice_amcb_e3sm_1 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_01.ARCTIC_CONTROL_ICEFRAC.nc')
seaice_amcb_e3sm_2 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_02.ARCTIC_CONTROL_ICEFRAC.nc')
seaice_amcb_e3sm_3 = xr.open_dataarray(data_dir+'/e3sm/E2_CNTL_03.ARCTIC_CONTROL_ICEFRAC.nc')
seaice_amcb_e3sm_ave = (seaice_amcb_e3sm_1+seaice_amcb_e3sm_2+seaice_amcb_e3sm_3)/3


# In[32]:


ukesm_seaice_hist_1 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_1_ref.nc').sea_ice_area_fraction
ukesm_seaice_hist_2 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_2_ref.nc').sea_ice_area_fraction
ukesm_seaice_hist_3 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_3_ref.nc').sea_ice_area_fraction


# In[33]:


DSA = xr.open_mfdataset(data_dir+'/ne30pg2.nc').rename({'grid_size':'ncol'})

seaice_ssp_ukesm_1 = xr.open_dataarray(data_dir+'/ukesm/seaice_ssp_1.nc')
seaice_ssp_ukesm_2 = xr.open_dataarray(data_dir+'/ukesm/seaice_ssp_2.nc')
seaice_ssp_ukesm_3 = xr.open_dataarray(data_dir+'/ukesm/seaice_ssp_3.nc')
#add the year 2014 to the UKESM data
seaice_ssp_ukesm_1 = xr.concat([ukesm_seaice_hist_1,seaice_ssp_ukesm_1],dim='time')
seaice_ssp_ukesm_2 = xr.concat([ukesm_seaice_hist_2,seaice_ssp_ukesm_2],dim='time')
seaice_ssp_ukesm_3 = xr.concat([ukesm_seaice_hist_3,seaice_ssp_ukesm_3],dim='time')
seaice_ssp_ukesm_1 = seaice_ssp_ukesm_1.where(seaice_ssp_ukesm_1 < 1e30)
seaice_ssp_ukesm_2 = seaice_ssp_ukesm_2.where(seaice_ssp_ukesm_2 < 1e30)
seaice_ssp_ukesm_3 = seaice_ssp_ukesm_3.where(seaice_ssp_ukesm_3 < 1e30)
seaice_ssp_ukesm_ave = (seaice_ssp_ukesm_1+seaice_ssp_ukesm_2+seaice_ssp_ukesm_3)/3

seaice_ukesm_target = np.array([((seaice_ssp_ukesm_1/1e6).where(seaice_ssp_ukesm_ave.latitude>60)\
    .sel(time=(seaice_ssp_ukesm_ave.time.dt.month==9)).sel(time=slice('2014','2033'))*grid_areas)\
    .sum(('latitude','longitude')).mean('time').values,
                         ((seaice_ssp_ukesm_2/1e6).where(seaice_ssp_ukesm_ave.latitude>60)\
    .sel(time=(seaice_ssp_ukesm_ave.time.dt.month==9)).sel(time=slice('2014','2033'))*grid_areas)\
    .sum(('latitude','longitude')).mean('time').values,
                         ((seaice_ssp_ukesm_3/1e6).where(seaice_ssp_ukesm_ave.latitude>60)\
    .sel(time=(seaice_ssp_ukesm_ave.time.dt.month==9)).sel(time=slice('2014','2033'))*grid_areas)\
    .sum(('latitude','longitude')).mean('time').values])

seaice_ssp_cesm_1 = xr.open_mfdataset(data_dir+'/cesm/*.001.*ICEFRAC*.nc').ICEFRAC
seaice_ssp_cesm_2 = xr.open_mfdataset(data_dir+'/cesm/*.002.*ICEFRAC*.nc').ICEFRAC
seaice_ssp_cesm_3 = xr.open_mfdataset(data_dir+'/cesm/*.003.*ICEFRAC*.nc').ICEFRAC
seaice_ssp_cesm = (seaice_ssp_cesm_1+seaice_ssp_cesm_2+seaice_ssp_cesm_3)/3

seaice_cesm_target = np.array([seaice_ssp_cesm_1.where(cesm_ssp245_ens.lat > 0).weighted(refdata.AREA[0]).sum(('lon','lat'))\
                                .sel(time=(cesm_ssp245_ens.time.dt.month==9)).sel(time=slice('2020','2039')).mean('time')/1e6,
                              seaice_ssp_cesm_2.where(cesm_ssp245_ens.lat > 0).weighted(refdata.AREA[0]).sum(('lon','lat'))\
                                .sel(time=(cesm_ssp245_ens.time.dt.month==9)).sel(time=slice('2020','2039')).mean('time')/1e6,
                              seaice_ssp_cesm_3.where(cesm_ssp245_ens.lat > 0).weighted(refdata.AREA[0]).sum(('lon','lat'))\
                                .sel(time=(cesm_ssp245_ens.time.dt.month==9)).sel(time=slice('2020','2039')).mean('time')/1e6])

e3sm_ICEFRAC_zarr_ctl_1 = xr.open_mfdataset(data_dir+'/e3sm/'+
                       '*.E2_CNTL_01.eam.h0.ICEFRAC.zarr', engine = 'zarr')
e3sm_ICEFRAC_zarr_ctl_2 = xr.open_mfdataset(data_dir+'/e3sm/'+
                       '*.E2_CNTL_02.eam.h0.ICEFRAC.zarr', engine = 'zarr')
e3sm_ICEFRAC_zarr_ctl_3 = xr.open_mfdataset(data_dir+'/e3sm/'+
                       '*.E2_CNTL_03.eam.h0.ICEFRAC.zarr', engine = 'zarr')
e3sm_ICEFRAC_zarr_ctl = (e3sm_ICEFRAC_zarr_ctl_1+e3sm_ICEFRAC_zarr_ctl_2+e3sm_ICEFRAC_zarr_ctl_3)/3
e3sm_ICEFRAC_zarr_amcb_1 = xr.open_mfdataset(data_dir+'/e3sm/'+
                       '*.E2_CNTL_01.ARCTIC_CONTROL.eam.h0.ICEFRAC.zarr', engine = 'zarr')
e3sm_ICEFRAC_zarr_amcb_2 = xr.open_mfdataset(data_dir+'/e3sm/'+
                       '*.E2_CNTL_02.ARCTIC_CONTROL.eam.h0.ICEFRAC.zarr', engine = 'zarr')
e3sm_ICEFRAC_zarr_amcb_3 = xr.open_mfdataset(data_dir+'/e3sm/'+
                       '*.E2_CNTL_03.ARCTIC_CONTROL.eam.h0.ICEFRAC.zarr', engine = 'zarr')
e3sm_ICEFRAC_zarr_amcb = (e3sm_ICEFRAC_zarr_amcb_1+e3sm_ICEFRAC_zarr_amcb_2+e3sm_ICEFRAC_zarr_amcb_3)/3
seaice_e3sm_target = np.array([((e3sm_ICEFRAC_zarr_ctl_1.ICEFRAC/1e6).where(e3sm_ICEFRAC_zarr_ctl_1.lat>0).sel(time=(e3sm_ICEFRAC_zarr_ctl_1.time.dt.month==9))\
                         .where(DSA.grid_center_lat>60).weighted(DSA.grid_area).sum('ncol')*6.371e6**2).sel(time=slice('2029','2048')).mean('time'),
                        ((e3sm_ICEFRAC_zarr_ctl_2.ICEFRAC/1e6).where(e3sm_ICEFRAC_zarr_ctl_2.lat>0).sel(time=(e3sm_ICEFRAC_zarr_ctl_2.time.dt.month==9))\
                         .where(DSA.grid_center_lat>60).weighted(DSA.grid_area).sum('ncol')*6.371e6**2).sel(time=slice('2029','2048')).mean('time'),
                        ((e3sm_ICEFRAC_zarr_ctl_3.ICEFRAC.sel(time=slice('2015','2084'))/1e6).where(e3sm_ICEFRAC_zarr_ctl_3.lat>0).sel(time=(e3sm_ICEFRAC_zarr_ctl_3.sel(time=slice('2015','2084')).time.dt.month==9))\
                         .where(DSA.grid_center_lat>60).weighted(DSA.grid_area).sum('ncol')*6.371e6**2).sel(time=slice('2029','2048')).mean('time')])


# In[34]:


fs=16
fig = plt.figure(figsize=(18,15),dpi=400)
ax1 = fig.add_subplot(3, 3, 1, projection=ccrs.NorthPolarStereo())
p = (seaice_amcb_ukesm_ave*100).sel(time=(seaice_amcb_ukesm_ave.time.dt.month==9))\
    .sel(time=slice('2055','2074')).mean('time').plot.contourf(
      levels=np.linspace(0,100,11),cmap='Blues',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"%"})
(seaice_amcb_ukesm_ave*100).sel(time=(seaice_amcb_ukesm_ave.time.dt.month==9)).sel(time=slice('2055','2074'))\
                                                          .mean('time').plot.contour(levels=[30],
                                                          transform=ccrs.PlateCarree(),
                                                          cmap='k',lw=3)
(seaice_ssp_ukesm_ave*100).sel(time=(seaice_ssp_ukesm_ave.time.dt.month==9)).sel(time=slice('2014','2033'))\
                                                         .mean('time').plot.contour(levels=[30],
                                                         transform=ccrs.PlateCarree(),
                                                         cmap='r',lw=3)
ax1.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax1.set_boundary(circle, transform=ax1.transAxes)
ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
ax1.set_title("(a) UKESM1 Sept Sea Ice Conc (2055-74)", fontsize=fs,fontweight='bold')
ax1.coastlines()

ax1 = fig.add_subplot(3, 3, 2, projection=ccrs.NorthPolarStereo())
p = (seaice_amcb_cesm_ave*100).sel(time=(seaice_amcb_cesm_ave.time.dt.month==9))\
    .sel(time=slice('2055','2074')).mean('time').plot.contourf(
      levels=np.linspace(0,100,11),cmap='Blues',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"%"})
(seaice_amcb_cesm_ave*100).sel(time=(seaice_amcb_cesm_ave.time.dt.month==9)).sel(time=slice('2055','2074'))\
                                                          .mean('time').plot.contour(levels=[30],
                                                          transform=ccrs.PlateCarree(),
                                                          cmap='k',lw=3)
(seaice_ssp_cesm_ave*100).sel(time=(seaice_ssp_cesm_ave.time.dt.month==9)).sel(time=slice('2020','2039'))\
                                                         .mean('time').plot.contour(levels=[30],
                                                         transform=ccrs.PlateCarree(),
                                                         cmap='r',lw=3)
ax1.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax1.set_boundary(circle, transform=ax1.transAxes)
ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
ax1.set_title("(b) CESM2 Sept Sea Ice Conc (2055-74)", fontsize=fs,fontweight='bold')
ax1.coastlines()

ax1 = fig.add_subplot(3, 3, 3, projection=ccrs.NorthPolarStereo())
p = (seaice_amcb_e3sm_ave*100).sel(time=(seaice_amcb_e3sm_ave.time.dt.month==9))\
    .sel(time=slice('2055','2074')).mean('time').plot.contourf(
      levels=np.linspace(0,100,11),cmap='Blues',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"%"})
(seaice_amcb_e3sm_ave*100).sel(time=(seaice_amcb_e3sm_ave.time.dt.month==9)).sel(time=slice('2055','2074'))\
                                                          .mean('time').plot.contour(levels=[30],
                                                          transform=ccrs.PlateCarree(),
                                                          cmap='k',lw=3)
(seaice_ssp_e3sm_ave*100).sel(time=(seaice_ssp_e3sm_ave.time.dt.month==9)).sel(time=slice('2029','2048'))\
                                                         .mean('time').plot.contour(levels=[30],
                                                         transform=ccrs.PlateCarree(),
                                                         cmap='r',lw=3)
ax1.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax1.set_boundary(circle, transform=ax1.transAxes)
ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
ax1.set_title("(c) E3SM Sept Sea Ice Conc (2055-74)", fontsize=fs,fontweight='bold')
ax1.coastlines()

ax1 = fig.add_subplot(3, 3, 4)
plt.fill_between(seaice_ssp_ukesm_ave.time.groupby('time.year').mean('time').year,np.mean(seaice_ukesm_target)-np.std(seaice_ukesm_target),
                 np.mean(seaice_ukesm_target)+np.std(seaice_ukesm_target),alpha=0.2,facecolor='black')
plt.axhline(y=np.mean(seaice_ukesm_target),lw=3,c='k',label='Target period')
plt.axhline(y=np.mean(seaice_ukesm_target),lw=8,c='k',xmin=0,xmax=8/55)
plt.plot(seaice_ssp_ukesm_ave.time.groupby('time.year').mean('time').year,
         ((seaice_ssp_ukesm_ave/1e6).where(seaice_ssp_ukesm_ave.latitude>60)\
         .sel(time=(seaice_ssp_ukesm_ave.time.dt.month==9))*grid_areas).sum(('latitude','longitude')),lw=3,c='r',label='SSP2-4.5')
plt.plot(seaice_amcb_ukesm_ave.time.groupby('time.year').mean('time').year,
         ((seaice_amcb_ukesm_ave/1e6).where(seaice_amcb_ukesm_ave.latitude>60)\
          .sel(time=(seaice_amcb_ukesm_ave.time.dt.month==9))*grid_areas).sum(('latitude','longitude')),lw=3,c='b',label='Arctic MCB')
for sim in [seaice_ssp_ukesm_1,seaice_ssp_ukesm_2,seaice_ssp_ukesm_3]:
    plt.plot(seaice_ssp_ukesm_1.time.groupby('time.year').mean('time').year,
             ((sim/1e6).where(sim.latitude>0).sel(time=(sim.time.dt.month==9))*grid_areas).sum(('latitude','longitude')),lw=1,alpha=0.5,c='r')
for sim in [seaice_amcb_ukesm_1,seaice_amcb_ukesm_2,seaice_amcb_ukesm_3]:
    plt.plot(seaice_amcb_ukesm_ave.time.groupby('time.year').mean('time').year,
             ((sim/1e6).where(sim.latitude>0).sel(time=(sim.time.dt.month==9))*grid_areas).sum(('latitude','longitude')),lw=1,alpha=0.5,c='b')
plt.title('(d) UKESM1 Arctic Sept Sea Ice Extent', fontsize=fs,fontweight='bold')
plt.grid()
plt.xlabel('Year',fontweight='bold')
plt.ylabel('SIE (km$^2$)',fontweight='bold')
plt.xlim(2020,2074)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend()
plt.ylim([0,6e6])

ax1 = fig.add_subplot(3, 3, 5)
plt.fill_between(seaice_ssp_cesm_ave.time.groupby('time.year').mean('time').year,np.mean(seaice_cesm_target)-np.std(seaice_cesm_target),
                 np.mean(seaice_cesm_target)+np.std(seaice_cesm_target),alpha=0.2,facecolor='black')
plt.axhline(y=np.mean(seaice_cesm_target),lw=3,c='k',label='Target period')
plt.axhline(y=np.mean(seaice_cesm_target),lw=8,c='k',xmin=5/55,xmax=14/55)
plt.plot(seaice_ssp_cesm.time.groupby('time.year').mean('time').year.isel(year=slice(0,86)),
         (seaice_ssp_cesm/1e6).where(cesm_ssp245_ens.lat > 0).weighted(refdata.AREA[0]).sum(('lon','lat'))\
         .sel(time=(seaice_ssp_cesm.time.dt.month==9)),lw=3,c='r',label='SSP2-4.5')
plt.plot(seaice_amcb_cesm_ave.time.groupby('time.year').mean('time').year,
         (seaice_amcb_cesm_ave/1e6).where(cesm_ssp245_ens.lat > 0).weighted(refdata.AREA[0]).sum(('lon','lat'))\
         .sel(time=(seaice_amcb_cesm_ave.time.dt.month==9)),lw=3,c='b',label='Arctic MCB')
for sim in [seaice_ssp_cesm_1,seaice_ssp_cesm_2,seaice_ssp_cesm_3]:
    plt.plot(sim.time.groupby('time.year').mean('time').year.isel(year=slice(0,86)),
         (sim/1e6).where(sim.lat > 0).weighted(refdata.AREA[0]).sum(('lon','lat'))\
         .sel(time=(sim.time.dt.month==9)),lw=1,alpha=0.5,c='r')
for sim in [seaice_amcb_cesm_1,seaice_amcb_cesm_2,seaice_amcb_cesm_3]:
    plt.plot(sim.time.groupby('time.year').mean('time').year,
         (sim/1e6).where(cesm_ssp245_ens.lat > 0).weighted(refdata.AREA[0]).sum(('lon','lat'))\
         .sel(time=(sim.time.dt.month==9)),lw=1,alpha=0.5,c='b')
plt.title('(e) CESM2 Arctic Sept Sea Ice Extent', fontsize=fs,fontweight='bold')
plt.grid()
plt.xlabel('Year',fontweight='bold')
plt.ylabel('SIE (km$^2$)',fontweight='bold')
plt.xlim(2020,2074)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend()
plt.ylim([0,6e6])

ax1 = fig.add_subplot(3, 3, 6)
plt.axhline(y=np.mean(seaice_e3sm_target),c='k',lw=3,label='Target period')
plt.axhline(y=np.mean(seaice_e3sm_target),lw=8,c='k',xmin=14/55,xmax=23/55)
plt.fill_between(seaice_ssp_e3sm_ave.time.groupby('time.year').mean('time').year,np.mean(seaice_e3sm_target)-np.std(seaice_e3sm_target),
                 np.mean(seaice_e3sm_target)+np.std(seaice_e3sm_target),alpha=0.2,facecolor='black')
plt.plot(seaice_ssp_e3sm_ave.sel(time=slice('2015','2084')).time.groupby('time.year').mean('time').year,
         ((e3sm_ICEFRAC_zarr_ctl.ICEFRAC/1e6).where(e3sm_ICEFRAC_zarr_ctl.lat>0).sel(time=(e3sm_ICEFRAC_zarr_ctl.time.dt.month==9))\
         .where(DSA.grid_center_lat>60).weighted(DSA.grid_area).sum('ncol')*6.371e6**2),lw=3,c='r',label='SSP2-4.5')
plt.plot(seaice_ssp_e3sm_ave.sel(time=slice('2036','2074')).time.groupby('time.year').mean('time').year,
         ((e3sm_ICEFRAC_zarr_amcb.ICEFRAC/1e6).where(e3sm_ICEFRAC_zarr_ctl.lat>0).sel(time=(e3sm_ICEFRAC_zarr_amcb.time.dt.month==9))\
         .where(DSA.grid_center_lat>60).weighted(DSA.grid_area).sum('ncol')*6.371e6**2),lw=3,c='b',label='Arctic MCB')

for sim in [e3sm_ICEFRAC_zarr_ctl_1.ICEFRAC,e3sm_ICEFRAC_zarr_ctl_2.ICEFRAC,e3sm_ICEFRAC_zarr_ctl_3.ICEFRAC]:
    plt.plot(sim.time.groupby('time.year').mean('time').year.isel(year=slice(0,70)),
         ((sim/1e6).where(e3sm_ICEFRAC_zarr_ctl.lat>0).sel(time=(sim.time.dt.month==9))\
         .where(DSA.grid_center_lat>60).weighted(DSA.grid_area).sum('ncol')*6.371e6**2).sel(time=slice('2015','2084')),lw=1,c='r',alpha=0.5)
for sim in [e3sm_ICEFRAC_zarr_amcb_1.ICEFRAC,e3sm_ICEFRAC_zarr_amcb_2.ICEFRAC,e3sm_ICEFRAC_zarr_amcb_3.ICEFRAC]:
    plt.plot(seaice_ssp_e3sm_ave.sel(time=slice('2036','2074')).time.groupby('time.year').mean('time').year,
         ((sim/1e6).where(e3sm_ICEFRAC_zarr_ctl.lat>0).sel(time=(e3sm_ICEFRAC_zarr_amcb.time.dt.month==9))\
         .where(DSA.grid_center_lat>60).weighted(DSA.grid_area).sum('ncol')*6.371e6**2),lw=1,c='b',alpha=0.5)
plt.title('(f) E3SM Arctic Sept Sea Ice Extent', fontsize=fs,fontweight='bold')
plt.grid()
plt.xlabel('Year',fontweight='bold')
plt.ylabel('SIE (km$^2$)',fontweight='bold')
plt.xlim(2020,2075)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend()
plt.ylim([0,6e6])

ax1 = fig.add_subplot(3, 3, 7)
((seaice_ssp_ukesm_ave/1e6).where(seaice_ssp_ukesm_ave.latitude>60)*grid_areas).sum(('latitude','longitude'))\
.sel(time=slice('2014','2033')).groupby('time.month').mean('time').plot(lw=3,label='Target period',c='k')
((seaice_ssp_ukesm_ave/1e6).where(seaice_ssp_ukesm_ave.latitude>60)*grid_areas).sum(('latitude','longitude'))\
.sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(lw=3,label='SSP2-4.5',c='r')
((seaice_amcb_ukesm_ave/1e6).where(seaice_amcb_ukesm_ave.latitude>60)*grid_areas).sum(('latitude','longitude'))\
.sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(lw=3,label='Arctic MCB',c='b')
plt.title('(g) Arctic sea ice extent (UKESM1, 2055-74)', fontsize=fs,fontweight='bold')
plt.grid()
plt.xlabel('Month',fontweight='bold')
plt.ylabel('SIE (km$^2$)',fontweight='bold')
plt.xlim([1,12])
plt.xticks(np.arange(1,13), ['J','F','M','A','M','J','J','A','S','O','N','D'],fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,1.5e7])
plt.legend()

ax1 = fig.add_subplot(3, 3, 8)
(cesm_ssp245_ens.ICEFRAC/1e6).where(cesm_ssp245_ens.lat > 60).weighted(refdata.AREA[0]).sum(('lon','lat'))\
.sel(time=slice('2020','2039')).groupby('time.month').mean('time').plot(lw=3,label='Target period',c='k')
(cesm_ssp245_ens.ICEFRAC/1e6).where(cesm_ssp245_ens.lat > 60).weighted(refdata.AREA[0]).sum(('lon','lat'))\
.sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(lw=3,label='SSP2-4.5',c='r')
(seaice_amcb_cesm_ave/1e6).where(cesm_ssp245_ens.lat > 60).weighted(refdata.AREA[0]).sum(('lon','lat'))\
.sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(lw=3,label='Arctic MCB',c='b')
plt.title('(h) Arctic sea ice extent (CESM2, 2055-74)', fontsize=fs,fontweight='bold')
plt.grid()
plt.xlabel('Month',fontweight='bold')
plt.ylabel('SIE (km$^2$)',fontweight='bold')
plt.xlim([1,12])
plt.xticks(np.arange(1,13), ['J','F','M','A','M','J','J','A','S','O','N','D'],fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,1.5e7])
plt.legend()

ax1 = fig.add_subplot(3, 3, 9)
((e3sm_ICEFRAC_zarr_ctl.ICEFRAC/1e6).where(DSA.grid_center_lat>60).sel(time=slice('2029','2048'))\
 .groupby('time.month').mean('time').weighted(DSA.grid_area).sum('ncol')*6.371e6**2).roll(month=-1).plot(lw=3,c='k',label='Target period')
((e3sm_ICEFRAC_zarr_ctl.ICEFRAC/1e6).where(DSA.grid_center_lat>60).sel(time=slice('2055','2074'))\
 .groupby('time.month').mean('time').weighted(DSA.grid_area).sum('ncol')*6.371e6**2).roll(month=-1).plot(lw=3,c='r',label='SSP2-4.5')
((e3sm_ICEFRAC_zarr_amcb.ICEFRAC/1e6).where(DSA.grid_center_lat>60).sel(time=slice('2055','2074'))\
 .groupby('time.month').mean('time').weighted(DSA.grid_area).sum('ncol')*6.371e6**2).roll(month=-1).plot(lw=3,c='b',label='Arctic MCB')
plt.title('(i) Arctic sea ice extent (E3SMv2, 2055-74)', fontsize=fs,fontweight='bold')
plt.grid()
plt.xlabel('Month',fontweight='bold')
plt.ylabel('SIE (km$^2$)',fontweight='bold')
plt.xlim([1,12])
plt.xticks(np.arange(1,13), ['J','F','M','A','M','J','J','A','S','O','N','D'],fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,1.5e7])
plt.legend()
plt.tight_layout()
plt.savefig('./figs/fig_amcb_3.pdf',bbox_inches='tight')
plt.show()


# Precipitation map / plot

# Get T-test p value for each plot

# In[41]:


ukesm_pr_hist_1 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_1_ref.nc').precipitation_flux.drop(['forecast_period','forecast_reference_time']).rename({'latitude':'lat','longitude':'lon'})
ukesm_pr_hist_2 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_2_ref.nc').precipitation_flux.drop(['forecast_period','forecast_reference_time']).rename({'latitude':'lat','longitude':'lon'})
ukesm_pr_hist_3 = xr.open_mfdataset(data_dir+'ukesm/amcb/hist_3_ref.nc').precipitation_flux.drop(['forecast_period','forecast_reference_time']).rename({'latitude':'lat','longitude':'lon'})
ukesm_ssp_pr_1 = xr.open_mfdataset('/data/users/managecmip/champ/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp245/r1i1p1f2/Amon/pr/gn/v20190507/*.nc').pr
ukesm_ssp_pr_2 = xr.open_mfdataset('/data/users/managecmip/champ/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp245/r2i1p1f2/Amon/pr/gn/v20190507/*.nc').pr
ukesm_ssp_pr_3 = xr.open_mfdataset('/data/users/managecmip/champ/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp245/r3i1p1f2/Amon/pr/gn/v20190507/*.nc').pr
#Add 2014 to ukesm data
ukesm_ssp_pr_1 = xr.concat([ukesm_pr_hist_1,ukesm_ssp_pr_1],dim='time')
ukesm_ssp_pr_2 = xr.concat([ukesm_pr_hist_2,ukesm_ssp_pr_2],dim='time')
ukesm_ssp_pr_3 = xr.concat([ukesm_pr_hist_3,ukesm_ssp_pr_3],dim='time')

ukesm_ssp_pr_arr = [ukesm_ssp_pr_1,ukesm_ssp_pr_2,ukesm_ssp_pr_3]
ukesm_ssp_pr_ave = (ukesm_ssp_pr_1+ukesm_ssp_pr_2+ukesm_ssp_pr_3)/3

ukesm_amcb_pr_1 = xr.open_dataarray(data_dir+'/ukesm/pr_amcb_ukesm_1.nc').rename({'latitude':'lat','longitude':'lon'})
ukesm_amcb_pr_2 = xr.open_dataarray(data_dir+'/ukesm/pr_amcb_ukesm_2.nc').rename({'latitude':'lat','longitude':'lon'})
ukesm_amcb_pr_3 = xr.open_dataarray(data_dir+'/ukesm/pr_amcb_ukesm_3.nc').rename({'latitude':'lat','longitude':'lon'})
ukesm_amcb_pr_ave = (ukesm_amcb_pr_1+ukesm_amcb_pr_2+ukesm_amcb_pr_3)/3


# In[42]:


r1 = (ukesm_ssp_pr_arr[0]*86400).sel(time=(ukesm_ssp_pr_arr[0].time.dt.season=='JJA'))\
     .sel(time=slice("2055","2074")).groupby('time.year').mean('time')
for i in range(1,3):
    r1 = xr.concat((r1,(ukesm_ssp_pr_arr[i]*86400).sel(time=(ukesm_ssp_pr_arr[i].time.dt.season=='JJA'))\
     .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

r2 = (ukesm_amcb_pr_1*86400)\
     .sel(time=(ukesm_amcb_pr_1.time.dt.season=='JJA'))\
     .sel(time=slice("2055","2074")).groupby('time.year').mean('time')
for sim in [ukesm_amcb_pr_2,ukesm_amcb_pr_3]:
    r2 = xr.concat((r2,(sim*86400)\
                    .sel(time=(sim.time.dt.season=='JJA'))\
                    .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

stat, pval = stats.ttest_ind(r1, r2)
pval_ukesm_jja = xr.DataArray(
    data=stats.false_discovery_control(pval),
    dims=["lat", "lon"],
    coords=dict(
        lon=(["lon"], r1.lon.values),
        lat=(["lat"], r1.lat.values),
    ),
)

r1 = (ukesm_ssp_pr_arr[0]*86400).sel(time=(ukesm_ssp_pr_arr[0].time.dt.season=='DJF'))\
     .sel(time=slice("2055","2074")).groupby('time.year').mean('time')
for i in range(1,3):
    r1 = xr.concat((r1,(ukesm_ssp_pr_arr[i]*86400).sel(time=(ukesm_ssp_pr_arr[i].time.dt.season=='DJF'))\
     .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

r2 = (ukesm_amcb_pr_1*86400)\
     .sel(time=(ukesm_amcb_pr_1.time.dt.season=='DJF'))\
     .sel(time=slice("2055","2074")).groupby('time.year').mean('time')
for sim in [ukesm_amcb_pr_2,ukesm_amcb_pr_3]:
    r2 = xr.concat((r2,(sim*86400)\
                    .sel(time=(sim.time.dt.season=='DJF'))\
                    .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

stat, pval = stats.ttest_ind(r1, r2)
pval_ukesm_djf = xr.DataArray(
    data=stats.false_discovery_control(pval),
    dims=["lat", "lon"],
    coords=dict(
        lon=(["lon"], r1.lon.values),
        lat=(["lat"], r1.lat.values),
    ),
)


# Get CESM precip pvals

# In[43]:


cesm_prec_ssp_1 = xr.open_mfdataset(data_dir+'cesm/*.001.*PRECT*.nc')
cesm_prec_ssp_2 = xr.open_mfdataset(data_dir+'cesm/*.002.*PRECT*.nc')
cesm_prec_ssp_3 = xr.open_mfdataset(data_dir+'cesm/*.003.*PRECT*.nc')
cesm_prec_ssp_1['time'] = cesm_prec_ssp_1.indexes['time'].shift(-15,"D")
cesm_prec_ssp_2['time'] = cesm_prec_ssp_2.indexes['time'].shift(-15,"D")
cesm_prec_ssp_3['time'] = cesm_prec_ssp_3.indexes['time'].shift(-15,"D")
cesm_prec_ssp_arr = [cesm_prec_ssp_1, cesm_prec_ssp_2, cesm_prec_ssp_3]


# In[44]:


r1 = xr.concat(((cesm_prec_ssp_1.PRECT*1000*86400).sel(time=(cesm_prec_ssp_1.time.dt.season=='JJA'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                 (cesm_prec_ssp_2.PRECT*1000*86400).sel(time=(cesm_prec_ssp_2.time.dt.season=='JJA'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                 (cesm_prec_ssp_3.PRECT*1000*86400).sel(time=(cesm_prec_ssp_3.time.dt.season=='JJA'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

r2 = xr.concat(((cesm_mcb_data[cesm_mcb_exp[0]].PRECT*1000*86400).sel(time=(cesm_mcb_data[cesm_mcb_exp[0]].time.dt.season=='JJA'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                 (cesm_mcb_data[cesm_mcb_exp[1]].PRECT*1000*86400).sel(time=(cesm_mcb_data[cesm_mcb_exp[1]].time.dt.season=='JJA'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                 (cesm_mcb_data[cesm_mcb_exp[2]].PRECT*1000*86400).sel(time=(cesm_mcb_data[cesm_mcb_exp[2]].time.dt.season=='JJA'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

mean_ssp245 = r1.mean("year")
mean_mcb = r2.mean("year")
std_dev_ssp245 = r1.std("year")
std_dev_mcb = r2.std("year")

stddev = np.sqrt(std_dev_mcb**2*(3*20 - 1) + std_dev_ssp245**2*(3*20 - 1))/np.sqrt(3*20 + 3*20 - 2)
tstat = (mean_mcb - mean_ssp245)/stddev/np.sqrt(1/(3*20) + 1/(3*20))
pval = stats.t.sf(np.abs(tstat), 3*20 + 3*20 - 2)

pval_cesm_jja = xr.DataArray(
    data=stats.false_discovery_control(pval),
    dims=["lat", "lon"],
    coords=dict(
        lon=(["lon"], r1.lon.values),
        lat=(["lat"], r1.lat.values),
    ),
)

r1 = xr.concat(((cesm_prec_ssp_1.PRECT*1000*86400).sel(time=(cesm_prec_ssp_1.time.dt.season=='DJF'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                 (cesm_prec_ssp_2.PRECT*1000*86400).sel(time=(cesm_prec_ssp_2.time.dt.season=='DJF'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                 (cesm_prec_ssp_3.PRECT*1000*86400).sel(time=(cesm_prec_ssp_3.time.dt.season=='DJF'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

r2 = xr.concat(((cesm_mcb_data[cesm_mcb_exp[0]].PRECT*1000*86400).sel(time=(cesm_mcb_data[cesm_mcb_exp[0]].time.dt.season=='DJF'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                 (cesm_mcb_data[cesm_mcb_exp[1]].PRECT*1000*86400).sel(time=(cesm_mcb_data[cesm_mcb_exp[1]].time.dt.season=='DJF'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                 (cesm_mcb_data[cesm_mcb_exp[2]].PRECT*1000*86400).sel(time=(cesm_mcb_data[cesm_mcb_exp[2]].time.dt.season=='DJF'))\
                 .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

mean_ssp245 = r1.mean("year")
mean_mcb = r2.mean("year")
std_dev_ssp245 = r1.std("year")
std_dev_mcb = r2.std("year")

stddev = np.sqrt(std_dev_mcb**2*(3*20 - 1) + std_dev_ssp245**2*(3*20 - 1))/np.sqrt(3*20 + 3*20 - 2)
tstat = (mean_mcb - mean_ssp245)/stddev/np.sqrt(1/(3*20) + 1/(3*20))
pval = stats.t.sf(np.abs(tstat), 3*20 + 3*20 - 2)

pval_cesm_djf = xr.DataArray(
    data=stats.false_discovery_control(pval),
    dims=["lat", "lon"],
    coords=dict(
        lon=(["lon"], r1.lon.values),
        lat=(["lat"], r1.lat.values),
    ),
)


# Get E3SM pvals

# In[45]:


e3sm_PRECT_ctl_1 = xr.open_dataarray(data_dir+'e3sm/E2_CNTL_01_PRECC.nc')+\
                   xr.open_dataarray(data_dir+'e3sm/E2_CNTL_01_PRECL.nc')
e3sm_PRECT_ctl_2 = xr.open_dataarray(data_dir+'e3sm/E2_CNTL_02_PRECC.nc')+\
                   xr.open_dataarray(data_dir+'e3sm/E2_CNTL_02_PRECL.nc')
e3sm_PRECT_ctl_3 = xr.open_dataarray(data_dir+'e3sm/E2_CNTL_03_PRECC.nc')+\
                   xr.open_dataarray(data_dir+'e3sm/E2_CNTL_03_PRECL.nc')

e3sm_PRECT_amcb_1 = xr.open_dataarray(data_dir+'e3sm/E2_CNTL_01.ARCTIC_CONTROL_PRECC.nc')+\
                    xr.open_dataarray(data_dir+'e3sm/E2_CNTL_01.ARCTIC_CONTROL_PRECL.nc')
e3sm_PRECT_amcb_2 = xr.open_dataarray(data_dir+'e3sm/E2_CNTL_02.ARCTIC_CONTROL_PRECC.nc')+\
                    xr.open_dataarray(data_dir+'e3sm/E2_CNTL_02.ARCTIC_CONTROL_PRECL.nc')
e3sm_PRECT_amcb_3 = xr.open_dataarray(data_dir+'e3sm/E2_CNTL_03.ARCTIC_CONTROL_PRECC.nc')+\
                    xr.open_dataarray(data_dir+'e3sm/E2_CNTL_03.ARCTIC_CONTROL_PRECL.nc')


e3sm_PRECT_ctl_1['time'] = e3sm_PRECT_ctl_1.indexes['time'].shift(-15,"D")
e3sm_PRECT_ctl_2['time'] = e3sm_PRECT_ctl_2.indexes['time'].shift(-15,"D")
e3sm_PRECT_ctl_3['time'] = e3sm_PRECT_ctl_3.indexes['time'].shift(-15,"D")
e3sm_PRECT_amcb_1['time'] = e3sm_PRECT_amcb_1.indexes['time'].shift(-15,"D")
e3sm_PRECT_amcb_2['time'] = e3sm_PRECT_amcb_2.indexes['time'].shift(-15,"D")
e3sm_PRECT_amcb_3['time'] = e3sm_PRECT_amcb_3.indexes['time'].shift(-15,"D")


# In[46]:


r1 = xr.concat(((e3sm_PRECT_ctl_1*86400*1000).sel(time=(e3sm_PRECT_ctl_1.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                (e3sm_PRECT_ctl_2*86400*1000).sel(time=(e3sm_PRECT_ctl_2.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                (e3sm_PRECT_ctl_3*86400*1000).sel(time=(e3sm_PRECT_ctl_3.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

r2 = xr.concat(((e3sm_PRECT_amcb_1*86400*1000).sel(time=(e3sm_PRECT_amcb_1.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                (e3sm_PRECT_amcb_2*86400*1000).sel(time=(e3sm_PRECT_amcb_2.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                (e3sm_PRECT_amcb_3*86400*1000).sel(time=(e3sm_PRECT_amcb_3.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

stat, pval = stats.ttest_ind(r1.transpose("year","lat","lon"), 
                             r2.transpose("year","lat","lon"))
pval_e3sm_jja = xr.DataArray(
    data=stats.false_discovery_control(pval),
    dims=["lat", "lon"],
    coords=dict(
        lon=(["lon"], r1.lon.values),
        lat=(["lat"], r1.lat.values),
    ),
)

r1 = xr.concat(((e3sm_PRECT_ctl_1*86400*1000).sel(time=(e3sm_PRECT_ctl_1.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                (e3sm_PRECT_ctl_2*86400*1000).sel(time=(e3sm_PRECT_ctl_2.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                (e3sm_PRECT_ctl_3*86400*1000).sel(time=(e3sm_PRECT_ctl_3.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

r2 = xr.concat(((e3sm_PRECT_amcb_1*86400*1000).sel(time=(e3sm_PRECT_amcb_1.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                (e3sm_PRECT_amcb_2*86400*1000).sel(time=(e3sm_PRECT_amcb_2.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time'),
                (e3sm_PRECT_amcb_3*86400*1000).sel(time=(e3sm_PRECT_amcb_3.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).groupby('time.year').mean('time')), dim='year')

stat, pval = stats.ttest_ind(r1.transpose("year","lat","lon"), 
                             r2.transpose("year","lat","lon"))
pval_e3sm_djf = xr.DataArray(
    data=stats.false_discovery_control(pval),
    dims=["lat", "lon"],
    coords=dict(
        lon=(["lon"], r1.lon.values),
        lat=(["lat"], r1.lat.values),
    ),
)


# In[51]:


levels_p=[-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1]

fig = plt.figure(figsize=(12,10),dpi=400)

ax3 = fig.add_subplot(3,2, 1, projection=ccrs.Robinson())
data = ((ukesm_amcb_pr_ave*86400)\
       .sel(time=(ukesm_amcb_pr_ave.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).mean('time')-\
     (ukesm_ssp_pr_ave*86400)\
       .sel(time=(ukesm_ssp_pr_ave.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).mean('time'))
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
       levels=levels_p,cmap='BrBG',
       subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
       transform=ccrs.PlateCarree(),
       extend = "both",
       cbar_kwargs={"shrink": 0.5, "label":"mm/day"})
data = pval_ukesm_djf
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
data_diff_cyclic.where(data_diff_cyclic>0.05).plot.contourf(levels=[0,1], colors='none', hatches=['///////'],add_colorbar=False,\
                subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),transform=ccrs.PlateCarree())
ax3.set_title("(a) UKESM1 Precip DJF (MCB - SSP2-4.5, 2055-74)", fontsize=10,fontweight='bold')
ax3.coastlines()

ax3 = fig.add_subplot(3,2, 2, projection=ccrs.Robinson())
data = ((ukesm_amcb_pr_ave*86400)\
       .sel(time=(ukesm_amcb_pr_ave.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).mean('time')-\
     (ukesm_ssp_pr_ave*86400)\
       .sel(time=(ukesm_ssp_pr_ave.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).mean('time'))
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
       levels=levels_p,cmap='BrBG',
       subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
       transform=ccrs.PlateCarree(),
       extend = "both",
       cbar_kwargs={"shrink": 0.5, "label":"mm/day"})
data = pval_ukesm_jja
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
data_diff_cyclic.where(data_diff_cyclic>0.05).plot.contourf(levels=[0,1], colors='none', hatches=['///////'],add_colorbar=False,\
                subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),transform=ccrs.PlateCarree())
ax3.set_title("(b) UKESM1 Precip JJA (MCB - SSP2-4.5, 2055-74)", fontsize=10,fontweight='bold')
ax3.coastlines()

ax3 = fig.add_subplot(3,2, 3, projection=ccrs.Robinson())
data = ((((cesm_mcb_data[cesm_mcb_exp[0]].PRECT+
       cesm_mcb_data[cesm_mcb_exp[1]].PRECT+
       cesm_mcb_data[cesm_mcb_exp[2]].PRECT)/3)*86400*1000)\
       .sel(time=(cesm_mcb_data[cesm_mcb_exp[0]].time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).mean('time')-\
    (cesm_ssp245_ens.PRECT*86400*1000).sel(time=(cesm_ssp245_ens.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).mean('time'))
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
       levels=levels_p,cmap='BrBG',
       subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
       transform=ccrs.PlateCarree(),
       extend = "both",
       cbar_kwargs={"shrink": 0.5, "label":"mm/day"})
data = pval_cesm_djf
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
data_diff_cyclic.where(data_diff_cyclic>0.05).plot.contourf(levels=[0,1], colors='none', hatches=['///////'],add_colorbar=False,\
                subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),transform=ccrs.PlateCarree())
ax3.set_title("(c) CESM2 Precip DJF (MCB - SSP2-4.5, 2055-74)", fontsize=10,fontweight='bold')
ax3.coastlines()

ax3 = fig.add_subplot(3,2, 4, projection=ccrs.Robinson())
data = ((((cesm_mcb_data[cesm_mcb_exp[0]].PRECT+
       cesm_mcb_data[cesm_mcb_exp[1]].PRECT+
       cesm_mcb_data[cesm_mcb_exp[2]].PRECT)/3)*86400*1000)\
       .sel(time=(cesm_mcb_data[cesm_mcb_exp[0]].time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).mean('time')-\
    (cesm_ssp245_ens.PRECT*86400*1000).sel(time=(cesm_ssp245_ens.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).mean('time'))
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
p = data_diff_cyclic.plot.contourf(
       levels=levels_p,cmap='BrBG',
       subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
       transform=ccrs.PlateCarree(),
       extend = "both",
       cbar_kwargs={"shrink": 0.5, "label":"mm/day"})
data = pval_cesm_jja
if 'lon' in data.coords:
    data_diff_cyclic = xr.concat([data,data.isel(lon=0).assign_coords(lon=361)],dim='lon')
else:
    data_diff_cyclic = xr.concat([data,data.isel(longitude=0).assign_coords(longitude=361)],dim='longitude')
data_diff_cyclic.where(data_diff_cyclic>0.05).plot.contourf(levels=[0,1], colors='none', hatches=['///////'],add_colorbar=False,\
                subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),transform=ccrs.PlateCarree())
ax3.set_title("(d) CESM2 Precip JJA (MCB - SSP2-4.5, 2055-74)", fontsize=10,fontweight='bold')
ax3.coastlines()

ax3 = fig.add_subplot(3,2, 5, projection=ccrs.Robinson())
p = ((((e3sm_PRECT_amcb_1+e3sm_PRECT_amcb_2+e3sm_PRECT_amcb_3)/3)*86400*1000)\
       .sel(time=(e3sm_PRECT_amcb_1.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).mean('time')-\
    (((e3sm_PRECT_ctl_1+e3sm_PRECT_ctl_2+e3sm_PRECT_ctl_3)/3)*86400*1000)\
       .sel(time=(e3sm_PRECT_ctl_1.time.dt.season=='DJF'))\
       .sel(time=slice("2055","2074")).mean('time')).plot.contourf(
       levels=levels_p,cmap='BrBG',
       subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
       transform=ccrs.PlateCarree(),
       extend = "both",
       cbar_kwargs={"shrink": 0.5, "label":"mm/day"})
pval_e3sm_djf.where(pval_e3sm_djf>0.05).plot.contourf(levels=[0,1], colors='none', hatches=['///////'],add_colorbar=False,\
                subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),transform=ccrs.PlateCarree())
ax3.set_title("(e) E3SM Precip DJF (MCB - SSP2-4.5, 2055-74)", fontsize=10,fontweight='bold')
ax3.coastlines()

ax3 = fig.add_subplot(3,2, 6, projection=ccrs.Robinson())
p = ((((e3sm_PRECT_amcb_1+e3sm_PRECT_amcb_2+e3sm_PRECT_amcb_3)/3)*86400*1000)\
       .sel(time=(e3sm_PRECT_amcb_1.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).mean('time')-\
    (((e3sm_PRECT_ctl_1+e3sm_PRECT_ctl_2+e3sm_PRECT_ctl_3)/3)*86400*1000).sel(time=(e3sm_PRECT_ctl_1.time.dt.season=='JJA'))\
       .sel(time=slice("2055","2074")).mean('time')).plot.contourf(
       levels=levels_p,cmap='BrBG',
       subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),
       transform=ccrs.PlateCarree(),
       extend = "both",
       cbar_kwargs={"shrink": 0.5, "label":"mm/day"})
pval_e3sm_jja.where(pval_e3sm_jja>0.05).plot.contourf(levels=[0,1], colors='none', hatches=['///////'],add_colorbar=False,\
                subplot_kws=dict(projection=ccrs.Robinson(), facecolor="gray"),transform=ccrs.PlateCarree())
ax3.add_patch(mpatches.Rectangle(xy=[-20,10], width=30, height=10,
                                facecolor='none',edgecolor='r',
                                transform=ccrs.PlateCarree(),lw=2))
ax3.set_title("(f) E3SM Precip JJA (MCB - SSP2-4.5, 2055-74)", fontsize=10,fontweight='bold')
ax3.coastlines()

plt.tight_layout()
plt.savefig('./figs/fig_amcb_4.pdf',bbox_inches='tight')
plt.show()


# In[42]:


fig = plt.figure(figsize=(16,4),dpi=400)
ax3 = fig.add_subplot(1,3,1)
(ukesm_ssp_pr_ave*86400).sel(lat=slice(10,20))\
    .where((ukesm_ssp_pr_ave.lon<=10)|(ukesm_ssp_pr_ave.lon>=340))\
    .weighted(weights_ukesm_ssp).mean(('lat','lon')).sel(time=slice('2014','2033')).groupby('time.month').mean('time')\
    .plot(c='k',lw=3,label='SSP2-4.5 2014-33')
(ukesm_ssp_pr_ave*86400).sel(lat=slice(10,20))\
    .where((ukesm_ssp_pr_ave.lon<=10)|(ukesm_ssp_pr_ave.lon>=340))\
    .weighted(weights_ukesm_ssp).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time')\
    .plot(c='r',lw=3,label='SSP2-4.5 2055-74')
(ukesm_amcb_pr_ave*86400).sel(lat=slice(10,20))\
    .where((ukesm_amcb_pr_ave.lon<=10)|(ukesm_amcb_pr_ave.lon>=340))\
    .weighted(weights_ukesm_ssp).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time')\
    .plot(c='b',lw=3,label='AMCB 2055-74')

(ukesm_amcb_pr_1*86400).sel(lat=slice(10,20))\
    .where((ukesm_amcb_pr_ave.lon<=10)|(ukesm_amcb_pr_ave.lon>=340))\
    .weighted(weights_ukesm_ssp).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time')\
    .plot(c='b',lw=1)
(ukesm_amcb_pr_2*86400).sel(lat=slice(10,20))\
    .where((ukesm_amcb_pr_ave.lon<=10)|(ukesm_amcb_pr_ave.lon>=340))\
    .weighted(weights_ukesm_ssp).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time')\
    .plot(c='b',lw=1)
(ukesm_amcb_pr_3*86400).sel(lat=slice(10,20))\
    .where((ukesm_amcb_pr_ave.lon<=10)|(ukesm_amcb_pr_ave.lon>=340))\
    .weighted(weights_ukesm_ssp).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time')\
    .plot(c='b',lw=1)
plt.title('(a) Sahel Precipitation (UKESM1)',fontweight='bold')
plt.grid()
plt.xlabel('Month',fontweight='bold')
plt.ylabel('mm/day',fontweight='bold')
plt.xlim([1,12])
plt.xticks(np.arange(1,13), ['J','F','M','A','M','J','J','A','S','O','N','D'],fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,6])
plt.legend()
ax3 = fig.add_subplot(1,3,2)
(cesm_ssp245_ens.PRECT*86400*1000).sel(lat=slice(10,20)).where((cesm_ssp245_ens.lon<=10)|\
   (cesm_ssp245_ens.lon>=340)).weighted(weights_cesm).mean(('lat','lon'))\
   .sel(time=slice('2020','2039')).groupby('time.month').mean('time').plot(c='k',lw=3,label='SSP2-4.5 2020-39')
(cesm_ssp245_ens.PRECT*86400*1000).sel(lat=slice(10,20)).where((cesm_ssp245_ens.lon<=10)|\
   (cesm_ssp245_ens.lon>=340)).weighted(weights_cesm).mean(('lat','lon'))\
   .sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='r',lw=3,label='SSP2-4.5 2055-74')
(((cesm_mcb_data[cesm_mcb_exp[0]].PRECT+cesm_mcb_data[cesm_mcb_exp[1]].PRECT+cesm_mcb_data[cesm_mcb_exp[2]].PRECT)/3)*86400*1000)\
   .sel(lat=slice(10,20)).where((cesm_ssp245_ens.lon<=10)|(cesm_ssp245_ens.lon>=340))\
   .weighted(weights_cesm).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='b',lw=3,label='AMCB 2055-74')
(cesm_mcb_data[cesm_mcb_exp[0]].PRECT*86400*1000)\
   .sel(lat=slice(10,20)).where((cesm_ssp245_ens.lon<=10)|(cesm_ssp245_ens.lon>=340))\
   .weighted(weights_cesm).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='b',lw=1)
(cesm_mcb_data[cesm_mcb_exp[1]].PRECT*86400*1000)\
   .sel(lat=slice(10,20)).where((cesm_ssp245_ens.lon<=10)|(cesm_ssp245_ens.lon>=340))\
   .weighted(weights_cesm).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='b',lw=1)
(cesm_mcb_data[cesm_mcb_exp[2]].PRECT*86400*1000)\
   .sel(lat=slice(10,20)).where((cesm_ssp245_ens.lon<=10)|(cesm_ssp245_ens.lon>=340))\
   .weighted(weights_cesm).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='b',lw=1)
plt.title('(b) Sahel Precipitation (CESM2)',fontweight='bold')
plt.grid()
plt.xlabel('Month',fontweight='bold')
plt.ylabel('mm/day',fontweight='bold')
plt.xlim([1,12])
plt.xticks(np.arange(1,13), ['J','F','M','A','M','J','J','A','S','O','N','D'],fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,6])
plt.legend()
ax3 = fig.add_subplot(1,3,3)
(((e3sm_PRECT_ctl_1+e3sm_PRECT_ctl_2+e3sm_PRECT_ctl_3)/3)*86400*1000).sel(lat=slice(10,20)).where((e3sm_PRECT_amcb_1.lon<=10)|\
   (e3sm_PRECT_amcb_1.lon>=340)).weighted(weights_e3sm).mean(('lat','lon'))\
   .sel(time=slice('2020','2039')).groupby('time.month').mean('time').plot(c='k',lw=3,label='SSP2-4.5 2020-39')
(((e3sm_PRECT_ctl_1+e3sm_PRECT_ctl_2+e3sm_PRECT_ctl_3)/3)*86400*1000).sel(lat=slice(10,20)).where((e3sm_PRECT_amcb_1.lon<=10)|\
   (e3sm_PRECT_amcb_1.lon>=340)).weighted(weights_e3sm).mean(('lat','lon'))\
   .sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='r',lw=3,label='SSP2-4.5 2055-74')
(((e3sm_PRECT_amcb_1+e3sm_PRECT_amcb_2+e3sm_PRECT_amcb_3)/3)*86400*1000)\
   .sel(lat=slice(10,20)).where((e3sm_PRECT_amcb_1.lon<=10)|(e3sm_PRECT_amcb_1.lon>=340))\
   .weighted(weights_e3sm).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='b',lw=3,label='AMCB 2055-74')
(e3sm_PRECT_amcb_1*86400*1000)\
   .sel(lat=slice(10,20)).where((e3sm_PRECT_amcb_1.lon<=10)|(e3sm_PRECT_amcb_1.lon>=340))\
   .weighted(weights_e3sm).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='b',lw=1)
(e3sm_PRECT_amcb_2*86400*1000)\
   .sel(lat=slice(10,20)).where((e3sm_PRECT_amcb_1.lon<=10)|(e3sm_PRECT_amcb_1.lon>=340))\
   .weighted(weights_e3sm).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='b',lw=1)
(e3sm_PRECT_amcb_3*86400*1000)\
   .sel(lat=slice(10,20)).where((e3sm_PRECT_amcb_1.lon<=10)|(e3sm_PRECT_amcb_1.lon>=340))\
   .weighted(weights_e3sm).mean(('lat','lon')).sel(time=slice('2055','2074')).groupby('time.month').mean('time').plot(c='b',lw=1)
plt.title('(c) Sahel Precipitation (E3SM)',fontweight='bold')
plt.grid()
plt.xlabel('Month',fontweight='bold')
plt.ylabel('mm/day',fontweight='bold')
plt.xlim([1,12])
plt.xticks(np.arange(1,13), ['J','F','M','A','M','J','J','A','S','O','N','D'],fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim([0,6])
plt.legend()
plt.tight_layout()
plt.savefig('./figs/fig_amcb_A6.pdf',bbox_inches='tight')
plt.show()


# In[44]:


fig = plt.figure(figsize=(16,4),dpi=400)
ax3 = fig.add_subplot(1,3,1)
plt.axhline(y=0,c='k',alpha=0.5)
((ukesm_ssp_pr_1.sel(time=slice("2055","2074")).mean(('time','lon'))*86400)-\
    ((ukesm_ssp_pr_1*86400).sel(time=slice("2014","2033")).mean(('time','lon')))).plot(c='r',lw=1)
((ukesm_ssp_pr_2.sel(time=slice("2055","2074")).mean(('time','lon'))*86400)-\
    ((ukesm_ssp_pr_2*86400).sel(time=slice("2014","2033")).mean(('time','lon')))).plot(c='r',lw=1)
((ukesm_ssp_pr_3.sel(time=slice("2055","2074")).mean(('time','lon'))*86400)-\
    ((ukesm_ssp_pr_3*86400).sel(time=slice("2014","2033")).mean(('time','lon')))).plot(c='r',lw=1)

((ukesm_amcb_pr_1.sel(time=slice("2055","2074")).mean(('time','lon'))*86400)-\
    ((ukesm_ssp_pr_1*86400).sel(time=slice("2014","2033")).mean(('time','lon')))).plot(c='b',lw=1)
((ukesm_amcb_pr_2.sel(time=slice("2055","2074")).mean(('time','lon'))*86400)-\
    ((ukesm_ssp_pr_2*86400).sel(time=slice("2014","2033")).mean(('time','lon')))).plot(c='b',lw=1)
((ukesm_amcb_pr_3.sel(time=slice("2055","2074")).mean(('time','lon'))*86400)-\
    ((ukesm_ssp_pr_3*86400).sel(time=slice("2014","2033")).mean(('time','lon')))).plot(c='b',lw=1)

((ukesm_amcb_pr_ave.sel(time=slice("2055","2074")).mean(('time','lon'))*86400)-\
 (ukesm_ssp_pr_ave*86400).sel(time=slice("2014","2033")).mean(('time','lon')))\
   .plot(c='b',lw=3,label='AMCB')

((ukesm_ssp_pr_ave*86400).sel(time=slice("2055","2074")).mean(('time','lon'))-\
 (ukesm_ssp_pr_ave*86400).sel(time=slice("2014","2033")).mean(('time','lon')))\
   .plot(c='r',lw=3,label='SSP2-4.5')
plt.grid()
plt.ylabel('mm/day',fontweight='bold')
plt.xlabel('Latitude (deg N)',fontweight='bold')
plt.xlim([-90,90])
plt.ylim([-0.4,0.9])
plt.xticks(np.linspace(-90,90,7),fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('(a) UKESM1 Precipitation (annual-mean)',fontweight='bold')
plt.legend()

ax3 = fig.add_subplot(1,3,2)
plt.axhline(y=0,c='k',alpha=0.5)
((cesm_mcb_data[cesm_mcb_exp[0]].PRECT*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
    (cesm_prec_ssp_1.PRECT*86400*1000).sel(time=slice("2020","2039")).mean(('lon','time'))).plot(c='b',lw=1)
((cesm_mcb_data[cesm_mcb_exp[1]].PRECT*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
    (cesm_prec_ssp_2.PRECT*86400*1000).sel(time=slice("2020","2039")).mean(('lon','time'))).plot(c='b',lw=1)
((cesm_mcb_data[cesm_mcb_exp[2]].PRECT*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
    (cesm_prec_ssp_3.PRECT*86400*1000).sel(time=slice("2020","2039")).mean(('lon','time'))).plot(c='b',lw=1)

((cesm_prec_ssp_1.PRECT*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
    (cesm_prec_ssp_1.PRECT*86400*1000).sel(time=slice("2020","2039")).mean(('lon','time'))).plot(c='r',lw=1)
((cesm_prec_ssp_2.PRECT*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
    (cesm_prec_ssp_2.PRECT*86400*1000).sel(time=slice("2020","2039")).mean(('lon','time'))).plot(c='r',lw=1)
((cesm_prec_ssp_3.PRECT*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
    (cesm_prec_ssp_3.PRECT*86400*1000).sel(time=slice("2020","2039")).mean(('lon','time'))).plot(c='r',lw=1)

((((cesm_mcb_data[cesm_mcb_exp[0]].PRECT+cesm_mcb_data[cesm_mcb_exp[1]].PRECT+cesm_mcb_data[cesm_mcb_exp[2]].PRECT)/3)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
 (cesm_ssp245_ens.PRECT*86400*1000).sel(time=slice("2020","2039")).mean(('lon','time'))).plot(c='b',lw=3,label='AMCB')
((cesm_ssp245_ens.PRECT*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
 (cesm_ssp245_ens.PRECT*86400*1000).sel(time=slice("2020","2039")).mean(('lon','time'))).plot(c='r',lw=3,label='SSP2-4.5')
plt.grid()
plt.ylabel('mm/day',fontweight='bold')
plt.xlabel('Latitude (deg N)',fontweight='bold')
plt.xlim([-90,90])
plt.ylim([-0.4,0.9])
plt.xticks(np.linspace(-90,90,7),fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('(b) CESM2 Precipitation (annual-mean)',fontweight='bold')
plt.legend()

ax3 = fig.add_subplot(1,3,3)
(((e3sm_PRECT_amcb_1)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
     ((e3sm_PRECT_ctl_1)*86400*1000).sel(time=slice("2029","2048")).mean(('lon','time'))).plot(c='b',lw=1)
(((e3sm_PRECT_amcb_2)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
     ((e3sm_PRECT_ctl_2)*86400*1000).sel(time=slice("2029","2048")).mean(('lon','time'))).plot(c='b',lw=1)
(((e3sm_PRECT_amcb_3)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
     ((e3sm_PRECT_ctl_3)*86400*1000).sel(time=slice("2029","2048")).mean(('lon','time'))).plot(c='b',lw=1)
(((e3sm_PRECT_ctl_1)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
     ((e3sm_PRECT_ctl_1)*86400*1000).sel(time=slice("2029","2048")).mean(('lon','time'))).plot(c='r',lw=1)
(((e3sm_PRECT_ctl_2)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
     ((e3sm_PRECT_ctl_2)*86400*1000).sel(time=slice("2029","2048")).mean(('lon','time'))).plot(c='r',lw=1)
(((e3sm_PRECT_ctl_3)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
     ((e3sm_PRECT_ctl_3)*86400*1000).sel(time=slice("2029","2048")).mean(('lon','time'))).plot(c='r',lw=1)
plt.axhline(y=0,c='k',alpha=0.5)
((((e3sm_PRECT_amcb_1+e3sm_PRECT_amcb_2+e3sm_PRECT_amcb_3)/3)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
 (((e3sm_PRECT_ctl_1+e3sm_PRECT_ctl_2+e3sm_PRECT_ctl_3)/3)*86400*1000).sel(time=slice("2029","2048")).mean(('lon','time'))).plot(c='b',lw=3,label='AMCB')
((((e3sm_PRECT_ctl_1+e3sm_PRECT_ctl_2+e3sm_PRECT_ctl_3)/3)*86400*1000).sel(time=slice("2055","2074")).mean(('lon','time'))-\
 (((e3sm_PRECT_ctl_1+e3sm_PRECT_ctl_2+e3sm_PRECT_ctl_3)/3)*86400*1000).sel(time=slice("2029","2048")).mean(('lon','time'))).plot(c='r',lw=3,label='SSP2-4.5')
plt.grid()
plt.ylabel('mm/day',fontweight='bold')
plt.xlabel('Latitude (deg N)',fontweight='bold')
plt.xlim([-90,90])
plt.ylim([-0.4,0.9])
plt.xticks(np.linspace(-90,90,7),fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('(c) E3SM Precipitation (annual-mean)',fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('./figs/fig_amcb_A5.pdf',bbox_inches='tight')
plt.show()


# ### Get AMOC data

# In[45]:


# This is the processing done for the UKESM AMOC index
# # y value closest to 26.5 = 228
# # AMOC index as the vertical maximum of the annual mean meridional streamfunction in the Atlantic at 26.5N
# # ukesm_amoc_index_amcb_1 = np.zeros(40)
# # for year in np.arange(2035,2075):
# #     data = xr.open_mfdataset('/data/users/mhenry/Arctic_MCB/u-da694/amoc_proc/nemo_da694o_1m_'+str(year)+'*.nc',
# #                               concat_dim='time_counter',combine='nested')
# #     ukesm_amoc_index_amcb_1[year-2035]=data.zomsfatl.isel(y=228).mean('time_counter').max('depthw').values


# In[46]:


# ukesm_amoc_ssp_1 = np.insert(ukesm_amoc_ssp_1,0,
#                    (xr.open_mfdataset('/data/users/matthew.henry/historical/u-bc179/amoc/*.nc').msftyz*1e-9).isel(basin=0)\
#                    .sel(time='2014').mean('time').isel(rlat=228).max('lev').values)
# ukesm_amoc_ssp_2 = np.insert(ukesm_amoc_ssp_2,0,
#                    (xr.open_mfdataset('/data/users/matthew.henry/historical/u-bc292/amoc/*.nc').msftyz*1e-9).isel(basin=0)\
#                    .sel(time='2014').mean('time').isel(rlat=228).max('lev').values)
# ukesm_amoc_ssp_3 = np.insert(ukesm_amoc_ssp_3,0,
#                    (xr.open_mfdataset('/data/users/matthew.henry/historical/u-bc370/amoc/*.nc').msftyz*1e-9).isel(basin=0)\
#                    .sel(time='2014').mean('time').isel(rlat=228).max('lev').values)

# np.save(data_dir+'ukesm/ukesm_amcb_u-be537_amoc.npy',ukesm_amoc_ssp_1)
# np.save(data_dir+'ukesm/ukesm_amcb_u-be606_amoc.npy',ukesm_amoc_ssp_2)
# np.save(data_dir+'ukesm/ukesm_amcb_u-be683_amoc.npy',ukesm_amoc_ssp_3)


# In[47]:


ukesm_amoc_ssp_1 = np.load(data_dir+'ukesm/ukesm_amcb_u-be537_amoc.npy')
ukesm_amoc_ssp_2 = np.load(data_dir+'ukesm/ukesm_amcb_u-be606_amoc.npy')
ukesm_amoc_ssp_3 = np.load(data_dir+'ukesm/ukesm_amcb_u-be683_amoc.npy')
ukesm_amoc_amcb_1 = np.load(data_dir+'ukesm/ukesm_amcb_u-da694_amoc.npy')
ukesm_amoc_amcb_2 = np.load(data_dir+'ukesm/ukesm_amcb_u-da695_amoc.npy')
ukesm_amoc_amcb_3 = np.load(data_dir+'ukesm/ukesm_amcb_u-da696_amoc.npy')


# In[48]:


cesm_amoc_ssp_1 = xr.open_mfdataset(data_dir+'cesm/amoc_ssp/*.001.*.nc')
cesm_amoc_ssp_2 = xr.open_mfdataset(data_dir+'cesm/amoc_ssp/*.002.*.nc')
cesm_amoc_ssp_3 = xr.open_mfdataset(data_dir+'cesm/amoc_ssp/*.003.*.nc')
cesm_amoc_amcb_1 = xr.open_mfdataset(data_dir+'cesm/'+
                       '*LE2-1011.001.pop.*.zarr', engine = 'zarr')
cesm_amoc_amcb_2 = xr.open_mfdataset(data_dir+'cesm/'+
                       '*LE2-1031.002.pop.*.zarr', engine = 'zarr')
cesm_amoc_amcb_3 = xr.open_mfdataset(data_dir+'cesm/'+
                       '*LE2-1051.003.pop.*.zarr', engine = 'zarr')


# In[49]:


e3sm_amoc_ctl_1 = xr.open_mfdataset(data_dir+'e3sm/'+
                       '20221014.v2.LR.WCYCLSSP245.E2_CNTL_01.m*.zarr', engine = 'zarr')
e3sm_amoc_ctl_2 = xr.open_mfdataset(data_dir+'e3sm/'+
                       '20221018.v2.LR.WCYCLSSP245.E2_CNTL_02.m*.zarr', engine = 'zarr')
e3sm_amoc_ctl_3 = xr.open_mfdataset(data_dir+'e3sm/'+
                       '20230277.v2.LR.WCYCLSSP245.E2_CNTL_03.m*.zarr', engine = 'zarr')
e3sm_amoc_amcb_1 = xr.open_mfdataset(data_dir+'e3sm/'+
                       '20240514.v2.LR.WCYCLSSP245.E2_CNTL_01.ARCTIC_CONTROL.*.zarr', engine = 'zarr')
e3sm_amoc_amcb_2 = xr.open_mfdataset(data_dir+'e3sm/'+
                       '20240514.v2.LR.WCYCLSSP245.E2_CNTL_02.ARCTIC_CONTROL.*.zarr', engine = 'zarr')
e3sm_amoc_amcb_3 = xr.open_mfdataset(data_dir+'e3sm/'+
                       '20240514.v2.LR.WCYCLSSP245.E2_CNTL_03.ARCTIC_CONTROL.*.zarr', engine = 'zarr')


# In[50]:


print('Target AMOC values')
print('UKESM1 average over 2014-33 : ',(ukesm_amoc_ssp_1[0:20].mean()+ukesm_amoc_ssp_2[0:20].mean()+ukesm_amoc_ssp_3[0:20].mean())/3,\
      ' +/- ',np.std([ukesm_amoc_ssp_1[0:20].mean(),ukesm_amoc_ssp_2[0:20].mean(),ukesm_amoc_ssp_3[0:20].mean()]))
print('CESM2 average over 2020-39 : ',((cesm_amoc_ssp_1['MOC']+cesm_amoc_ssp_2['MOC']+cesm_amoc_ssp_3['MOC'])/3)\
    .sel(year=slice(2020,2039)).mean().values,' +/- ',np.std([cesm_amoc_ssp_1['MOC'].sel(year=slice(2020,2039)).mean().values,\
                                                              cesm_amoc_ssp_2['MOC'].sel(year=slice(2020,2039)).mean().values,\
                                                              cesm_amoc_ssp_3['MOC'].sel(year=slice(2020,2039)).mean().values]))
print('E3SMV2 average over 2029-48 : ',((e3sm_amoc_ctl_1['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112]+\
                                         e3sm_amoc_ctl_2['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112]+\
                                         e3sm_amoc_ctl_3['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112])/3)\
    .groupby('Time.year').mean().max("nVertLevels").sel(year=slice(2029,2048)).mean().values,\
      ' +/- ',np.std([e3sm_amoc_ctl_1['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year')\
          .mean().max("nVertLevels").sel(year=slice(2029,2048)).mean().values,\
                      e3sm_amoc_ctl_2['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year')\
          .mean().max("nVertLevels").sel(year=slice(2029,2048)).mean().values,\
                      e3sm_amoc_ctl_3['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year')\
          .mean().max("nVertLevels").sel(year=slice(2029,2048)).mean().values]))


# In[51]:


fig = plt.figure(figsize=(18,4),dpi=200)

plt.subplot(131)
plt.fill_between(np.arange(2015,2101),15.60-0.34,15.60+0.34,alpha=0.2,facecolor='black')
plt.axhline(y=15.60,lw=3,c='k',label='Target')
plt.plot(np.arange(2014,2101),ukesm_amoc_ssp_1,c='r',lw=1,alpha=0.5)
plt.plot(np.arange(2014,2101),ukesm_amoc_ssp_2,c='r',lw=1,alpha=0.5)
plt.plot(np.arange(2014,2101),ukesm_amoc_ssp_3,c='r',lw=1,alpha=0.5)
plt.plot(np.arange(2014,2101),(ukesm_amoc_ssp_1+ukesm_amoc_ssp_2\
         +ukesm_amoc_ssp_3)/3,c='r',lw=3,label='SSP2-4.5')
plt.plot(np.arange(2035,2075),ukesm_amoc_amcb_1,c='b',lw=1,alpha=0.5)
plt.plot(np.arange(2035,2075),ukesm_amoc_amcb_2,c='b',lw=1,alpha=0.5)
plt.plot(np.arange(2035,2075),ukesm_amoc_amcb_3,c='b',lw=1,alpha=0.5)
plt.plot(np.arange(2035,2075),(ukesm_amoc_amcb_1+ukesm_amoc_amcb_2\
         +ukesm_amoc_amcb_3)/3,c='b',lw=3,label='Arctic MCB')
plt.title('(a) AMOC (UKESM1)',fontweight='bold')
plt.ylabel('Sv (10$^6$m$^3$/s)',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlim([2015,2080])
plt.ylim([7,20])
plt.grid()
# plt.legend()

plt.subplot(132)
plt.fill_between(np.arange(2015,2101),16.2-0.24,16.2+0.24,alpha=0.2,facecolor='black')
plt.axhline(y=16.2,lw=3,c='k',label='Target')
cesm_amoc_ssp_1['MOC'].plot(c='r',lw=1,alpha=0.5)
cesm_amoc_ssp_2['MOC'].plot(c='r',lw=1,alpha=0.5)
cesm_amoc_ssp_3['MOC'].plot(c='r',lw=1,alpha=0.5)
((cesm_amoc_ssp_1['MOC']+cesm_amoc_ssp_2['MOC']+cesm_amoc_ssp_3['MOC'])/3).plot(c='r',lw=3,label='SSP2-4.5')
cesm_amoc_amcb_1['MOC'].sel(time=slice('2035','2074'))[:,1,0,:,274].groupby('time.year').mean().max("moc_z").plot(c='b',lw=1,alpha=0.5)
cesm_amoc_amcb_2['MOC'].sel(time=slice('2035','2074'))[:,1,0,:,274].groupby('time.year').mean().max("moc_z").plot(c='b',lw=1,alpha=0.5)
cesm_amoc_amcb_3['MOC'].sel(time=slice('2035','2074'))[:,1,0,:,274].groupby('time.year').mean().max("moc_z").plot(c='b',lw=1,alpha=0.5)
((cesm_amoc_amcb_1['MOC'].sel(time=slice('2035','2074'))[:,1,0,:,274].groupby('time.year').mean().max("moc_z")+
 cesm_amoc_amcb_2['MOC'].sel(time=slice('2035','2074'))[:,1,0,:,274].groupby('time.year').mean().max("moc_z")+
 cesm_amoc_amcb_3['MOC'].sel(time=slice('2035','2074'))[:,1,0,:,274].groupby('time.year').mean().max("moc_z"))/3).plot(c='b',lw=3,label='Arctic MCB')
plt.title('(b) AMOC (CESM2)',fontweight='bold')
plt.ylabel('Sv (10$^6$m$^3$/s)',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xlim([2015,2080])
plt.ylim([7,20])
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
# plt.legend()

plt.subplot(133)
plt.fill_between(np.arange(2015,2101),10.2-0.23,10.2+0.23,alpha=0.2,facecolor='black')
plt.axhline(y=10.2,lw=3,c='k',label='Target')
e3sm_amoc_ctl_1['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels").plot(c='r',alpha=0.5,lw=1)
e3sm_amoc_ctl_2['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels").plot(c='r',alpha=0.5,lw=1)
e3sm_amoc_ctl_3['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels").plot(c='r',alpha=0.5,lw=1)
((e3sm_amoc_ctl_1['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels")+
  e3sm_amoc_ctl_2['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels")+
  e3sm_amoc_ctl_3['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels"))/3).plot(c='r',lw=3,label='SSP2-4.5')
e3sm_amoc_amcb_1['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels").plot(c='b',alpha=0.5,lw=1)
e3sm_amoc_amcb_2['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels").plot(c='b',alpha=0.5,lw=1)
e3sm_amoc_amcb_3['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels").plot(c='b',alpha=0.5,lw=1)
((e3sm_amoc_amcb_1['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels")+
  e3sm_amoc_amcb_2['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels")+
  e3sm_amoc_amcb_3['timeMonthly_avg_mocStreamvalLatAndDepthRegion'][:,0,:,112].groupby('Time.year').mean().max("nVertLevels"))/3).plot(c='b',lw=3,label='Arctic MCB')
plt.title('(c) AMOC (E3SM)',fontweight='bold')
plt.ylabel('Sv (10$^6$m$^3$/s)',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xlim([2015,2080])
plt.ylim([7,20])
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
plt.legend()
plt.savefig('./figs/fig_amcb_5.pdf',bbox_inches='tight')
plt.show()


# Global temperature change

# In[52]:


fig = plt.figure(figsize=(18,4),dpi=200)
plt.subplot(131)
plt.axhline(y=0, color='k', linestyle='--',lw=2)
(tas_ukesm_ctl-288.06).weighted(weights_ukesm_ssp).mean(('lat','lon')).groupby('time.year')\
  .mean('time').plot(c='r',lw=3,label='SSP2-4.5')
(tas_ukesm_amcb-288.06).weighted(weights_ukesm_amcb)\
  .mean(('latitude','longitude')).groupby('time.year').mean('time').plot(c='b',lw=3,label='Arctic MCB')
for sim in [tas_ukesm_ctl_1,tas_ukesm_ctl_2,tas_ukesm_ctl_3]:
    (sim-288.06).weighted(weights_ukesm_ssp).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='r',lw=1,alpha=0.5)
for sim in [tas_ukesm_amcb_1,tas_ukesm_amcb_2,tas_ukesm_amcb_3]:
    (sim-288.06).weighted(weights_ukesm_amcb).mean(('latitude','longitude')).groupby('time.year').mean('time').plot(c='b',lw=1,alpha=0.5)
plt.title('(a) UKESM1 Global-mean temperature',fontweight='bold')
plt.ylim([-0.5,3])
plt.ylabel('K',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
plt.xlim([2020,2075])
plt.legend()
plt.subplot(132)
plt.axhline(y=0, color='k', linestyle='--',lw=2)
(tas_cesm_ctl-288.81).weighted(cesm_ssp245_ens.gw).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='r',lw=3,label='SSP2-4.5')
(tas_cesm_amcb-288.81).weighted(cesm_ssp245_ens.gw).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='b',lw=3,label='Arctic MCB')
for sim in [tas_cesm_ctl_1,tas_cesm_ctl_2,tas_cesm_ctl_3]:
    (sim-288.81).weighted(weights_cesm).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='r',lw=1,alpha=0.5)
for sim in [tas_cesm_amcb_1,tas_cesm_amcb_2,tas_cesm_amcb_3]:
    (sim-288.81).weighted(cesm_ssp245_ens.gw).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='b',lw=1,alpha=0.5)
plt.title('(b) CESM2 Global-mean temperature',fontweight='bold')
plt.ylim([-0.5,3])
plt.ylabel('K',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
plt.xlim([2020,2075])
plt.legend()
plt.tight_layout()
plt.subplot(133)
plt.axhline(y=0, color='k', linestyle='--',lw=2)
(tas_e3sm_ctl-288.01).weighted(weights_e3sm).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='r',lw=3,label='SSP2-4.5')
(tas_e3sm_amcb-288.01).weighted(weights_e3sm).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='b',lw=3,label='Arctic MCB')
for sim in [tas_e3sm_ctl_1,tas_e3sm_ctl_2,tas_e3sm_ctl_3]:
    (sim-288.01).weighted(weights_e3sm).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='r',lw=1,alpha=0.5)
for sim in [tas_e3sm_amcb_1,tas_e3sm_amcb_2,tas_e3sm_amcb_3]:
    (sim-288.01).weighted(weights_e3sm).mean(('lat','lon')).groupby('time.year').mean('time').plot(c='b',lw=1,alpha=0.5)
plt.title('(c) E3SM Global-mean temperature',fontweight='bold')
plt.ylim([-0.5,3])
plt.ylabel('K',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
plt.xlim([2020,2075])
plt.legend()
plt.savefig('./figs/fig_amcb_A1.pdf',bbox_inches='tight')
plt.show()


# In[53]:


fig = plt.figure(figsize=(18,4),dpi=300)
ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.NorthPolarStereo())
p = (tas_ukesm_amcb.sel(time=slice('2055','2074')).mean('time').rename({'latitude':'lat','longitude':'lon'})- \
     tas_ukesm_ctl.sel(time=slice('2014','2033')).mean('time')).plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
ax1.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax1.set_boundary(circle, transform=ax1.transAxes)
sim = tas_ukesm_ctl
sim.mean('time').where(sim.lat>=60).where(sim.lat<=80)\
            .plot.contourf(levels=[0,1], colors='None',hatches=['..'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
ax1.set_title("(a) UKESM1 $\Delta$T$_S$ (2055-74) - (2014-33)",fontweight='bold')
ax1.coastlines()
ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.NorthPolarStereo())
p = (tas_cesm_amcb.sel(time=slice('2055','2074')).mean('time')- \
     tas_cesm_ctl.sel(time=slice('2020','2039')).mean('time')).plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
ax2.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax2.set_boundary(circle, transform=ax2.transAxes)
ax2.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
sim = tas_cesm_ctl
sim.mean('time').where(sim.lat>=60).where(sim.lat<=80)\
            .plot.contourf(levels=[0,1], colors='None',hatches=['..'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
ax2.set_title("(b) CESM2 $\Delta$T$_S$ (2055-74) - (2020-39)",fontweight='bold')
ax2.coastlines()
ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.NorthPolarStereo())
p = (tas_e3sm_amcb.sel(time=slice('2055','2074')).mean('time')-tas_e3sm_ctl.sel(time=slice('2029','2048')).mean('time')).plot.contourf(
      levels=np.linspace(-9,9,19),cmap='bwr',
      subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
      transform=ccrs.PlateCarree(),
      extend = "both",
      cbar_kwargs={"shrink":0.5, "label":"K"})
ax3.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
ax3.set_boundary(circle, transform=ax3.transAxes)
ax3.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black',alpha=0.3)
sim = tas_e3sm_ctl
sim.mean('time').where(sim.lat>=60).where(sim.lat<=80)\
            .plot.contourf(levels=[0,1], colors='None',hatches=['..'],add_colorbar=False,\
            subplot_kws=dict(projection=ccrs.NorthPolarStereo(), facecolor="gray"),
            transform=ccrs.PlateCarree())
ax3.set_title("(c) E3SM $\Delta$T$_S$ (2055-74) - (2029-48)",fontweight='bold')
ax3.coastlines()
plt.tight_layout()
plt.savefig('./figs/fig_amcb_A2.pdf',bbox_inches='tight')
plt.show()


# In[54]:


fig = plt.figure(figsize=(6,4),dpi=200)
(tas_ukesm_amcb-tas_ukesm_ctl.rename({'lat':'latitude','lon':'longitude'}))\
  .sel(latitude=slice(70,90)).weighted(weights_ukesm_amcb).mean(('latitude','longitude')).sel(time=slice('2055','2074'))\
  .groupby('time.month').mean('time').plot(c='k',lw=3,label='UKESM1')
(tas_cesm_amcb-tas_cesm_ctl).sel(lat=slice(70,90)).sel(time=slice('2055','2074'))\
  .weighted(cesm_ssp245_ens.gw).mean(('lat','lon')).groupby('time.month').mean('time').plot(c='b',lw=3,label='CESM2')
(tas_e3sm_amcb-tas_e3sm_ctl).sel(lat=slice(70,90)).weighted(weights_e3sm).mean(('lat','lon')).sel(time=slice('2055','2074'))\
  .groupby('time.month').mean('time').plot(c='r',lw=3,label='E3SMv2')
plt.title('Arctic $\Delta$T$_S$ (2055-74, 70N-90N)',fontweight='bold')
plt.ylabel('K',fontweight='bold')
plt.xlabel('Year',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid()
plt.xlim([1,12])
plt.xticks(np.arange(1,13), ['J','F','M','A','M','J','J','A','S','O','N','D'],fontweight='bold')
plt.ylim([-10,0])
plt.legend()
plt.savefig('./figs/fig_amcb_A3.pdf',bbox_inches='tight')
plt.show()


# Useful functions

# In[55]:


ne30area = './ne30pg2.nc'
DSA = xr.open_mfdataset(ne30area)
lon_cs = DSA.grid_center_lon
lat_cs = DSA.grid_center_lat
area_cs = DSA.grid_area

# map data to lat/lon grid

dinc = 1.  # increment of mesh in degrees
lon_ll=np.arange(0.,361, dinc)
lat_ll=np.arange(-90.,90.,dinc)
xoutm,youtm=np.meshgrid(lon_ll,lat_ll)


def interp_ap(xt, yt, data2d,lat,lon,method=None):
    """
    # interp an arbitrary set of points at xt, yt 
    # from data on an unstructured mesh at lat, lon
    #
    # interpolating in lat/lon space has issues with triangulation 
    # at pole and wrapping at greenwich, so interpolate in stereographic projection:
    #
    # input:
    #    data2d(ncol,...),lat(ncol),lon(ncol): data and coords on unstructured mesh
    #    data2d can be multidimensional array, but ncol must be first coordinate
    #    xt, yt: lat and lon coordinates of locations to interpolate to
    #    method: optional, use cubic interpolation if method='cubic'
    #
    # output 
    #    returns an array with same shape as xt with interpolated data
    #
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.interpolate import CloughTocher2DInterpolator

    intp2D = LinearNDInterpolator
    if method == 'cubic':
        intp2D = CloughTocher2DInterpolator

    ld = data2d.shape[0] # length of first coord of input data
    lx = lon.shape[0]
    ly = lat.shape[0]
    if ((ld != lx) | (ld != ly)):
        print('inconsistent data2d, lon, lat arrays', ld, lx, ly)
        raise TypeError("inconsistent input in interp_ap")

    # mesh grid
    dproj=ccrs.PlateCarree()

    # select interpolation points located in the nh and sh
    inds = np.where(yt <= 0)
    indn = np.where(yt > 0)

    xtn = xt[indn]
    ytn = yt[indn]
    xts = xt[inds]
    yts = yt[inds]

    # take source data in the correct hemisphere, include extra halo points for interpolation
    # using the full global data sometimes confuses interpolation with points being mapped close to infinity
    halo = 15 # degrees
    data2d_h=data2d[lat<halo]

    lon_h=lon[lat<halo]
    lat_h=lat[lat<halo]
    coords_in  = ccrs.SouthPolarStereo().transform_points(dproj,lon_h,lat_h)

    dims = list(xt.shape)+list(data2d[0,...].shape)

    data_i = np.zeros(dims,dtype=data2d.dtype)
    data_i[:] = np.nan

    data_s = []
    if len(yts) > 0:
        cto = ccrs.SouthPolarStereo().transform_points(dproj,xts,yts)
        interp = intp2D(coords_in[:,0:2], data2d_h)
        data_s = interp(cto[:,0],cto[:,1])
        data_i[inds] = data_s

    data2d_h=data2d[lat>-halo]
    lon_h=lon[lat>-halo]
    lat_h=lat[lat>-halo]
    coords_in  = ccrs.NorthPolarStereo().transform_points(dproj,lon_h,lat_h)

    data_n = []
    if len(ytn) > 0:
        cto = ccrs.NorthPolarStereo().transform_points(dproj,xtn,ytn)
        interp = intp2D(coords_in[:,0:2], data2d_h)
        data_n = interp(cto[:,0],cto[:,1])
        data_i[indn] = data_n

    return data_i


def remap(data):
    return xr.DataArray(interp_ap(xoutm, youtm, data.values,lat_cs.values,lon_cs.values), 
                        coords={'lat': lat_ll,'lon': lon_ll, 'time' : data.time},
                        attrs=data.attrs,
                        dims=["lat", "lon", "time"])
def remap_3D(data):
    return xr.DataArray(interp_ap(xoutm, youtm, data.values,lat_cs.values,lon_cs.values), 
                        coords={'lat': lat_ll,'lon': lon_ll, 'time' : data.time,'lev': data.lev},
                        attrs=data.attrs,
                        dims=["lat", "lon", "time","lev"])

