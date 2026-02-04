import copernicusmarine

# Load Jason3 data
ds = copernicusmarine.read_dataframe(
    dataset_id="cmems_obs-wave_glo_phy-swh_nrt_j3-l3_PT1S",
    dataset_version="202211",
    variables=["VAVH"],
    minimum_longitude=-5.8,
    maximum_longitude=-1.13,
    minimum_latitude=47,
    maximum_latitude=49.858,
    start_datetime="2023-10-01T00:00:00",
    end_datetime="2023-12-31T23:00:00",
    minimum_depth=0,
    maximum_depth=0,
)

# Only keep the usefull columns
ds = ds[["variable", "platform_id", "time", "longitude", "latitude", "value"]]
ds.to_csv(
    "/scratch/shared/ww3/datas/obs/BRETAGNE0002/jason3/cmems_obs-wave_glo_phy-swh_nrt_j3-l3_PT1S_VAVH_5.80W-1.13W_47.00N-49.86N_0.00m_2023-12-19-2023-12-31.csv",
    index=False,
)
