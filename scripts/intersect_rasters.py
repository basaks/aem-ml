from pathlib import Path
import rasterio
import geopandas as gpd
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed

data_location = \
    Path("/g/data/ge3/covariates/national_albers_filled_new/albers_cropped/")
# Read points from shapefile

shapefile_location = Path("/g/data/ge3/aem_sections/AEM_covariates/")
# shapefile_location = Path("configs/data")

geotifs = {
    "relief_radius4.tif": "relief4",
    "national_Wii_RF_multirandomforest_prediction.tif": "mrf_pred",
    "MvrtpLL_smooth.tif": "mrvtpLL_s",
    "MvrtpLL_fin.tif": "mvrtpLL_f",
    "LOC_distance_to_coast.tif": "LOC_dis",
    "Gravity_land.tif": "gravity",
    "dem_fill.tif": "dem",
    "Clim_Prescott_LindaGregory.tif": "clim_linda",
    "clim_PTA_albers.tif": "clim_alber",
    "SagaWET9cell_M.tif": "sagawet",
    "ceno_euc_aust1.tif": "ceno_euc"
}


ratio = 10  # keep 1 in 10


def intersect_and_sample_shp(shp: Path):
    print("====================================\n", f"intersecting {shp.as_posix()}")
    pts = gpd.read_file(shp)
    coords = [(p.x, p.y) for p in pts.geometry]
    for k, v in geotifs.items():
        print(f"adding {k} to output dataframe")
        src = rasterio.open(data_location.joinpath(k))
        # Sample the raster at every point location and store values in DataFrame
        pts[v] = [x[0] for x in src.sample(coords)]
    pts.to_file(Path('out').joinpath(shp.name))
    pts.to_csv(Path("out").joinpath(shp.stem + ".csv"), index=False)


rets = Parallel(
    n_jobs=-1,
    verbose=100,
)(delayed(intersect_and_sample_shp)(s) for s in shapefile_location.glob("*.shp"))

