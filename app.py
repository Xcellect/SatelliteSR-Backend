from flask import Flask, request, jsonify
from flask_cors import CORS
from skyfield.api import EarthSatellite, Topos, load, utc
from datetime import datetime, timedelta
import requests
import geopandas as gp
import pandas as pd
import earthaccess
from osgeo import gdal
import logging
import rioxarray as rxr
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image
# Programmatically login to earthaccess using .netrc file
earthaccess.login(persist=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder='downloads')
CORS(app)

ts = load.timescale()

# Define a function to find overpasses
def find_overpasses(satellite, observer_location, start, end):
    t, events = satellite.find_events(observer_location, start, end, altitude_degrees=10.0)
    overpasses = []
    for ti, event in zip(t, events):
        if event == 0:  # Rise time
            overpasses.append(ti.utc_iso())
    return overpasses

@app.route('/overpass', methods=['POST'])
def get_overpass():
    logging.info('Processing overpass request')
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']

    logging.info('Fetching TLE data for Landsat 8 and Landsat 9')
    tle_landsat_8 = requests.get('https://celestrak.org/NORAD/elements/gp.php?CATNR=39084').text.splitlines()
    tle_landsat_9 = requests.get('https://celestrak.org/NORAD/elements/gp.php?CATNR=49260').text.splitlines()

    # Load TLE data for Landsat 8 and Landsat 9
    landsat_8 = EarthSatellite(tle_landsat_8[1], tle_landsat_8[2], 'LANDSAT 8', ts)
    landsat_9 = EarthSatellite(tle_landsat_9[1], tle_landsat_9[2], 'LANDSAT 9', ts)

    # Define observer location
    observer_location = Topos(latitude_degrees=latitude, longitude_degrees=longitude)

    # Define time window to search for overpasses (next 7 days)
    logging.info('Calculating overpasses for the next 7 days')
    current_time = datetime.utcnow().replace(tzinfo=utc)
    start_time = ts.utc(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute)
    end_time = ts.utc((current_time + timedelta(days=7)).year, (current_time + timedelta(days=7)).month, (current_time + timedelta(days=7)).day)

    # Search for overpasses within the time window
    logging.info('Finding overpasses for Landsat 8 and Landsat 9')
    landsat_8_overpasses = find_overpasses(landsat_8, observer_location, start_time, end_time)
    landsat_9_overpasses = find_overpasses(landsat_9, observer_location, start_time, end_time)

    logging.info('Overpass request processed successfully')
    return jsonify({
        'landsat_8_overpasses': landsat_8_overpasses,
        'landsat_9_overpasses': landsat_9_overpasses
    })

@app.route('/historical_overpass', methods=['POST'])
def get_historical_overpass():
    logging.info('Processing historical overpass request')
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    cloud_coverage_threshold = data.get('cloud_coverage_threshold', 15)

    logging.info('Fetching TLE data for Landsat 8 and Landsat 9')
    tle_landsat_8 = requests.get('https://celestrak.org/NORAD/elements/gp.php?CATNR=39084').text.splitlines()
    tle_landsat_9 = requests.get('https://celestrak.org/NORAD/elements/gp.php?CATNR=49260').text.splitlines()

    # Load TLE data for Landsat 8 and Landsat 9
    landsat_8 = EarthSatellite(tle_landsat_8[1], tle_landsat_8[2], 'LANDSAT 8', ts)
    landsat_9 = EarthSatellite(tle_landsat_9[1], tle_landsat_9[2], 'LANDSAT 9', ts)

    # Define observer location
    observer_location = Topos(latitude_degrees=latitude, longitude_degrees=longitude)

    # Define time window for historical overpasses
    if start_date and end_date:
        logging.info(f'Using provided date range: {start_date} to {end_date}')
        start_time = ts.utc(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=utc))
        end_time = ts.utc(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=utc))
    else:
        logging.info('No date range provided, using most recent 16 days')
        current_time = datetime.utcnow().replace(tzinfo=utc)
        start_time = ts.utc((current_time - timedelta(days=16)).year, (current_time - timedelta(days=16)).month, (current_time - timedelta(days=16)).day)
        end_time = ts.utc(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute)

    # Search for historical overpasses within the time window
    logging.info('Finding historical overpasses for Landsat 8 and Landsat 9')
    landsat_8_overpasses = find_overpasses(landsat_8, observer_location, start_time, end_time)
    landsat_9_overpasses = find_overpasses(landsat_9, observer_location, start_time, end_time)

    # Use earthaccess to find HLS data matching the location and time window
    logging.info('Searching for HLS data matching the location and time window')
    bbox = (longitude - 0.01, latitude - 0.01, longitude + 0.01, latitude + 0.01)
    temporal = (start_date, end_date) if start_date and end_date else None
    results = earthaccess.search_data(
        short_name=['HLSL30', 'HLSS30'],
        bounding_box=bbox,
        temporal=temporal,
        count=100
    )

    # Filter results by cloud coverage
    logging.info('Filtering results by cloud coverage')
    filtered_results = []

    for granule in results:
        cloud_coverage = None
        # Iterate over the AdditionalAttributes to find cloud coverage
        for attribute in granule['umm']['AdditionalAttributes']:
            if attribute['Name'] == 'CLOUD_COVERAGE':
                cloud_coverage = float(attribute['Values'][0])  # Assuming it's always a single value and can be converted to float
                break
        # If cloud coverage is found, apply the threshold filter
        if cloud_coverage is not None and cloud_coverage <= cloud_coverage_threshold:
            filtered_results.append(granule)

    metadata = pd.json_normalize([granule for granule in filtered_results])

    logging.info('Historical overpass request processed successfully')
    return jsonify({
        'landsat_8_overpasses': landsat_8_overpasses,
        'landsat_9_overpasses': landsat_9_overpasses,
        'metadata': metadata.to_dict(orient='records')
    })

def create_quality_mask(quality_data, bit_nums=[1, 2, 3, 4, 5]):
    """
    Creates a quality mask based on bit numbers to filter out poor quality pixels.
    """
    mask_array = np.zeros(quality_data.shape, dtype=np.int8)
    quality_data = np.nan_to_num(quality_data, 0).astype(np.int8)
    for bit in bit_nums:
        mask_temp = np.array(quality_data) & (1 << bit) > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array

ts = load.timescale()

# Helper function to create quality mask
def create_quality_mask(quality_data, bit_nums=[1, 2, 3, 4, 5]):
    """
    Creates a quality mask based on bit numbers to filter out poor quality pixels.
    """
    mask_array = np.zeros(quality_data.shape, dtype=np.int8)
    quality_data = np.nan_to_num(quality_data, 0).astype(np.int8)
    for bit in bit_nums:
        mask_temp = np.array(quality_data) & (1 << bit) > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array

@app.route('/evi', methods=['POST'])
def get_evi():
    logging.info('Processing EVI request')
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']

    # Use earthaccess to find HLS data matching the location
    logging.info('Searching for HLS data matching the location')
    bbox = (longitude - 0.01, latitude - 0.01, longitude + 0.01, latitude + 0.01)
    results = earthaccess.search_data(
        short_name=['HLSL30', 'HLSS30'],
        bounding_box=bbox,
        count=1
    )

    if not results:
        return jsonify({"error": "No granules found for the specified location"}), 404

    # Download the first available granule
    granule = results[0]
    logging.info(f'Downloading granule: {granule}')

    # Define the local path to save the files
    local_path = "./downloads"
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # Download the granule
    files = earthaccess.download([granule], local_path=local_path)

    if not files:
        return jsonify({"error": "Failed to download granule"}), 500

    # Load bands needed for EVI calculation using rioxarray
    evi_band_links = []
    for f in files:
        if any(band in f for band in ['B05', 'B04', 'B02', 'Fmask']):  # Using Landsat 8 codes
            evi_band_links.append(f)

    if len(evi_band_links) < 4:
        return jsonify({"error": "Missing required bands for EVI calculation"}), 500

    # Load the required bands for EVI calculation
    try:
        nir = rxr.open_rasterio([f for f in evi_band_links if 'B05' in f][0], masked=True).squeeze()
        red = rxr.open_rasterio([f for f in evi_band_links if 'B04' in f][0], masked=True).squeeze()
        blue = rxr.open_rasterio([f for f in evi_band_links if 'B02' in f][0], masked=True).squeeze()
        fmask = rxr.open_rasterio([f for f in evi_band_links if 'Fmask' in f][0], masked=True).squeeze()
    except Exception as e:
        logging.error(f"Failed to load required bands: {e}")
        return jsonify({"error": "Failed to load required bands"}), 500

    # Apply scaling and masking based on region
    logging.info('Applying scaling and masking based on region')
    bbox_geom = gp.GeoDataFrame(geometry=gp.points_from_xy([longitude], [latitude]), crs="EPSG:4326").to_crs(nir.rio.crs)
    nir_cropped = nir.rio.clip(bbox_geom.buffer(0.01).geometry, nir.rio.crs, all_touched=True)
    red_cropped = red.rio.clip(bbox_geom.buffer(0.01).geometry, red.rio.crs, all_touched=True)
    blue_cropped = blue.rio.clip(bbox_geom.buffer(0.01).geometry, blue.rio.crs, all_touched=True)
    fmask_cropped = fmask.rio.clip(bbox_geom.buffer(0.01).geometry, fmask.rio.crs, all_touched=True)

    # Quality Filter
    mask_layer = create_quality_mask(fmask_cropped.data, bit_nums=[1, 2, 3, 4, 5])
    evi_cropped = 2.5 * (nir_cropped - red_cropped) / (nir_cropped + 6 * red_cropped - 7.5 * blue_cropped + 1)
    evi_cropped_qf = evi_cropped.where(~mask_layer)

    # Generate GeoTIFF files for 3x3 grid around the point and concatenate them
    evi_geotiff_urls = []
    evi_values = []
    all_evi_data = []

    for lat_offset in [-0.01, 0, 0.01]:
        for lon_offset in [-0.01, 0, 0.01]:
            current_lat = latitude + lat_offset
            current_lon = longitude + lon_offset
            current_bbox_geom = gp.GeoDataFrame(geometry=gp.points_from_xy([current_lon], [current_lat]), crs="EPSG:4326").to_crs(nir.rio.crs)
            nir_cropped = nir.rio.clip(current_bbox_geom.buffer(0.01).geometry, nir.rio.crs, all_touched=True)
            red_cropped = red.rio.clip(current_bbox_geom.buffer(0.01).geometry, red.rio.crs, all_touched=True)
            blue_cropped = blue.rio.clip(current_bbox_geom.buffer(0.01).geometry, blue.rio.crs, all_touched=True)
            fmask_cropped = fmask.rio.clip(current_bbox_geom.buffer(0.01).geometry, fmask.rio.crs, all_touched=True)
            mask_layer = create_quality_mask(fmask_cropped.data, bit_nums=[1, 2, 3, 4, 5])
            evi_cropped = 2.5 * (nir_cropped - red_cropped) / (nir_cropped + 6 * red_cropped - 7.5 * blue_cropped + 1)
            evi_cropped_qf = evi_cropped.where(~mask_layer)

            # Check if the current EVI data has enough non-NaN pixels
            non_nan_count = np.sum(~np.isnan(evi_cropped_qf.data))
            total_count = evi_cropped_qf.data.size
            non_nan_ratio = non_nan_count / total_count

            # Only append the data if at least 50% of the pixels are non-NaN
            if non_nan_ratio > 0.5:
                output_filename = f"./downloads/evi_cropped_output_{current_lat}_{current_lon}.tif"
                try:
                    evi_cropped_qf.rio.to_raster(output_filename, driver='COG')
                    evi_geotiff_urls.append(output_filename)
                    evi_values.append(evi_cropped_qf.mean(dim=['x', 'y']).item())
                    all_evi_data.append(evi_cropped_qf)
                except Exception as e:
                    logging.error(f"Failed to save EVI output for ({current_lat}, {current_lon}): {e}")
                    evi_values.append(None)

    # Concatenate all EVI GeoTIFF files into a single raster
    logging.info("Concatenating all EVI GeoTIFF files into a single raster")
    try:
        if all_evi_data:
            evi_concat = xr.concat(all_evi_data, dim="y")
            composite_output_filename = "./downloads/evi_composite_output.tif"
            evi_concat.rio.to_raster(composite_output_filename, driver='COG')
        else:
            logging.error("No valid EVI data to concatenate")
            return jsonify({"error": "No valid EVI data to concatenate"}), 500
    except Exception as e:
        logging.error(f"Failed to concatenate EVI outputs: {e}")
        return jsonify({"error": "Failed to concatenate EVI outputs"}), 500

    try:
        # Generate PNG using imshow from matplotlib
        composite_output_filename_png = "./downloads/evi_composite_output.png"
        accessible_filename_png = "/downloads/evi_composite_output.png"
        # Plot and save the composite image
        plt.figure(figsize=(10, 10))
        plt.imshow(evi_concat, cmap='YlGn', extent=[
            longitude - 0.02, longitude + 0.02,
            latitude - 0.02, latitude + 0.02
        ])
        plt.axis('off')  # Remove axes
        plt.savefig(composite_output_filename_png, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        logging.error(f"Failed to generate EVI output: {e}")
        return jsonify({"error": "Failed to generate EVI output"}), 500

    return jsonify({
        "latitude": latitude,
        "longitude": longitude,
        "evi_geotiff_urls": evi_geotiff_urls,
        "evi_values": evi_values,
        "evi_composite_jpg": accessible_filename_png  # Returning the PNG path in the response
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)