# SatelliteSR-Backend

This is a Flask-based backend built for the NASA SpaceApps 2024 challenge, titled "Landsat Reflectance Data: On the Fly and at Your Fingertips."

As part of this challenge, the backend provides functionality for performing data transformation on HLS (Harmonized Landsat-Sentinel) data for calculating the Enhanced Vegetation Index (EVI), along with managing satellite overpass data for Landsat 8 and Landsat 9.

## Features

- **Calculate EVI**: Automatically downloads relevant HLS data and calculates the Enhanced Vegetation Index for given coordinates.
- **Satellite Overpass Calculation**: Retrieves upcoming and historical overpasses for Landsat 8 and Landsat 9, providing insights into satellite observation windows.
- **GeoTIFF Generation**: Generates GeoTIFFs for 3x3 grids around selected locations, providing easy visualization and data usage.
- **Data Filtering**: Filters results by cloud coverage to ensure only high-quality data is considered for analysis.

## Setup

### Prerequisites
- Python 3.8+
- [Conda](https://docs.conda.io/en/latest/miniconda.html) for package management

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SatelliteSR-Backend.git
   cd SatelliteSR-Backend
   ```

2. Set up the environment using Conda:
   ```bash
   conda env create -f environment.yml
   conda activate satellitesr-backend
   ```

3. A [NASA Earthdata Login account](https://urs.earthdata.nasa.gov/) is required to download the data used in this tutorial. You can create an account at the link provided.

4. Create the `.env` file with the credentials from NASA Earthdata account.
   ```
    EARTHDATA_USERNAME="<username>"
    EARTHDATA_PASSWORD="<password>"    
   ```

4. Run the Flask server:
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5001` by default.

### Endpoints

- **/overpass** (POST):
  - Finds the next overpasses for Landsat 8 and Landsat 9 for a given location within the next 7 days.
  - Request Body:
    ```json
    {
      "latitude": <float>,
      "longitude": <float>
    }
    ```

- **/historical_overpass** (POST):
  - Finds historical overpasses and associated metadata for Landsat 8 and Landsat 9 for a given time window.
  - Request Body:
    ```json
    {
      "latitude": <float>,
      "longitude": <float>,
      "start_date": "<yyyy-mm-dd>",
      "end_date": "<yyyy-mm-dd>"
    }
    ```

- **/evi** (POST):
  - Downloads HLS data and calculates EVI for a 3x3 grid around a given location, generating PNG and GeoTIFF files for visualization.
  - Request Body:
    ```json
    {
      "latitude": <float>,
      "longitude": <float>
    }
    ```

## Directory Structure

```
SatelliteSR-Backend/
├── app.py               # Main application file for Flask backend
├── downloads/           # Folder for storing downloaded GeoTIFF and PNG files
├── environment.yml      # Conda environment file
├── .env                 # Environment variables
├── README.md            # Project documentation
└── test_evi.ipynb       # Jupyter notebook for testing EVI calculations
```

## How It Works

1. **Satellite Overpass Retrieval**: The backend calculates future or historical overpasses for Landsat satellites over a given location using TLE (Two-Line Elements) data and the Skyfield library.
2. **HLS Data Access**: The backend uses the `earthaccess` library to search and download HLS data from NASA's data servers, based on given geographical coordinates.
3. **EVI Calculation and Visualization**: The downloaded data is processed using raster tools (`rioxarray`, `gdal`) to calculate EVI and generate downloadable files.

## Use Cases
- **Agriculture Monitoring**: Use EVI data to track crop health and vegetation growth, helping farmers make data-driven decisions.
- **Environmental Studies**: Assess vegetation cover and the effects of climate change or land use changes over time.
- **Disaster Response**: Monitor areas affected by natural disasters like wildfires or droughts by analyzing changes in vegetation indices.
- **Urban Planning**: Assist city planners to study urban sprawl and its effects on surrounding vegetation.
- **Biodiversity Conservation**: Track changes in vegetation in protected areas to monitor ecosystem health and biodiversity.
- **Research and Education**: Provide datasets and visualizations for academic research and educational purposes related to earth observation and environmental monitoring.

## Contributing
Contributions are welcome! If you have suggestions or feature requests, please create an issue or open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## The Team
- Jasper Grant ([@JasperGrant](https://github.com/JasperGrant))
- Aniq Elahi ([@Aniq-byte](https://github.com/Aniq-byte))
- Paras Nath Seth ([@parass05](https://github.com/parass05))
- Christian Simoneau ([@ChrisSimoneau](https://github.com/ChrisSimoneau))
- Aishik Sanyal ([@Xcellect](https://github.com/Xcellect))
