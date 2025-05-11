# Data Description for GeoExplore Project

## Study Area
The project focuses on a study area of approximately 39,000 sq. km spanning portions of Karnataka and Andhra Pradesh in India. This region is known for its rich geological diversity and mineral potential.

## Primary Datasets

### 1. Lithology Dataset (lithology_gcs_ngdr.shp)
This dataset contains polygon features representing different rock units and formations in the study area.

**Key Attributes:**
- `ROCK_TYPE`: Main lithological classifications (e.g., granite, schist, basalt)
- `AGE`: Geological age of the rock units
- `FORMATION`: Name of the geological formation
- `COMPOSITION`: Mineralogical composition information
- `geometry`: Polygon geometry of each lithological unit

**Data Format:** WGS84 geographic coordinate system (EPSG:4326)
**Data Source:** Geological Survey of India (GSI) geospatial database

### 2. Fault Dataset (fault_gcs_ngdr_20250224141337303.shp)
This dataset contains linear features representing major and minor fault structures in the region.

**Key Attributes:**
- `TYPE`: Type of fault (e.g., normal, reverse, strike-slip)
- `LENGTH_KM`: Length of fault in kilometers
- `CONFIDENCE`: Confidence level of the fault interpretation (high, medium, low)
- `DISPLACEMENT`: Amount of displacement where measured
- `geometry`: LineString geometry of each fault

**Data Format:** WGS84 geographic coordinate system (EPSG:4326)
**Data Source:** Geological Survey of India (GSI) geospatial database

### 3. Fold Dataset (Fold.shp)
This dataset contains linear features representing fold axes in the region.

**Key Attributes:**
- `FOLD_TYPE`: Type of fold (e.g., anticline, syncline)
- `AMPLITUDE`: Amplitude of the fold where measured
- `PLUNGE`: Plunge direction and angle
- `WAVELENGTH`: Wavelength of the fold where measured
- `geometry`: LineString geometry of each fold axis

**Data Format:** WGS84 geographic coordinate system (EPSG:4326)
**Data Source:** Geological Survey of India (GSI) geospatial database

## Derived Datasets and Features

### 1. Fault Density Map
A raster dataset calculated from the fault line data, representing the density of faults per unit area.

**Generation Method:** Kernel density estimation with a search radius of 5km
**Resolution:** 100m per pixel
**Units:** Linear km of faults per sq. km

### 2. Distance Rasters
Raster datasets representing the distance from each location to the nearest:
- Fault
- Fold axis
- Lithological contact

**Resolution:** 100m per pixel
**Units:** Kilometers

### 3. Geological Intersections
Point features representing the intersections between:
- Fault-fault intersections
- Fault-fold intersections
- Fault-lithological contact intersections

**Attributes:**
- `TYPE`: Type of intersection
- `ANGLE`: Angle of intersection where applicable
- `geometry`: Point geometry of each intersection

## Training Datasets for ML Models

For each mineral type (REE, Ni-PGE, copper, diamond, iron, manganese, and gold), we prepared training datasets containing:

1. **Known Occurrences:** Point locations of known mineral occurrences (positive samples)
2. **Negative Samples:** Random points verified to not contain the target mineral
3. **Feature Variables:**
   - Distance to nearest fault
   - Distance to nearest fold axis
   - Distance to lithological contacts
   - Fault density
   - Lithology type (categorical)
   - Distance to fault-fold intersections
   - Other geological parameter values at each point

**Sample Sizes:**
- Gold model: 84 samples (42 positive, 42 negative)
- REE model: 76 samples (38 positive, 38 negative)
- Copper model: 92 samples (46 positive, 46 negative)
- Other minerals: 60-80 samples per mineral type

## Data Preprocessing Steps

1. **Coordinate Standardization:** All datasets were transformed to a common WGS84 geographic coordinate system
2. **Topological Cleaning:** Fixing of overlaps, gaps, and dangles in spatial datasets
3. **Attribute Standardization:** Normalization of attribute naming and values for consistency
4. **Categorical Encoding:** One-hot encoding of categorical variables like rock types
5. **Feature Scaling:** Normalization of numerical features to 0-1 range for model training
6. **Validation Splitting:** Random stratified sampling to create training and validation subsets

## Data Limitations and Assumptions

1. **Resolution Limitations:** Base geological maps were compiled at 1:50,000 scale, limiting the precision of smaller features
2. **Incomplete Subsurface Data:** Limited borehole and geophysical data means deeper structures are interpreted with lower confidence
3. **Age Considerations:** The datasets represent the most current geological understanding but may not reflect recent discoveries
4. **Spatial Uncertainty:** Positional accuracy of geological contacts and structures varies from 10-50m
5. **Known Occurrence Bias:** Training data for known mineral occurrences may be biased toward easily discoverable, surface or near-surface deposits