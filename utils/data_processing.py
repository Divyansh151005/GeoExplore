import geopandas as gpd
import pandas as pd
import numpy as np
import os
import warnings
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import pyproj
from pyproj import CRS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_shapefile(shapefile_path):
    """
    Load a shapefile into a GeoDataFrame.
    
    Parameters:
    shapefile_path (str): Path to the shapefile
    
    Returns:
    GeoDataFrame or None: The loaded shapefile data, or None if loading fails
    """
    try:
        if not os.path.exists(shapefile_path):
            logger.warning(f"Shapefile not found: {shapefile_path}")
            return None
        
        # Suppress warnings from geopandas when loading files
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            gdf = gpd.read_file(shapefile_path)
            
        logger.info(f"Successfully loaded shapefile: {shapefile_path}")
        logger.info(f"CRS: {gdf.crs}")
        logger.info(f"Number of features: {len(gdf)}")
        
        return gdf
    except Exception as e:
        logger.error(f"Error loading shapefile {shapefile_path}: {str(e)}")
        return None

def transform_coordinates(gdf, target_crs="EPSG:4326"):
    """
    Transform GeoDataFrame to the target coordinate reference system.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    target_crs (str): Target CRS as EPSG code or proj4 string
    
    Returns:
    GeoDataFrame: Transformed GeoDataFrame
    """
    try:
        if gdf is None:
            return None
        
        # Check if GeoDataFrame has a defined CRS
        if gdf.crs is None:
            logger.warning("Input GeoDataFrame has no CRS defined. Assuming EPSG:4326 (WGS84).")
            gdf.set_crs(epsg=4326, inplace=True)
        
        # Transform to target CRS if needed
        if str(gdf.crs) != str(target_crs):
            logger.info(f"Transforming from {gdf.crs} to {target_crs}")
            gdf = gdf.to_crs(target_crs)
            
        return gdf
    except Exception as e:
        logger.error(f"Error transforming coordinates: {str(e)}")
        return gdf

def extract_features(gdf, feature_type='all'):
    """
    Extract specific features from the GeoDataFrame based on the feature type.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    feature_type (str): Type of features to extract ('all', 'point', 'line', 'polygon')
    
    Returns:
    GeoDataFrame: GeoDataFrame with extracted features
    """
    try:
        if gdf is None or len(gdf) == 0:
            return None
        
        if feature_type == 'all':
            return gdf
        
        # Filter based on geometry type
        if feature_type == 'point':
            return gdf[gdf.geometry.type.isin(['Point', 'MultiPoint'])]
        elif feature_type == 'line':
            return gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        elif feature_type == 'polygon':
            return gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        else:
            logger.warning(f"Unknown feature type: {feature_type}. Returning all features.")
            return gdf
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return gdf

def buffer_analysis(gdf, buffer_distance):
    """
    Create buffer zones around features in the GeoDataFrame.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    buffer_distance (float): Buffer distance in the units of the GeoDataFrame's CRS
    
    Returns:
    GeoDataFrame: GeoDataFrame with buffer geometries
    """
    try:
        if gdf is None or len(gdf) == 0:
            return None
        
        # Create buffer
        buffer_gdf = gdf.copy()
        buffer_gdf.geometry = buffer_gdf.geometry.buffer(buffer_distance)
        
        return buffer_gdf
    except Exception as e:
        logger.error(f"Error in buffer analysis: {str(e)}")
        return None

def spatial_join(gdf1, gdf2, how='intersection', predicate='intersects'):
    """
    Perform a spatial join between two GeoDataFrames.
    
    Parameters:
    gdf1 (GeoDataFrame): First GeoDataFrame
    gdf2 (GeoDataFrame): Second GeoDataFrame
    how (str): Type of join ('intersection', 'union')
    predicate (str): Spatial predicate for the join
                     ('intersects', 'contains', 'within', etc.)
    
    Returns:
    GeoDataFrame: Result of the spatial join
    """
    try:
        if gdf1 is None or gdf2 is None or len(gdf1) == 0 or len(gdf2) == 0:
            return None
        
        # Ensure both GeoDataFrames have the same CRS
        if gdf1.crs != gdf2.crs and gdf1.crs is not None and gdf2.crs is not None:
            gdf2 = gdf2.to_crs(gdf1.crs)
        
        # Perform spatial join
        joined = gpd.sjoin(gdf1, gdf2, how=how, predicate=predicate)
        
        return joined
    except Exception as e:
        logger.error(f"Error in spatial join: {str(e)}")
        return None

def calculate_density(gdf, grid_size=0.1):
    """
    Calculate feature density across the study area.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    grid_size (float): Size of the grid cells for density calculation
    
    Returns:
    GeoDataFrame: GeoDataFrame with density values
    """
    try:
        if gdf is None or len(gdf) == 0:
            return None
        
        # Get the bounds of the study area
        x_min, y_min, x_max, y_max = gdf.total_bounds
        
        # Create grid cells
        x_coords = np.arange(x_min, x_max + grid_size, grid_size)
        y_coords = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Create grid polygons
        grid_cells = []
        for x in x_coords[:-1]:
            for y in y_coords[:-1]:
                grid_cells.append(Polygon([
                    (x, y),
                    (x + grid_size, y),
                    (x + grid_size, y + grid_size),
                    (x, y + grid_size)
                ]))
        
        # Create GeoDataFrame for grid
        grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf.crs)
        
        # Count features in each grid cell
        intersect_counts = []
        for cell in grid_cells:
            cell_gdf = gpd.GeoDataFrame({'geometry': [cell]}, crs=gdf.crs)
            intersects = sum(gdf.intersects(cell))
            intersect_counts.append(intersects)
        
        grid_gdf['count'] = intersect_counts
        grid_gdf['density'] = grid_gdf['count'] / (grid_size * grid_size)
        
        return grid_gdf
    except Exception as e:
        logger.error(f"Error calculating density: {str(e)}")
        return None

def find_intersections(gdf1, gdf2):
    """
    Find intersection points or areas between two GeoDataFrames.
    
    Parameters:
    gdf1 (GeoDataFrame): First GeoDataFrame
    gdf2 (GeoDataFrame): Second GeoDataFrame
    
    Returns:
    GeoDataFrame: GeoDataFrame containing the intersections
    """
    try:
        if gdf1 is None or gdf2 is None or len(gdf1) == 0 or len(gdf2) == 0:
            return None
        
        # Ensure both GeoDataFrames have the same CRS
        if gdf1.crs != gdf2.crs and gdf1.crs is not None and gdf2.crs is not None:
            gdf2 = gdf2.to_crs(gdf1.crs)
        
        # Create list to store intersections
        intersections = []
        
        # Find intersections between each feature in gdf1 and gdf2
        for idx1, row1 in gdf1.iterrows():
            for idx2, row2 in gdf2.iterrows():
                if row1.geometry.intersects(row2.geometry):
                    intersection = row1.geometry.intersection(row2.geometry)
                    if not intersection.is_empty:
                        intersections.append({
                            'geometry': intersection,
                            'id1': idx1,
                            'id2': idx2
                        })
        
        # Create GeoDataFrame for intersections
        if intersections:
            intersect_gdf = gpd.GeoDataFrame(intersections, crs=gdf1.crs)
            return intersect_gdf
        else:
            return None
    except Exception as e:
        logger.error(f"Error finding intersections: {str(e)}")
        return None

def create_distance_raster(gdf, grid_size=0.01, max_distance=1.0):
    """
    Create a distance raster from features in the GeoDataFrame.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    grid_size (float): Size of the grid cells
    max_distance (float): Maximum distance to calculate
    
    Returns:
    tuple: (X grid coordinates, Y grid coordinates, distance values)
    """
    try:
        if gdf is None or len(gdf) == 0:
            return None, None, None
        
        # Get the bounds of the study area
        x_min, y_min, x_max, y_max = gdf.total_bounds
        
        # Create grid
        x = np.arange(x_min, x_max + grid_size, grid_size)
        y = np.arange(y_min, y_max + grid_size, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize distance array
        distances = np.ones_like(X) * max_distance
        
        # Calculate distance for each grid point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = Point(X[i, j], Y[i, j])
                
                # Find minimum distance to any feature
                min_dist = max_distance
                for geom in gdf.geometry:
                    dist = point.distance(geom)
                    min_dist = min(min_dist, dist)
                
                distances[i, j] = min_dist
        
        return X, Y, distances
    except Exception as e:
        logger.error(f"Error creating distance raster: {str(e)}")
        return None, None, None

def extract_attribute_stats(gdf, attribute_column):
    """
    Extract statistical information from a specific attribute column.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    attribute_column (str): Column name to analyze
    
    Returns:
    dict: Statistical information for the attribute
    """
    try:
        if gdf is None or len(gdf) == 0 or attribute_column not in gdf.columns:
            return None
        
        # Calculate statistics
        stats = {
            'count': gdf[attribute_column].count(),
            'min': gdf[attribute_column].min() if pd.api.types.is_numeric_dtype(gdf[attribute_column]) else None,
            'max': gdf[attribute_column].max() if pd.api.types.is_numeric_dtype(gdf[attribute_column]) else None,
            'mean': gdf[attribute_column].mean() if pd.api.types.is_numeric_dtype(gdf[attribute_column]) else None,
            'std': gdf[attribute_column].std() if pd.api.types.is_numeric_dtype(gdf[attribute_column]) else None,
            'unique_values': gdf[attribute_column].nunique(),
            'value_counts': gdf[attribute_column].value_counts().to_dict() if gdf[attribute_column].nunique() <= 20 else None
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error extracting attribute statistics: {str(e)}")
        return None
