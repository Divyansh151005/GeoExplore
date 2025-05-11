import folium
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_folium_map(gdf, layer_name, center_lat=15.0, center_lon=78.0, zoom_start=7):
    """
    Create a Folium map with the GeoDataFrame data.
    
    Parameters:
    gdf (GeoDataFrame): GeoDataFrame to visualize
    layer_name (str): Name for the layer
    center_lat (float): Latitude for map center
    center_lon (float): Longitude for map center
    zoom_start (int): Initial zoom level
    
    Returns:
    folium.Map: Folium map with the data layer
    """
    try:
        if gdf is None or len(gdf) == 0:
            logger.warning(f"Empty or None GeoDataFrame provided for {layer_name} layer")
            # Return a basic map centered on Karnataka/Andhra Pradesh
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
            return m
        
        # Set map center to GeoDataFrame centroid if not specified
        if center_lat is None or center_lon is None:
            center = gdf.unary_union.centroid
            center_lat = center.y
            center_lon = center.x
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)
        
        # Determine the type of geometry
        geom_types = gdf.geometry.type.unique()
        
        # Style function for different geometry types
        if 'Point' in geom_types or 'MultiPoint' in geom_types:
            # Point data - use MarkerCluster
            if len(gdf) > 100:  # Use clustering for large datasets
                marker_cluster = MarkerCluster(name=layer_name).add_to(m)
                for idx, row in gdf.iterrows():
                    tooltip = f"{layer_name} {idx}"
                    if hasattr(row, 'geometry') and row.geometry is not None:
                        folium.Marker(
                            location=[row.geometry.y, row.geometry.x],
                            tooltip=tooltip
                        ).add_to(marker_cluster)
            else:
                # Add points directly to the map
                for idx, row in gdf.iterrows():
                    tooltip = f"{layer_name} {idx}"
                    if hasattr(row, 'geometry') and row.geometry is not None:
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.7,
                            tooltip=tooltip
                        ).add_to(m)
        
        elif 'LineString' in geom_types or 'MultiLineString' in geom_types:
            # Line data
            if layer_name.lower() == 'faults':
                line_color = 'red'
            elif layer_name.lower() == 'folds':
                line_color = 'blue'
            else:
                line_color = 'green'
            
            # Add GeoJSON layer
            folium.GeoJson(
                gdf,
                name=layer_name,
                style_function=lambda x: {
                    'color': line_color,
                    'weight': 2,
                    'opacity': 0.7
                },
                tooltip=folium.GeoJsonTooltip(fields=gdf.columns[:min(3, len(gdf.columns))].tolist())
            ).add_to(m)
        
        elif 'Polygon' in geom_types or 'MultiPolygon' in geom_types:
            # Polygon data
            if layer_name.lower() == 'lithology':
                # Create a colormap based on lithology types
                # Try to find a column likely to contain lithology type
                type_column = None
                for col in gdf.columns:
                    if 'type' in col.lower() or 'lith' in col.lower() or 'rock' in col.lower():
                        type_column = col
                        break
                
                if type_column and type_column in gdf.columns:
                    # Create categorical colormap
                    categories = gdf[type_column].unique()
                    color_dict = {}
                    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
                    
                    for i, cat in enumerate(categories):
                        color_dict[cat] = colors[i % len(colors)]
                    
                    def style_function(feature):
                        cat = feature['properties'][type_column]
                        return {
                            'fillColor': color_dict.get(cat, 'gray'),
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.7
                        }
                    
                    folium.GeoJson(
                        gdf,
                        name=layer_name,
                        style_function=style_function,
                        tooltip=folium.GeoJsonTooltip(fields=[type_column])
                    ).add_to(m)
                else:
                    # Default style without categorization
                    folium.GeoJson(
                        gdf,
                        name=layer_name,
                        style_function=lambda x: {
                            'fillColor': 'green',
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.5
                        },
                        tooltip=folium.GeoJsonTooltip(fields=gdf.columns[:min(3, len(gdf.columns))].tolist())
                    ).add_to(m)
            else:
                # Default polygon style
                folium.GeoJson(
                    gdf,
                    name=layer_name,
                    style_function=lambda x: {
                        'fillColor': 'blue',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.5
                    },
                    tooltip=folium.GeoJsonTooltip(fields=gdf.columns[:min(3, len(gdf.columns))].tolist())
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    except Exception as e:
        logger.error(f"Error creating Folium map: {str(e)}")
        # Return a basic map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        return m

def plot_feature_distribution(gdf, column=None, plot_type='histogram', title=None, figsize=(10, 6)):
    """
    Plot the distribution of a feature in the GeoDataFrame.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    column (str): Column to plot (if None, will use geometry properties)
    plot_type (str): Type of plot ('histogram', 'bar', 'pie', 'box')
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: The plot figure
    """
    try:
        if gdf is None or len(gdf) == 0:
            logger.warning("Empty or None GeoDataFrame provided for feature distribution plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available for plotting",
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if column is None:
            # Use geometry properties
            geom_types = gdf.geometry.type.unique()
            
            if 'Point' in geom_types or 'MultiPoint' in geom_types:
                # For points, plot spatial distribution
                gdf.plot(ax=ax, color='blue', markersize=5)
                ax.set_title(title or "Spatial Distribution of Points")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.grid(True)
            
            elif 'LineString' in geom_types or 'MultiLineString' in geom_types:
                # For lines, plot length distribution
                if hasattr(gdf, 'length'):
                    gdf.length.plot.hist(ax=ax, bins=20, color='blue')
                    ax.set_title(title or "Distribution of Line Lengths")
                    ax.set_xlabel("Length")
                    ax.set_ylabel("Frequency")
                    ax.grid(True)
                else:
                    gdf.plot(ax=ax, color='blue')
                    ax.set_title(title or "Line Features")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.grid(True)
            
            elif 'Polygon' in geom_types or 'MultiPolygon' in geom_types:
                # For polygons, plot area distribution
                if hasattr(gdf, 'area'):
                    gdf.area.plot.hist(ax=ax, bins=20, color='green')
                    ax.set_title(title or "Distribution of Polygon Areas")
                    ax.set_xlabel("Area")
                    ax.set_ylabel("Frequency")
                    ax.grid(True)
                else:
                    gdf.plot(ax=ax, color='green', alpha=0.5)
                    ax.set_title(title or "Polygon Features")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.grid(True)
        
        else:
            # Plot specific column
            if column not in gdf.columns:
                raise ValueError(f"Column '{column}' not found in GeoDataFrame")
            
            data = gdf[column]
            
            if plot_type == 'histogram' and pd.api.types.is_numeric_dtype(data):
                data.plot.hist(ax=ax, bins=20, color='blue')
                ax.set_title(title or f"Distribution of {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                ax.grid(True)
            
            elif plot_type == 'bar':
                if pd.api.types.is_numeric_dtype(data):
                    # For numeric data, create bins
                    bins = pd.cut(data, 10)
                    bins.value_counts().sort_index().plot.bar(ax=ax, color='blue')
                    ax.set_title(title or f"Binned Distribution of {column}")
                else:
                    # For categorical data
                    data.value_counts().plot.bar(ax=ax, color='blue')
                    ax.set_title(title or f"Category Distribution of {column}")
                
                ax.set_xlabel(column)
                ax.set_ylabel("Count")
                ax.grid(True)
            
            elif plot_type == 'pie' and data.nunique() <= 10:
                data.value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
                ax.set_title(title or f"Proportion of {column} Values")
            
            elif plot_type == 'box' and pd.api.types.is_numeric_dtype(data):
                sns.boxplot(x=data, ax=ax)
                ax.set_title(title or f"Box Plot of {column}")
                ax.set_xlabel(column)
                ax.grid(True)
            
            else:
                logger.warning(f"Cannot create {plot_type} plot for column '{column}'")
                ax.text(0.5, 0.5, f"Cannot create {plot_type} plot for the given data",
                       horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting feature distribution: {str(e)}")
        # Return a basic figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
               horizontalalignment='center', verticalalignment='center')
        return fig

def create_heatmap(gdf, value_column=None, radius=15, center_lat=15.0, center_lon=78.0, zoom_start=7):
    """
    Create a heatmap using Folium.
    
    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame
    value_column (str): Column with values for heatmap intensity (if None, all points have equal weight)
    radius (int): Radius of influence for each point
    center_lat (float): Latitude for map center
    center_lon (float): Longitude for map center
    zoom_start (int): Initial zoom level
    
    Returns:
    folium.Map: Folium map with heatmap
    """
    try:
        if gdf is None or len(gdf) == 0:
            logger.warning("Empty or None GeoDataFrame provided for heatmap")
            # Return a basic map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
            return m
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)
        
        # Check geometry type
        geom_types = gdf.geometry.type.unique()
        
        # Extract points for heatmap
        if 'Point' in geom_types or 'MultiPoint' in geom_types:
            # For point geometries, use directly
            if value_column and value_column in gdf.columns and pd.api.types.is_numeric_dtype(gdf[value_column]):
                heat_data = [[row.geometry.y, row.geometry.x, row[value_column]] for idx, row in gdf.iterrows() 
                             if hasattr(row, 'geometry') and row.geometry is not None]
            else:
                heat_data = [[row.geometry.y, row.geometry.x, 1] for idx, row in gdf.iterrows() 
                             if hasattr(row, 'geometry') and row.geometry is not None]
        
        elif 'LineString' in geom_types or 'MultiLineString' in geom_types:
            # For line geometries, generate points along lines
            heat_data = []
            for idx, row in gdf.iterrows():
                if hasattr(row, 'geometry') and row.geometry is not None:
                    # Get coordinates from the line
                    if row.geometry.geom_type == 'LineString':
                        coords = list(row.geometry.coords)
                    else:  # MultiLineString
                        coords = [point for line in row.geometry.geoms for point in line.coords]
                    
                    # Add points to heat data
                    weight = row[value_column] if value_column and value_column in gdf.columns else 1
                    heat_data.extend([[y, x, weight] for x, y in coords])
        
        elif 'Polygon' in geom_types or 'MultiPolygon' in geom_types:
            # For polygon geometries, use centroids
            heat_data = []
            for idx, row in gdf.iterrows():
                if hasattr(row, 'geometry') and row.geometry is not None:
                    centroid = row.geometry.centroid
                    weight = row[value_column] if value_column and value_column in gdf.columns else 1
                    heat_data.append([centroid.y, centroid.x, weight])
        
        else:
            logger.warning(f"Unsupported geometry types for heatmap: {geom_types}")
            return m
        
        # Create heatmap layer
        HeatMap(
            heat_data,
            radius=radius,
            blur=15,
            max_zoom=zoom_start + 2,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
        ).add_to(m)
        
        return m
    
    except Exception as e:
        logger.error(f"Error creating heatmap: {str(e)}")
        # Return a basic map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
        return m

def plot_mineral_potential(x_grid, y_grid, potential_values, threshold=0.7, figsize=(12, 8)):
    """
    Plot mineral potential as a contour map with highlighted high-potential areas.
    
    Parameters:
    x_grid (numpy.ndarray): Grid of X coordinates
    y_grid (numpy.ndarray): Grid of Y coordinates
    potential_values (numpy.ndarray): Predicted potential values
    threshold (float): Threshold for high-potential areas
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: The plot figure
    """
    try:
        if x_grid is None or y_grid is None or potential_values is None:
            logger.warning("Missing data for mineral potential plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available for plotting",
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create contour plot
        contour = ax.contourf(x_grid, y_grid, potential_values, cmap='YlOrRd', levels=10)
        
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Mineral Potential')
        
        # Highlight high-potential areas
        high_potential = ax.contour(x_grid, y_grid, potential_values, 
                                     levels=[threshold], colors='red', linewidths=2)
        
        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Mineral Potential Map')
        ax.grid(True)
        
        # Add threshold legend
        ax.plot([], [], color='red', linewidth=2, label=f'High Potential (>{threshold})')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting mineral potential: {str(e)}")
        # Return a basic figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}",
               horizontalalignment='center', verticalalignment='center')
        return fig

def create_correlation_plot(df, figsize=(10, 8)):
    """
    Create a correlation matrix plot for numerical columns.
    
    Parameters:
    df (DataFrame): Input DataFrame
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: The correlation plot figure
    """
    try:
        if df is None or len(df) == 0:
            logger.warning("Empty or None DataFrame provided for correlation plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available for plotting",
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] < 2:
            logger.warning("Not enough numeric columns for correlation plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Not enough numeric data for correlation analysis",
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating correlation plot: {str(e)}")
        # Return a basic figure with error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error creating correlation plot: {str(e)}",
               horizontalalignment='center', verticalalignment='center')
        return fig

def create_3d_surface(x_grid, y_grid, z_values, figsize=(10, 8)):
    """
    Create a 3D surface plot.
    
    Parameters:
    x_grid (numpy.ndarray): Grid of X coordinates
    y_grid (numpy.ndarray): Grid of Y coordinates
    z_values (numpy.ndarray): Z values for the surface
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: The 3D plot figure
    """
    try:
        if x_grid is None or y_grid is None or z_values is None:
            logger.warning("Missing data for 3D surface plot")
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available for 3D plotting",
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(x_grid, y_grid, z_values, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Value')
        ax.set_title('3D Surface Plot')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error creating 3D surface plot: {str(e)}")
        # Return a basic figure with error message
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Error creating 3D plot: {str(e)}",
               horizontalalignment='center', verticalalignment='center')
        return fig
