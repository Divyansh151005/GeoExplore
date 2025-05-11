import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import numpy as np
import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from matplotlib.backends.backend_pdf import PdfPages

# Set up styles
styles = getSampleStyleSheet()
title_style = styles['Title']
heading1_style = styles['Heading1']
heading2_style = styles['Heading2']
normal_style = styles['Normal']

# Create a PDF document
def create_report_pdf():
    doc = SimpleDocTemplate("GeoExplore_Project_Report.pdf", pagesize=letter)
    story = []

    # Title
    story.append(Paragraph("GeoExplore: Geological Data Analysis Platform for Mineral Exploration", title_style))
    story.append(Paragraph("Mineral Exploration Target Identification in Karnataka and Andhra Pradesh", heading2_style))
    story.append(Spacer(1, 0.2*inch))

    # Participant Details
    story.append(Paragraph("Participant Details", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>Team Name:</b> Visionary", normal_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>Team Leader:</b>", normal_style))
    story.append(Paragraph("Divyansh Barodiya", normal_style))
    story.append(Paragraph("B.Tech. Computer Science and Engineering, 2nd Year", normal_style))
    story.append(Paragraph("Indian Institute of Technology Ropar", normal_style))
    story.append(Paragraph("Email: 2023csb1119@iitrpr.ac.in", normal_style))
    story.append(Paragraph("Mobile: 7387142321", normal_style))
    story.append(Spacer(1, 0.2*inch))

    # Resources Used
    story.append(Paragraph("Resources Used", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("<b>Hardware</b>", heading2_style))
    story.append(Paragraph("• Cloud-based computational resources", normal_style))
    story.append(Paragraph("• High-performance computing for geospatial data processing", normal_style))
    story.append(Paragraph("• 16 GB RAM, 4-core CPU workstations for data analysis", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("<b>Software</b>", heading2_style))
    story.append(Paragraph("• Programming Languages: Python 3.11", normal_style))
    story.append(Paragraph("• GIS Processing Libraries: GeoPandas, Shapely, PyProj", normal_style))
    story.append(Paragraph("• Data Analysis Libraries: NumPy, Pandas, SciPy", normal_style))
    story.append(Paragraph("• Machine Learning: Scikit-learn", normal_style))
    story.append(Paragraph("• Visualization Tools: Matplotlib, Seaborn, Folium", normal_style))
    story.append(Paragraph("• Web Interface: Streamlit", normal_style))
    story.append(Paragraph("• Version Control: Git", normal_style))
    story.append(Spacer(1, 0.2*inch))

    # Data Used
    story.append(Paragraph("Data Used", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("<b>Geological Datasets</b>", heading2_style))
    story.append(Paragraph("1. <b>Lithology Data</b> (lithology_gcs_ngdr.shp)", normal_style))
    story.append(Paragraph("   • Comprehensive rock type information", normal_style))
    story.append(Paragraph("   • Geological ages and formations", normal_style))
    story.append(Paragraph("   • WGS84 geographic coordinate system", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("2. <b>Fault Line Data</b> (fault_gcs_ngdr_20250224141337303.shp)", normal_style))
    story.append(Paragraph("   • Fault types, orientations, and lengths", normal_style))
    story.append(Paragraph("   • Critical structures influencing mineralization", normal_style))
    story.append(Paragraph("   • WGS84 geographic coordinate system", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("3. <b>Fold Structure Data</b> (Fold.shp)", normal_style))
    story.append(Paragraph("   • Fold types and orientations", normal_style))
    story.append(Paragraph("   • Associated structural features", normal_style))
    story.append(Paragraph("   • WGS84 geographic coordinate system", normal_style))
    story.append(Spacer(1, 0.2*inch))

    # Add more sections from the report...
    story.append(Paragraph("Methodology", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("<b>Data Processing and Curation Workflow</b>", heading2_style))
    story.append(Paragraph("1. <b>Data Acquisition and Preparation</b>", normal_style))
    story.append(Paragraph("   • Importing shapefile data (lithology, faults, folds)", normal_style))
    story.append(Paragraph("   • Coordinate system validation and transformation (standardized to WGS84)", normal_style))
    story.append(Paragraph("   • Topology checking and error correction", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("2. <b>Feature Extraction and Transformation</b>", normal_style))
    story.append(Paragraph("   • Buffer generation around key structures", normal_style))
    story.append(Paragraph("   • Distance raster creation", normal_style))
    story.append(Paragraph("   • Density surface calculation", normal_style))
    story.append(Paragraph("   • Intersection analysis between datasets", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("3. <b>Statistical Analysis</b>", normal_style))
    story.append(Paragraph("   • Spatial correlation analysis", normal_style))
    story.append(Paragraph("   • Proximity statistics calculation", normal_style))
    story.append(Paragraph("   • Feature distribution analysis", normal_style))
    story.append(Paragraph("   • Anomaly detection", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("4. <b>Integration and Modeling</b>", normal_style))
    story.append(Paragraph("   • Data layer integration", normal_style))
    story.append(Paragraph("   • Feature importance assessment", normal_style))
    story.append(Paragraph("   • Predictive model generation", normal_style))
    story.append(Paragraph("   • Model validation and refinement", normal_style))
    story.append(Spacer(1, 0.2*inch))

    # Machine Learning Approach
    story.append(Paragraph("<b>Machine Learning Approach</b>", heading2_style))
    story.append(Paragraph("We employed a multi-model approach to mineral potential mapping:", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("1. <b>Random Forest Classification</b>", normal_style))
    story.append(Paragraph("   • Capable of handling non-linear relationships", normal_style))
    story.append(Paragraph("   • Feature importance assessment for understanding key drivers", normal_style))
    story.append(Paragraph("   • Robust to outliers and noise in geological data", normal_style))
    story.append(Paragraph("   • Achieved 85-97% accuracy depending on mineral type", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("2. <b>Support Vector Machines (SVM)</b>", normal_style))
    story.append(Paragraph("   • Effective for creating optimal decision boundaries", normal_style))
    story.append(Paragraph("   • Kernel-based approach to handle complex relationships", normal_style))
    story.append(Paragraph("   • Particularly effective for Ni-PGE and copper targeting", normal_style))
    story.append(Paragraph("   • 82-87% accuracy with RBF kernel", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("3. <b>Logistic Regression</b>", normal_style))
    story.append(Paragraph("   • Probabilistic interpretation of results", normal_style))
    story.append(Paragraph("   • More interpretable model for geoscientists", normal_style))
    story.append(Paragraph("   • Used for validation of more complex models", normal_style))
    story.append(Paragraph("   • 76-82% accuracy across mineral types", normal_style))
    story.append(Spacer(1, 0.2*inch))

    # Outcome / Result
    story.append(Paragraph("Outcome / Result", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("<b>Predictive Maps</b>", heading2_style))
    story.append(Paragraph("The GeoExplore platform has generated mineral potential maps for the 39,000 sq. km area covering parts of Karnataka and Andhra Pradesh. These maps integrate multiple geological features and machine learning predictions to identify high-priority exploration targets.", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("Key findings include:", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("1. <b>Gold Exploration Targets</b>", normal_style))
    story.append(Paragraph("   • 12 high-potential zones identified", normal_style))
    story.append(Paragraph("   • Concentrated along major fault systems in the north-central region", normal_style))
    story.append(Paragraph("   • Total area of ~850 sq. km of high-potential zones", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("2. <b>REE Potential Areas</b>", normal_style))
    story.append(Paragraph("   • 8 significant target zones", normal_style))
    story.append(Paragraph("   • Associated with alkaline intrusives in the eastern portion", normal_style))
    story.append(Paragraph("   • Covering approximately 620 sq. km", normal_style))
    story.append(Spacer(1, 0.2*inch))

    # Conclusion
    story.append(Paragraph("Conclusion", heading1_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("The GeoExplore platform demonstrates the power of integrating traditional geological knowledge with advanced data analytics and machine learning. By systematically processing, analyzing, and modeling multi-layer geological data, we have identified numerous high-potential exploration targets across the Karnataka and Andhra Pradesh region.", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("This project emphasizes the importance of a knowledge-driven approach to mineral exploration, where machine learning algorithms enhance rather than replace geological expertise. The identified targets represent promising areas for focused exploration efforts, potentially leading to the discovery of new mineral resources critical for India's technological and industrial development.", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<i>Report prepared by Team Visionary</i>", normal_style))
    story.append(Paragraph("<i>May 2025</i>", normal_style))

    # Build the PDF
    doc.build(story)
    return "GeoExplore_Project_Report.pdf"

# Main function to run when script is executed
if __name__ == "__main__":
    pdf_path = create_report_pdf()
    print(f"Report PDF created successfully at: {pdf_path}")