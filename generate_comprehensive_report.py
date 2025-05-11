import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
import re

def clean_markdown_for_pdf(text):
    """Clean markdown text for PDF conversion."""
    # Remove markdown heading markers but keep the text
    text = re.sub(r'^#+ (.*?)$', r'\1', text, flags=re.MULTILINE)
    # Remove markdown bold markers but keep the text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove markdown emphasis markers but keep the text
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    return text

def generate_comprehensive_report():
    """Generate a comprehensive PDF report with images for all sections."""
    
    # Make sure visualizations directory exists
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Output PDF path
    output_path = "GeoExplore_Comprehensive_Report.pdf"
    
    # Create the PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=20,
        alignment=1  # Center alignment
    )
    
    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        spaceBefore=24
    )
    
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=16
    )
    
    normal_style = ParagraphStyle(
        'BodyText',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=8,
        leading=14
    )
    
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=1,  # Center alignment
        fontStyle='italic'
    )
    
    # Document elements list
    elements = []
    
    # Add title
    elements.append(Paragraph("GeoExplore Project", title_style))
    elements.append(Paragraph("AI-Powered Mineral Potential Mapping Platform", heading2_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add project information
    info_data = [
        ["Date:", "May 11, 2025"],
        ["Team Name:", "Visionary"],
        ["Participant:", "Divyansh Barodiya"],
        ["Affiliation:", "IIT Ropar, CSE Department (2nd Year)"],
        ["Email:", "2023csb1119@iitrpr.ac.in"],
        ["Mobile:", "7387142321"]
    ]
    
    info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    
    elements.append(info_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # Load report content from the Markdown file
    with open("GeoExplore_Project_Report.md", "r") as file:
        report_content = file.read()
    
    # Extract sections from the markdown
    sections = report_content.split('###')
    
    # Skip the title and project info (first section)
    for section in sections[2:]:  # Start from the Executive Summary
        if not section.strip():
            continue
            
        # Extract section title and content
        lines = section.strip().split('\n')
        section_title = lines[0].strip()
        section_content = '\n'.join(lines[1:]).strip()
        
        # Add section title
        elements.append(Paragraph(section_title, heading1_style))
        
        # Special handling for sections with images
        if section_title == "Results and Outcomes":
            # Add explanatory text
            paragraphs = section_content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if "Gold Potential Mapping" in para and i == 0:
                    elements.append(Paragraph(clean_markdown_for_pdf(para), normal_style))
                    
                    # Add gold potential map
                    if os.path.exists("visualizations/gold_potential_map.png"):
                        img = Image("visualizations/gold_potential_map.png", width=6*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Paragraph("Figure 1: Gold Mineral Potential Map showing high-probability zones", caption_style))
                        elements.append(Spacer(1, 0.2*inch))
                        
                    # Add feature importance
                    if os.path.exists("visualizations/feature_importance.png"):
                        img = Image("visualizations/feature_importance.png", width=6*inch, height=3*inch)
                        elements.append(img)
                        elements.append(Paragraph("Figure 2: Feature Importance for Gold Potential Model", caption_style))
                        elements.append(Spacer(1, 0.3*inch))
                        
                elif "REE Potential Mapping" in para:
                    elements.append(Paragraph(clean_markdown_for_pdf(para), normal_style))
                    
                    # Add REE potential map
                    if os.path.exists("visualizations/ree_potential_map.png"):
                        img = Image("visualizations/ree_potential_map.png", width=6*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Paragraph("Figure 3: Rare Earth Elements (REE) Potential Map showing target zones", caption_style))
                        elements.append(Spacer(1, 0.3*inch))
                        
                elif "Copper and Ni-PGE Potential Mapping" in para:
                    elements.append(Paragraph(clean_markdown_for_pdf(para), normal_style))
                    elements.append(Spacer(1, 0.3*inch))
                    
        elif section_title == "Methodology":
            # Add content
            paragraphs = section_content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if "Data Preparation" in para and i == 0:
                    elements.append(Paragraph(clean_markdown_for_pdf(para), normal_style))
                    
                    # Add fault density map near data preparation section
                    if os.path.exists("visualizations/fault_density_heatmap.png"):
                        img = Image("visualizations/fault_density_heatmap.png", width=6*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Paragraph("Figure 4: Fault Density Analysis Map for Karnataka & Andhra Pradesh", caption_style))
                        elements.append(Spacer(1, 0.3*inch))
                        
                else:
                    elements.append(Paragraph(clean_markdown_for_pdf(para), normal_style))
                    
        elif section_title == "Machine Learning Implementation":
            # Add content first
            elements.append(Paragraph(clean_markdown_for_pdf(section_content), normal_style))
            
            # Add model comparison chart
            if os.path.exists("visualizations/model_comparison.png"):
                img = Image("visualizations/model_comparison.png", width=6*inch, height=3.5*inch)
                elements.append(img)
                elements.append(Paragraph("Figure 5: Performance Comparison of Different ML Models", caption_style))
                elements.append(Spacer(1, 0.2*inch))
                
            # Add confusion matrix
            if os.path.exists("visualizations/confusion_matrix.png"):
                img = Image("visualizations/confusion_matrix.png", width=5*inch, height=3.5*inch)
                elements.append(img)
                elements.append(Paragraph("Figure 6: Confusion Matrix for Gold Potential Model", caption_style))
                elements.append(Spacer(1, 0.3*inch))
                
        elif section_title == "Validation and Confidence Assessment":
            # Add page break before validation section
            elements.append(PageBreak())
            elements.append(Paragraph(section_title, heading1_style))
            elements.append(Paragraph(clean_markdown_for_pdf(section_content), normal_style))
                
        else:
            # Standard paragraph handling for other sections
            paragraphs = section_content.split('\n\n')
            for para in paragraphs:
                clean_para = clean_markdown_for_pdf(para)
                if clean_para.strip():
                    elements.append(Paragraph(clean_para, normal_style))
    
    # Add Data Description section with new page
    elements.append(PageBreak())
    elements.append(Paragraph("Data Description", heading1_style))
    
    with open("data_description.md", "r") as file:
        data_description = file.read()
    
    # Extract sections from data description
    data_sections = data_description.split('##')
    
    # Skip the title (first section)
    for section in data_sections[1:]:
        if not section.strip():
            continue
            
        # Extract section title and content
        lines = section.strip().split('\n')
        section_title = lines[0].strip()
        section_content = '\n'.join(lines[1:]).strip()
        
        # Add section title
        elements.append(Paragraph(section_title, heading2_style))
        
        # Add content
        paragraphs = section_content.split('\n\n')
        for para in paragraphs:
            clean_para = clean_markdown_for_pdf(para)
            if clean_para.strip():
                elements.append(Paragraph(clean_para, normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    print(f"Comprehensive report created successfully at: {output_path}")

if __name__ == "__main__":
    generate_comprehensive_report()