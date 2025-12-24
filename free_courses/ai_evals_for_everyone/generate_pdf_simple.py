#!/usr/bin/env python3
"""
Simple PDF generator using reportlab
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import re

def create_pdf():
    # Read the markdown file
    with open('AI_Evals_Text_Only_Clean.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF
    doc = SimpleDocTemplate(
        "AI_Evals_for_Everyone_Complete_Course.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor='#d97706'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=30,
        textColor='#1f2937'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=15,
        spaceBefore=20,
        textColor='#374151'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leading=14
    )
    
    # Story container
    story = []
    
    # Split content into lines
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
            
        if line.startswith('# '):
            # Main title
            text = line[2:].strip()
            story.append(Paragraph(text, title_style))
            story.append(PageBreak())
        elif line.startswith('## '):
            # Chapter heading
            text = line[3:].strip()
            story.append(Paragraph(text, heading1_style))
        elif line.startswith('### '):
            # Section heading
            text = line[4:].strip()
            story.append(Paragraph(text, heading2_style))
        elif line.startswith('**') and line.endswith('**'):
            # Bold text
            text = f"<b>{line[2:-2]}</b>"
            story.append(Paragraph(text, body_style))
        elif line.startswith('- ') or line.startswith('* '):
            # List items
            text = f"• {line[2:]}"
            story.append(Paragraph(text, body_style))
        elif re.match(r'^\d+\.', line):
            # Numbered lists
            story.append(Paragraph(line, body_style))
        elif line.startswith('---'):
            # Horizontal rule
            story.append(Spacer(1, 20))
        else:
            # Regular paragraph
            if line:
                story.append(Paragraph(line, body_style))
    
    # Build PDF
    doc.build(story)
    print("✅ PDF created successfully: AI_Evals_for_Everyone_Complete_Course.pdf")

if __name__ == "__main__":
    create_pdf()