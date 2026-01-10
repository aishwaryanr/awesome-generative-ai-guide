#!/usr/bin/env python3
"""
Build script for the State of AI 2025 presentation.
Combines section files into a single index.html.

Usage:
    python build.py          # Build index.html from sections
    python build.py --split  # Split current index.html into sections
"""

import os
import re
import sys

SECTIONS_DIR = "sections"
OUTPUT_FILE = "index.html"
STYLES_FILE = "styles.css"
SCRIPT_FILE = "script.js"

# Section files in order
SECTION_FILES = [
    "00-opening.html",
    "01-input-layer.html",
    "02-model-layer.html",
    "03-application-layer.html",
    "04-output-layer.html",
    "05-challenges.html",
    "06-road-ahead.html",
    "07-closing.html",
]

HTML_HEADER = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>State of Applied AI in 2025</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="progress-bar" id="progress"></div>

    <div class="presentation" id="presentation">
'''

HTML_FOOTER = '''
    </div>

    <nav class="nav">
        <button onclick="prevSlide()">← Prev</button>
        <span class="counter"><span id="current">1</span> / <span id="total">80</span></span>
        <button onclick="nextSlide()">Next →</button>
    </nav>

    <script src="script.js"></script>
</body>
</html>
'''


def build():
    """Combine all section files into index.html"""
    content = HTML_HEADER

    for section_file in SECTION_FILES:
        filepath = os.path.join(SECTIONS_DIR, section_file)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                section_content = f.read()
                content += f"\n{section_content}\n"
            print(f"Added: {section_file}")
        else:
            print(f"Warning: {section_file} not found, skipping")

    content += HTML_FOOTER

    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)

    print(f"\nBuilt {OUTPUT_FILE} successfully!")


def split():
    """Split current index.html into section files"""
    if not os.path.exists(SECTIONS_DIR):
        os.makedirs(SECTIONS_DIR)

    with open(OUTPUT_FILE, 'r') as f:
        lines = f.readlines()

    # Define section boundaries (line numbers from grep, 1-indexed)
    sections = [
        ("00-opening.html", 185, 305, "Opening slides (1-10)"),
        ("01-input-layer.html", 306, 483, "Section 1: Input Layer"),
        ("02-model-layer.html", 484, 794, "Section 2: Model Layer"),
        ("03-application-layer.html", 795, 1031, "Section 3: Application Layer"),
        ("04-output-layer.html", 1032, 1241, "Section 4: Output Layer"),
        ("05-challenges.html", 1242, 1429, "Section 5: What's Broken"),
        ("06-road-ahead.html", 1430, 1575, "Section 6: Road Ahead"),
        ("07-closing.html", 1576, 1633, "Closing slides"),
    ]

    for filename, start, end, description in sections:
        # Convert to 0-indexed
        section_lines = lines[start-1:end]
        section_content = ''.join(section_lines)

        filepath = os.path.join(SECTIONS_DIR, filename)
        with open(filepath, 'w') as f:
            f.write(section_content)

        print(f"Created: {filename} ({description})")

    print(f"\nSplit into {len(sections)} section files in {SECTIONS_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--split":
        split()
    else:
        build()
