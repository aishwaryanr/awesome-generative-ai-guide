#!/usr/bin/env python3
"""
Generate a PDF from all markdown chapters
"""

import os
import subprocess
import tempfile

def main():
    # Get all chapter files in order
    chapter_files = []
    chapters_dir = "./chapters"
    
    # Add files in numerical order
    for i in range(1, 11):
        filename = f"{i:02d}_*.md"
        for file in sorted(os.listdir(chapters_dir)):
            if file.startswith(f"{i:02d}_"):
                chapter_files.append(os.path.join(chapters_dir, file))
                break
    
    # Create combined markdown file
    combined_content = """# AI Evals for Everyone - Complete Course

**Created by Aishwarya Naresh Reganti & Kiriti Badam**

---

"""
    
    # Add each chapter
    for chapter_file in chapter_files:
        with open(chapter_file, 'r') as f:
            content = f.read()
            # Add page break before each chapter except the first
            if chapter_file != chapter_files[0]:
                combined_content += "\n\\newpage\n\n"
            combined_content += content + "\n\n"
    
    # Write combined content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp.write(combined_content)
        tmp_path = tmp.name
    
    # Generate PDF using pandoc
    output_file = "AI_Evals_for_Everyone_Complete_Course.pdf"
    
    # Check if pandoc is installed
    try:
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
        
        # Run pandoc with nice formatting options
        cmd = [
            'pandoc',
            tmp_path,
            '-o', output_file,
            '--pdf-engine=xelatex',
            '--toc',
            '--toc-depth=2',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'linkcolor=blue',
            '-V', 'urlcolor=blue',
            '--highlight-style=tango'
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✅ PDF generated successfully: {output_file}")
        
    except FileNotFoundError:
        print("❌ pandoc is not installed. Please install it first:")
        print("   brew install pandoc")
        print("   brew install --cask mactex-no-gui  # for LaTeX support")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating PDF: {e}")
        print("Make sure you have LaTeX installed for PDF generation")
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()