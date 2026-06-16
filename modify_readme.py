import re

with open("README.md", "r") as f:
    content = f.read()

# Remove the frontend installation step
content = re.sub(r'### 5\) Install Node\.js/npm and frontend dependencies.*?(?=\n---\n)', '', content, flags=re.DOTALL)

# Remove the Data Dictionary section
content = re.sub(r'##\s*Data Dictionary \(For UI Tooltips & Labels\).*?(?=\n---\n|\Z)', '', content, flags=re.DOTALL)

# Remove the Integration Notes for Frontend section
content = re.sub(r'## Integration Notes for Frontend.*?(?=\n---\n|\Z)', '', content, flags=re.DOTALL)

# Clean up multiple empty lines and excessive separators
content = re.sub(r'\n---\n\s*\n---\n', '\n---\n', content)
content = re.sub(r'\n{3,}', '\n\n', content)

# Add the link to the frontend documentation and URL
frontend_section = """
## Frontend Documentation & Application

The frontend user interface is built as a separate application inside the `rice-navigator/` directory.

- **Frontend Application URL**: [Rice Navigator Frontend](https://rice-navigator.lovable.app)
- **Frontend Documentation**: See [rice-navigator/README.md](rice-navigator/README.md) for UI data dictionary, integration notes, and run commands.
"""

content += frontend_section

with open("README.md", "w") as f:
    f.write(content.strip() + '\n')

