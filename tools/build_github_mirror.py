#!/usr/bin/env python3
"""
Builds a standalone, 100% self-contained GitHub HTML mirror of README.md
with all image assets embedded as Base64 data URIs.
"""

import base64
import re
import json
from pathlib import Path

repo_dir = Path(__file__).resolve().parent.parent
readme_path = repo_dir / "README.md"
out_html = Path.home() / "Desktop" / "STUDIOLIGHT_GITHUB_MIRROR.html"

with open(readme_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# Replace image paths with base64 data URIs
def get_mime(path_str):
    if path_str.endswith(".svg"):
        return "image/svg+xml"
    elif path_str.endswith(".jpg") or path_str.endswith(".jpeg"):
        return "image/jpeg"
    elif path_str.endswith(".gif"):
        return "image/gif"
    elif path_str.endswith(".webp"):
        return "image/webp"
    return "image/png"

def embed_b64_html(match):
    rel_path = match.group(1)
    full_path = repo_dir / rel_path
    if full_path.exists():
        mime = get_mime(rel_path)
        with open(full_path, "rb") as img_f:
            b64_data = base64.b64encode(img_f.read()).decode("utf-8")
        return f'src="data:{mime};base64,{b64_data}"'
    return match.group(0)

def embed_b64_md(match):
    alt = match.group(1)
    rel_path = match.group(2)
    full_path = repo_dir / rel_path
    if full_path.exists():
        mime = get_mime(rel_path)
        with open(full_path, "rb") as img_f:
            b64_data = base64.b64encode(img_f.read()).decode("utf-8")
        return f'![{alt}](data:{mime};base64,{b64_data})'
    return match.group(0)

converted_md = re.sub(r'src="(assets/[^"]+)"', embed_b64_html, md_content)
converted_md = re.sub(r'!\[([^\]]*)\]\((assets/[^\)]+)\)', embed_b64_md, converted_md)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ComfyUI-StudioLight — GitHub Mirror</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <style>
        body {{
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 32px 16px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 1012px;
            width: 100%;
            background-color: #0d1117;
            padding: 32px;
            border-radius: 8px;
            border: 1px solid #30363d;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        .markdown-body {{
            background-color: transparent !important;
            color: #c9d1d9 !important;
            font-size: 15px;
            line-height: 1.6;
        }}
        .markdown-body table {{
            display: table !important;
            width: 100% !important;
            margin-bottom: 24px;
            border-collapse: collapse;
        }}
        .markdown-body table th, .markdown-body table td {{
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            color: #c9d1d9 !important;
            padding: 12px;
        }}
        .markdown-body hr {{
            background-color: #30363d !important;
            height: 1px;
            border: none;
            margin: 32px 0;
        }}
        .markdown-body pre {{
            background-color: #161b22 !important;
            border: 1px solid #30363d;
            border-radius: 6px;
        }}
        .markdown-body blockquote {{
            border-left: 4px solid #388bfd !important;
            background-color: rgba(56, 139, 253, 0.08) !important;
            color: #c9d1d9 !important;
            padding: 12px 16px;
            border-radius: 0 6px 6px 0;
        }}
        .markdown-body img {{
            border-radius: 6px;
            border: 1px solid #30363d;
        }}
        .badge-bar {{
            display: flex;
            gap: 8px;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div id="content" class="markdown-body"></div>
    </div>
    <script>
        const rawMarkdown = {json.dumps(converted_md)};
        marked.setOptions({{
            gfm: true,
            breaks: false
        }});
        let renderedHtml = marked.parse(rawMarkdown);
        
        // Render LaTeX inline $...$
        renderedHtml = renderedHtml.replace(/\\$([^$]+)\\$/g, function(match, expr) {{
            try {{
                return katex.renderToString(expr, {{ displayMode: false, throwOnError: false }});
            }} catch(e) {{
                return match;
            }}
        }});
        
        document.getElementById('content').innerHTML = renderedHtml;
    </script>
</body>
</html>
"""

with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"✓ Standalone GitHub Mirror built successfully at: {out_html}")
print(f"  Size: {out_html.stat().st_size / (1024*1024):.2f} MB (All 9 images 100% Base64 embedded)")
