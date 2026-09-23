import os
import markdown

def convert(md_file, html_file, title):
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "nl2br", "sane_lists"])
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css">
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            padding: 20px;
            margin: 0;
        }}
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 1250px;
            margin: 0 auto;
            padding: 45px;
            background-color: #161b22;
            color: #c9d1d9;
            border-radius: 12px;
            border: 1px solid #30363d;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        }}
        .markdown-body table {{
            display: table;
            width: 100%;
            margin-bottom: 20px;
        }}
        .markdown-body th, .markdown-body td {{
            border: 1px solid #30363d;
            padding: 10px 14px;
        }}
        .markdown-body tr:nth-child(2n) {{
            background-color: #0d1117;
        }}
        .markdown-body blockquote {{
            color: #8b949e;
            border-left: .25em solid #1f6feb;
            background: #1f242c;
            padding: 12px 18px;
            border-radius: 6px;
            margin: 15px 0;
        }}
        .markdown-body pre {{
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
        }}
        .top-bar {{
            max-width: 1250px;
            margin: 0 auto 15px auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #21262d;
            padding: 12px 20px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }}
        .btn-print {{
            background: #238636;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="top-bar">
        <span>📄 <strong>{title}</strong></span>
        <button class="btn-print" onclick="window.print()">🖨️ Print / Save to PDF</button>
    </div>
    <article class="markdown-body">
        {body}
    </article>
</body>
</html>
"""
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    print("Exported HTML:", html_file)

if __name__ == "__main__":
    convert(
        r"c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\AUTHOR_REBUTTAL_LETTER_STANFORD_REVIEW.md",
        r"c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\AUTHOR_REBUTTAL_LETTER_STANFORD_REVIEW.html",
        "Author Rebuttal Letter - Response to Stanford Agentic Reviewer"
    )
