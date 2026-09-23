import os
import markdown
import re

md_path = r"c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\figures\scientific_exports\CHUYEN_SAU_KIEN_TRUC_VA_DOI_CHIEU_CODE.md"
html_path = r"c:\Users\ADMIN\Downloads\capstone AI\review1_genspark_package\figures\scientific_exports\CHUYEN_SAU_KIEN_TRUC_VA_DOI_CHIEU_CODE.html"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Convert markdown
html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "nl2br", "sane_lists"])

template = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chuyên Sâu Kiến Trúc & Đối Chiếu Mã Nguồn Rep-YOLO11s-P2 AFPN</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css">
    <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }
    };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            padding: 20px;
            margin: 0;
        }
        .markdown-body {
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
        }
        .markdown-body table {
            display: table;
            width: 100%;
            margin-bottom: 20px;
        }
        .markdown-body th, .markdown-body td {
            border: 1px solid #30363d;
            padding: 10px 14px;
        }
        .markdown-body tr:nth-child(2n) {
            background-color: #0d1117;
        }
        .markdown-body blockquote {
            color: #8b949e;
            border-left: .25em solid #1f6feb;
            background: #1f242c;
            padding: 12px 18px;
            border-radius: 6px;
            margin: 15px 0;
        }
        .markdown-body a {
            color: #58a6ff;
            text-decoration: none;
        }
        .markdown-body a:hover {
            text-decoration: underline;
        }
        .markdown-body pre {
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
        }
        .markdown-body code {
            background-color: rgba(110,118,129,0.2);
            color: #ff7b72;
            padding: 0.2em 0.4em;
            border-radius: 4px;
        }
        .markdown-body pre code {
            color: #c9d1d9;
            background-color: transparent;
            padding: 0;
        }
        .markdown-body h1, .markdown-body h2, .markdown-body h3 {
            border-bottom: 1px solid #21262d;
            padding-bottom: 0.3em;
        }
        .top-bar {
            max-width: 1250px;
            margin: 0 auto 15px auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #21262d;
            padding: 12px 20px;
            border-radius: 8px;
            border: 1px solid #30363d;
        }
        .btn-print {
            background: #238636;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-print:hover {
            background: #2ea043;
        }
    </style>
</head>
<body>
    <div class="top-bar">
        <span>📖 <strong>Tài liệu Chuyên Sâu: Đối Chiếu Mã Nguồn & 10 Sơ Đồ Khoa Học Rep-YOLO11s-P2 AFPN</strong></span>
        <button class="btn-print" onclick="window.print()">🖨️ In ra PDF / Xem chế độ in</button>
    </div>
    <article class="markdown-body">
        """ + html_body + """
    </article>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(template)

print("Created HTML successfully at:", html_path)
