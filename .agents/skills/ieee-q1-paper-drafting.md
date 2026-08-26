# Skill Name: ieee-q1-paper-drafting
# Description: Skill giúp soạn thảo và phản biện bài báo khoa học AI theo chuẩn IEEE Transactions / Q1 Journals.

## Core Rules & Instructions for IEEE Q1 Writing:
1. **Academic Tone & Style**:
   - Sử dụng văn phong học thuật thụ động/chủ động chuẩn mực (Objective, Concise, Formal).
   - Tuyệt đối không dùng các từ cảm xúc hoặc khẳng định cảm tính (e.g., "groundbreaking", "revolutionary", "perfect"). Thay bằng con số thực nghiệm định lượng (e.g., "achieves a +3.2% mAP improvement").

2. **IEEE Structure Standard**:
   - **Abstract**: Exactly 150-250 words (Context -> Problem -> Proposed Method -> Key Result -> Impact).
   - **Introduction**: Paragraph 1 (Background & Significance), Paragraph 2 (Existing Limitations/Gap), Paragraph 3 (Core Contributions - bullet points), Paragraph 4 (Paper Organization).
   - **Related Work**: Taxonomical categorization of existing methods, highlighting explicit technical gaps.
   - **Proposed Methodology**: Mathematical formulation with LaTeX ($...$ and $$...$$), Architecture diagrams, RepConv/Attention module mechanics.
   - **Experiments & Results**: Datasets, Metrics, Implementation Details, Comparison with SOTA, Ablation Studies.
   - **Discussion & Limitations**: Honest analysis of failure cases & computational trade-offs.
   - **Conclusion**: Summary and future research directions.

3. **LaTeX & Table Formatting**:
   - Generate production-ready LaTeX code (`IEEEtran` document class compatibility).
   - Format comparison tables with best results in **bold** and second-best underlined.