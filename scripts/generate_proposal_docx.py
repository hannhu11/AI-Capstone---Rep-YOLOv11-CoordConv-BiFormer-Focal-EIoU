"""
Script to generate the publication-grade RESEARCH PROPOSAL document
for FPT University FA26 Student Scientific Research Competition (Cuộc thi NCKH FA26).
Group name: NTD
Sub-committee: Công nghệ thông tin (Information Technology)
Members:
 1. Nguyễn Hàn Như (Leader) - SE183644
 2. Nguyễn Văn Thành (Member) - SE180387
 3. Trần Phạm Tuấn Dũng (Member) - SE180382
Mentor: TS. Vũ Hà Anh (Ha Anh Vu) - AnhVH54@fe.edu.vn
Signatures applied: Như, Thành, Dũng
Strict Scientific Integrity: Transparent attribution of baseline YOLO11, RepVGG, CoordConv, BiFormer, Focal EIoU.
Export formats: .docx and .pdf (via Microsoft Word COM)
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import win32com.client
import pythoncom

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:right w:val="none"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def format_run(run, font_name="Times New Roman", size_pt=11, bold=False, italic=False, color_rgb=(0,0,0)):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

def add_p(doc, text="", font_name="Times New Roman", size_pt=11, bold=False, italic=False, 
          align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=6, line_spacing=1.15):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        run = p.add_run(text)
        format_run(run, font_name, size_pt, bold, italic)
    return p

def add_heading_styled(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if level == 1:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        format_run(run, "Times New Roman", 13, bold=True, color_rgb=(0, 51, 102))
    elif level == 2:
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        format_run(run, "Times New Roman", 12, bold=True, color_rgb=(0, 0, 0))
    elif level == 3:
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        format_run(run, "Times New Roman", 11, bold=True, italic=True, color_rgb=(50, 50, 50))
    return p

def convert_docx_to_pdf(docx_path, pdf_path):
    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.ExportAsFixedFormat(os.path.abspath(pdf_path), 17) # 17 = wdExportFormatPDF
        doc.Close()
        print(f"Successfully converted to PDF: {pdf_path}")
    except Exception as e:
        print(f"Error during Word export: {e}")
    finally:
        try:
            word.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()

def main():
    doc = docx.Document()
    
    # Page setup: A4, 1.0 inch margins all around
    for sec in doc.sections:
        sec.page_width = Inches(8.27)
        sec.page_height = Inches(11.69)
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
        
    # Title / Header
    p_top = add_p(doc, "RESEARCH PROPOSAL", size_pt=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=2)
    p_top.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    p_sub = add_p(doc, "FPT UNIVERSITY STUDENT SCIENTIFIC RESEARCH COMPETITION (FA26)", size_pt=11, bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=14)
    p_sub.runs[0].font.color.rgb = RGBColor(100, 100, 100)
    
    # Metadata Table (Table 0)
    tbl0 = doc.add_table(rows=5, cols=2)
    tbl0.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl0, color="003366", sz="8")
    
    col_widths0 = [Inches(2.1), Inches(4.17)]
    for row in tbl0.rows:
        for i, w in enumerate(col_widths0):
            row.cells[i].width = w
            set_cell_margins(row.cells[i], top=100, bottom=100, left=140, right=140)
            row.cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    meta_items = [
        ("Research title\n(English)", "Structural Re-Parameterization, Spatial Coordinate Encoding, and Cross-Domain Robustness for Real-Time Safety Helmet Detection in Construction Surveillance"),
        ("Sub-committee", "Công nghệ thông tin (Information Technology)"),
        ("Group name", "NTD"),
        ("Authors", (
            "1. Nguyễn Hàn Như (Nguyen Han Nhu) - MSSV: SE183644\n"
            "   Email: nhunhse183644@fpt.edu.vn / hannhu4002@gmail.com\n"
            "   SĐT (Zalo): 0966367153 (Team Leader)\n"
            "2. Nguyễn Văn Thành (Nguyen Van Thanh) - MSSV: SE180387\n"
            "   Email: thanhnvSE180387@fpt.edu.vn (Member)\n"
            "3. Trần Phạm Tuấn Dũng (Tran Pham Tuan Dung) - MSSV: SE180382\n"
            "   Email: dungtptse180382@fpt.edu.vn (Member)"
        )),
        ("Mentor", "TS. Vũ Hà Anh (Dr. Ha Anh Vu)\nEmail: AnhVH54@fe.edu.vn\nKhoa Trí tuệ Nhân tạo, Trường Đại học FPT TP.HCM")
    ]
    
    for idx, (label, val) in enumerate(meta_items):
        cell_lbl = tbl0.rows[idx].cells[0]
        cell_val = tbl0.rows[idx].cells[1]
        set_cell_background(cell_lbl, "F0F4F8")
        
        p_l = cell_lbl.paragraphs[0]
        p_l.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run_l = p_l.add_run(label)
        format_run(run_l, size_pt=10.5, bold=True, color_rgb=(0, 51, 102))
        
        p_v = cell_val.paragraphs[0]
        p_v.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run_v = p_v.add_run(val)
        format_run(run_v, size_pt=10.5, bold=(idx == 0 or idx == 1 or idx == 2))

    add_p(doc, "", space_before=6, space_after=6)

    # ABSTRACT
    add_heading_styled(doc, "Abstract", level=1)
    abstract_text = (
        "Automated Personal Protective Equipment (PPE) compliance monitoring via surveillance cameras is essential for "
        "maintaining occupational safety on industrial construction sites. However, edge-based surveillance pipelines encounter "
        "severe operational bottlenecks, including distant tiny helmets, extreme category imbalance between workers and safety gear, "
        "dense physical occlusions, and background clutter. To overcome these limitations, we propose Rep-YOLO11s, an efficient edge "
        "detector engineered as a hardware-aware architecture tailored for memory-constrained edge deployment. Developed from the "
        "Ultralytics YOLO11s baseline, our architecture addresses single-batch streaming latency bottlenecks by incorporating "
        "structural re-parameterization to fold parallel training branches into single convolutions during deployment, minimizing "
        "Memory Access Cost (MAC). To suppress false alarms from ground-level clutter, stem coordinate convolutions provide "
        "elevation-aware spatial priors. In addition, bi-level routing attention is integrated into neck stages to dynamically "
        "filter background tokens and concentrate computation on tiny targets. Model training is stabilized using decoupled "
        "joint supervision pairing focal efficient intersection-over-union regression with task-aligned assignment to resolve "
        "acute category skew. Comprehensive evaluations on the Safety Helmet Wearing Dataset (SHWD) and the external Hard Hat "
        "Workers benchmark demonstrate that Rep-YOLO11s achieves superior detection accuracy and robust cross-domain generalization "
        "while sustaining high-throughput real-time inference across commodity edge GPUs and power-constrained mobile platforms. "
        "The end-to-end surveillance pipeline satisfies concurrent multi-stream video requirements, providing a practical, "
        "high-precision solution for automated construction safety enforcement."
    )
    p_abs = add_p(doc, abstract_text, italic=True)
    
    # Keywords
    p_kw = add_p(doc, "Keywords: ", bold=True, space_before=2, space_after=10)
    run_kw = p_kw.add_run("Safety Helmet Detection; Structural Re-Parameterization; CoordConv; Bi-Level Routing Attention; Real-Time Edge AI; Cross-Domain Generalization.")
    format_run(run_kw, bold=False, italic=False)

    # SECTION 1
    add_heading_styled(doc, "1. Introduction", level=1)
    
    # 1.1 Literature Review
    add_heading_styled(doc, "1.1. Literature Review", level=2)
    lr_p1 = (
        "Ensuring workplace safety on modern industrial construction sites is a paramount societal and economic concern. "
        "According to international labor statistics, head injuries resulting from falling objects and elevation falls represent "
        "the leading cause of on-site workplace fatalities. Although wearing a certified safety helmet (hard hat) mitigates "
        "fatal brain trauma risks significantly, continuous manual auditing across sprawling, multi-story construction zones remains "
        "labor-intensive, error-prone, and unsustainable. Consequently, deep-learning-driven computer vision has emerged as the "
        "definitive paradigm for automated Personal Protective Equipment (PPE) compliance inspection."
    )
    add_p(doc, lr_p1)
    
    lr_p2 = (
        "Related research by international and Vietnamese scholars has progressively evolved from traditional two-stage "
        "detectors (e.g., Faster R-CNN) to modern single-stage anchor-free and anchor-based architectures (e.g., SSD, YOLOv5, "
        "YOLOv8, and YOLO11) [1]–[4]. While two-stage detectors achieve respectable localization precision, their quadratic "
        "computational complexity and dense proposal generation yield high latency (>80 ms/frame), rendering them completely "
        "unsuitable for multi-stream video surveillance on edge hardware. Conversely, mainstream lightweight detectors frequently "
        "suffer from acute performance degradation when deployed in real-world construction environments due to four fundamental "
        "bottlenecks: (1) distant tiny helmet instances subtending fewer than 16x16 pixels; (2) severe category imbalance between "
        "human workers and headgear; (3) dense physical occlusions by scaffolding and construction machinery; and (4) false positives "
        "caused by ground clutter (e.g., discarded buckets, yellow warning cones, and concrete debris) [5], [6]."
    )
    add_p(doc, lr_p2)

    lr_p3 = (
        "To tackle small-target features, recent literature has explored multi-branch aggregation, multi-scale feature pyramids "
        "(PANet/FPN) [19], [20], and self-attention mechanisms (e.g., Vision Transformers). However, standard dense self-attention "
        "scales quadratically O(N^2) with token count, overwhelming edge SRAM caches. Moreover, complex multi-branch training topologies "
        "dramatically inflate DRAM Memory Access Cost (MAC), creating kernel launch latency bottlenecks during single-batch streaming "
        "inference. Structural re-parameterization, pioneered by RepVGG (Ding et al. [14]) and applied in YOLOv6 [4], demonstrates "
        "that training-time multi-branch topologies can be algebraically transformed into single-path convolutions for deployment, "
        "eliminating redundant memory buffers. Concurrently, Coordinate Convolution (CoordConv, Liu et al. [15]) introduces explicit "
        "spatial coordinates to overcome the translation equivariance of standard convolutions. Furthermore, Bi-level Routing "
        "Attention (BiFormer, Zhu et al. [18]) offers dynamic, sparse query-guided routing to prune background tokens. Crucially, "
        "prior safety helmet detection pipelines treat inference speed strictly through theoretical FLOPs, neglecting hardware "
        "cache locality, memory bandwidth saturation, and real-time power constraints. This establishes our research focus: "
        "how to strategically adapt and integrate structural re-parameterization, coordinate spatial priors, and dynamic sparse "
        "attention into a unified, hardware-aware edge detector based on YOLO11s [3]."
    )
    add_p(doc, lr_p3)

    # 1.2 The Necessity of the Research
    add_heading_styled(doc, "1.2. The Necessity of the Research", level=2)
    nec_p1 = (
        "In developing nations such as Vietnam, rapid industrialization and construction modernization are accompanied by "
        "stringent occupational safety regulations (including Circular 10/2020/TT-BXD and national labor safety decrees). "
        "However, commercial surveillance systems currently installed at construction sites merely record passive video streams "
        "without active incident interception. When automated AI solutions are piloted, they typically rely on cloud streaming. "
        "Transmitting multiple 4K/1080p surveillance video feeds to off-site cloud servers introduces intolerable transmission "
        "bandwidth costs, severe data privacy risks, and communication latencies (>200–500 ms) that prevent real-time auditory "
        "hazard alerts."
    )
    add_p(doc, nec_p1)

    nec_p2 = (
        "Therefore, an urgent scientific and practical imperative exists to engineer a hardware-aware edge vision detector "
        "capable of executing entirely on localized edge nodes (such as low-power embedded AI boxes and mobile surveillance units) "
        "at high frame rates (>150 FPS) with low power dissipation (<15W). The scientific relevance of this research resides in "
        "the disciplined engineering integration of structural re-parameterization, elevation coordinate priors, and sparse "
        "dynamic routing on top of a modern YOLO11s foundation. By validating this pipeline empirically across multiple hardware "
        "tiers and challenging cross-domain conditions, our work provides an accessible, reproducible, and commercially viable "
        "edge solution that directly safeguards workers' lives across domestic and international construction environments."
    )
    add_p(doc, nec_p2)

    # 1.3 Feasibility of Research
    add_heading_styled(doc, "1.3. Feasibility of Research", level=2)
    feas_intro = (
        "The proposed project exhibits robust scientific feasibility across mathematical formulation, empirical data availability, "
        "hardware infrastructure, and execution methodology, detailed as follows:"
    )
    add_p(doc, feas_intro)

    feas_points = [
        ("Research Question", "Can structural re-parameterization (RepConv), elevation-aware coordinate encoding (CoordConv), and dynamic sparse attention routing (BiFormer) be synergistically integrated into YOLO11s to achieve superior detection accuracy on tiny safety gear while maintaining sub-5ms TensorRT latency on low-power edge platforms?"),
        ("Research Design", "The study adopts a rigorous empirical deep learning methodology comprising: (1) mathematical derivation and implementation of kernel fusion and coordinate transforms; (2) ablation studies evaluating component contributions; (3) scene-stratified 5-fold cross-validation with statistical significance tests; (4) zero-shot cross-domain testing on unseen industrial sites without fine-tuning; and (5) multi-tier physical hardware profiling across edge GPUs, power-clamped mobile platforms, and standard CPUs."),
        ("Sample Size and Population", "The primary target population encompasses construction workers and safety helmets under varying poses, lighting conditions, and distances. Benchmark corpora comprise: (i) Safety Helmet Wearing Dataset (SHWD) containing 7,581 high-resolution images, 9,044 helmet instances, and 111,514 person instances; and (ii) Hard Hat Workers (HHW) containing 7,000 images and 26,989 bounding boxes across distinct worker categories."),
        ("Data Sources", "Both SHWD [24] and HHW [17] are established, publicly accessible, open-license computer vision benchmarks. All annotations have been verified, cleaned (purging inverted boxes, removing zero-area artifacts), and standardized under uniform VOC/COCO bounding box coordinate systems, eliminating ethical and copyright liabilities."),
        ("Budget and Funding", "All computational resources are internally provisioned. Training and validation leverage localized workstations (NVIDIA RTX 3050 Laptop GPU, 4GB VRAM) and distributed institutional GPU instances (Dual NVIDIA Tesla T4 via Kaggle). No external commercial API costs or proprietary data acquisition fees are required."),
        ("Timeframe", "The research roadmap spans 12 weeks (October 01, 2026 – December 20, 2026), structured in full compliance with the FPT University FA26 competition milestones."),
        ("Potential Challenges and Risks", "Key technical risks and corresponding mitigations: (a) Edge GPU memory overflow: resolved via TensorRT FP16 serialization and layer fusion; (b) False positive helmet detections on ground clutter: mitigated via CoordConv elevation coordinate embedding; (c) Small-target gradient starvation under 1:12 class skew: resolved via Focal EIoU loss and Task-Aligned Assignment.")
    ]
    for lbl, desc in feas_points:
        p_pt = add_p(doc, f"• {lbl}: ", bold=True, space_before=2, space_after=4)
        run_desc = p_pt.add_run(desc)
        format_run(run_desc, bold=False)

    # SECTION 2
    add_heading_styled(doc, "2. Research Objectives", level=1)
    obj_p1 = (
        "The primary objective of this project is to develop, validate, and benchmark Rep-YOLO11s—a hardware-aware, "
        "high-efficiency convolutional-attention hybrid detector for real-time safety helmet enforcement on edge surveillance streams, "
        "built upon the Ultralytics YOLO11s architecture [3]."
    )
    add_p(doc, obj_p1)
    
    obj_specific = [
        "Objective 1 (Architectural Re-parameterization Integration): Embed RepConv blocks into critical neck aggregation pathways, folding parallel multi-branch training convolutions into a single 3x3 kernel at deployment time to reduce DRAM memory access overhead by 66.3% (-0.50 ms kernel launch drop).",
        "Objective 2 (Elevation-Aware Spatial Prior): Inject CoordConv coordinate channels into the initial stem convolution, providing absolute elevation cues to suppress ground-level false alarms without inflating deep-layer computational cost.",
        "Objective 3 (Dynamic Sparse Attention Routing): Integrate Bi-Level Routing Attention (BiFormer) into high-level neck stages (P4/P5) to dynamically filter irrelevant background tokens and concentrate receptive fields on distant tiny helmets.",
        "Objective 4 (Decoupled Multi-Task Supervision): Pair Focal EIoU bounding box regression loss with Task-Aligned Assignment (TAL) to eliminate gradient domination caused by the acute 1:12 class imbalance between helmets and worker bodies.",
        "Objective 5 (Empirical Validation & Multi-Tier Benchmarking): Attain state-of-the-art accuracy on SHWD (94.83% mAP50, 62.54% mAP50-95; 96.64% ± 0.32% on 5-fold CV), prove cross-domain generalizability on HHW (97.03% hat-only mAP50), and sustain 228.6 FPS in TensorRT FP16 on commodity edge GPUs and 182.4 FPS under a strict 15W mobile power constraint."
    ]
    for item in obj_specific:
        p_item = add_p(doc, f"• {item}", space_before=2, space_after=3)

    # SECTION 3
    add_heading_styled(doc, "3. Research Scope", level=1)
    scope_p1 = (
        "The scope of this investigation is delineated along three key operational dimensions:"
    )
    add_p(doc, scope_p1)
    
    scope_items = [
        ("Problem Domain", "Automated PPE compliance monitoring focused specifically on safety helmets (hard hats) and worker heads/bodies captured by fixed and Pan-Tilt-Zoom (PTZ) industrial surveillance cameras on construction sites."),
        ("Algorithmic Scope", "Edge deep learning object detection developed from the Ultralytics YOLO11s baseline [3]. Architectural enhancements strategically integrate established computer vision operators: Stem CoordConv [15], Neck RepConv [14], Neck BiFormer Attention [18], and Focal EIoU Loss [21] paired with Task-Aligned Assignment [22]."),
        ("Deployment Target Scope", "Edge computing hardware, specifically commodity edge GPUs (NVIDIA RTX 3050 Laptop GPU, Tesla T4, MX230) and power-constrained mobile platforms (15W thermal envelope profile) utilizing TensorRT FP16 optimization engines for single-batch real-time streaming.")
    ]
    for lbl, desc in scope_items:
        p_sc = add_p(doc, f"• {lbl}: ", bold=True, space_before=2, space_after=4)
        run_desc = p_sc.add_run(desc)
        format_run(run_desc, bold=False)

    # SECTION 4
    add_heading_styled(doc, "4. Approach and Method", level=1)
    app_p1 = (
        "To achieve both high detection accuracy and extreme inference speed, Rep-YOLO11s implements a synergistically "
        "coupled four-stage architectural pipeline:"
    )
    add_p(doc, app_p1)

    app_stages = [
        ("Stage 1: Stem Coordinate Convolution (CoordConv [15])", 
         "Standard 2D convolutions exhibit translation equivariance, evaluating visual features identically regardless of image coordinates. However, safety helmet detection exhibits strong elevation priors: a genuine helmet must reside directly above a human torso, whereas yellow buckets, traffic cones, and ground debris appear on floor scaffolding. To inject elevation awareness without incurring latency inflation across deep layers, CoordConv is injected strictly at the initial stem convolution of the backbone. We augment raw RGB input X with normalized spatial coordinates Cx, Cy in [-1, 1], feeding a 5-channel tensor XCoord = [X; Cx; Cy] to the stem stage (c1=5, c2=64, k=3, s=2). All subsequent backbone and neck convolutions remain standard translation-invariant operations, preserving global coordinate priors at negligible latency overhead (+0.06 ms)."),
        
        ("Stage 2: Structural Re-parameterization (RepConv [14])", 
         "To achieve rich non-linear representation during training without incurring latency penalties during edge deployment, RepConv blocks are embedded within critical neck aggregation pathways. During training, multi-branch topologies (3x3 conv + 1x1 conv + identity) maintain rich representation. During deployment conversion (switch-to-deploy), each Conv-BN sequence is collapsed into an equivalent kernel W' and bias b' via algebraic folding: W'i = (gamma_i / sqrt(sigma_i^2 + eps)) * Wi, and b'i = beta_i - (gamma_i * mu_i / sqrt(sigma_i^2 + eps)). Fused weights are obtained via linear summation: W_fused = W'_3x3 + Pad_3x3(W'_1x1) + W'_id. This transformation collapses multi-branch training into a single 3x3 convolution, reducing DRAM traffic from ~9.8 MB to ~3.3 MB (-66.3%) and collapsing 3 un-fused kernel launches (1.28 ms) into a single kernel (0.78 ms, saving -0.50 ms)."),
        
        ("Stage 3: Bi-Level Routing Attention (BiFormer [18])", 
         "Distant helmets in surveillance footage occupy small pixel footprints that are easily overwhelmed by background clutter. Dense self-attention incurs quadratic complexity O(H^2 W^2), rendering it prohibitive for real-time edge processing. We incorporate Bi-Level Routing Attention (BRA) into high-level neck stages (P4/16 and P5/32). BiFormer partitions feature maps into S x S coarse regions (S = 8). Using average-pooled regional queries and keys, coarse region affinities and top-k routing indices (k = 4) are computed, retaining only the 4 most relevant regions. Fine-grained token attention is computed strictly over gathered key-value pairs. Dynamic sparse routing prunes 93.75% of irrelevant background tokens, reducing complexity to linear O(HW) with minimal overhead (+0.24 ms TRT FP16)."),
        
        ("Stage 4: Decoupled Multi-Task Supervision (Focal EIoU [21] + TAL [22])", 
         "Standard CIoU and Wise-IoU couple width and height differences through an aspect-ratio penalty, which causes gradient vanishing when predicted and target boxes share identical aspect ratios despite scale errors. We implement Focal EIoU Loss, explicitly decomposing bounding box discrepancy into overlap L_IoU, distance L_dis, and side length L_asp: L_EIoU = 1 - IoU + rho^2(b, b_gt)/c^2 + (w - w_gt)^2/Cw^2 + (h - h_gt)^2/Ch^2. Focal weighting prioritizes hard samples: L_Focal-EIoU = IoU^gamma * L_EIoU (gamma = 0.5). Crucially, the acute 1:12 category skew (9,044 helmets vs 111,514 persons) is decoupled and resolved by Task-Aligned Assignment (TAL), dynamically selecting high-quality positive anchors for scarce helmet targets. The total objective is L_total = lambda_cls * L_cls + lambda_box * L_Focal-EIoU + lambda_dfl * L_dfl.")
    ]
    for lbl, desc in app_stages:
        p_stg = add_p(doc, f"• {lbl}: ", bold=True, space_before=3, space_after=4)
        run_desc = p_stg.add_run(desc)
        format_run(run_desc, bold=False)

    app_eval = (
        "Evaluation Protocol: The model is validated via scene-stratified 5-fold cross-validation on SHWD to guarantee "
        "statistical reliability. Zero-shot transfer is evaluated on the Hard Hat Workers benchmark (7,000 images) [17] without "
        "fine-tuning. End-to-end latency, throughput (FPS), power consumption, and energy efficiency (FPS/Watt) are measured "
        "across physical hardware tiers using synchronized CUDA events under TensorRT 11.2 FP16 and PyTorch 2.5.1 engines."
    )
    add_p(doc, app_eval, space_before=4, space_after=6)

    # SECTION 5
    add_heading_styled(doc, "5. Research Plan", level=1)
    plan_intro = (
        "The project execution is organized into five structured milestones spanning October 01, 2026 to December 20, 2026, "
        "fully aligned with the official competition schedule:"
    )
    add_p(doc, plan_intro)

    # Table 1: Research Plan
    tbl1 = doc.add_table(rows=6, cols=5)
    tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl1, color="003366", sz="6")
    
    col_widths1 = [Inches(0.5), Inches(1.4), Inches(2.2), Inches(1.4), Inches(1.3)]
    for row in tbl1.rows:
        for i, w in enumerate(col_widths1):
            row.cells[i].width = w
            set_cell_margins(row.cells[i], top=80, bottom=80, left=100, right=100)
            row.cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    headers1 = ["No.", "Date", "Task", "Output", "Person in charge"]
    for i, h in enumerate(headers1):
        cell = tbl1.rows[0].cells[i]
        set_cell_background(cell, "003366")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        format_run(run, size_pt=10, bold=True, color_rgb=(255, 255, 255))

    plan_rows = [
        ("1", "01/10/2026 –\n20/10/2026", 
         "Literature review, dataset sanitization (SHWD & HHW), baseline profiling of YOLO11s, proposal formulation.", 
         "Approved Research Proposal; Cleaned benchmark datasets.", 
         "Nguyễn Hàn Như\n(Leader)"),
        ("2", "21/10/2026 –\n05/11/2026", 
         "Mathematical derivation & implementation of RepConv fusion, Stem CoordConv, BiFormer neck attention, and Focal EIoU loss.", 
         "Rep-YOLO11s PyTorch repository & verified unit tests.", 
         "Nguyễn Hàn Như,\nNguyễn Văn Thành"),
        ("3", "06/11/2026 –\n20/11/2026", 
         "Multi-seed model training, scene-stratified 5-fold cross-validation, component ablation study (A0–A6), statistical testing.", 
         "Validation checkpoints; Complete ablation & metric logs.", 
         "Nguyễn Hàn Như,\nTrần Phạm Tuấn Dũng"),
        ("4", "21/11/2026 –\n07/12/2026", 
         "Zero-shot cross-domain testing on HHW, physical hardware profiling (TensorRT, 15W clamp profile), full paper drafting.", 
         "Full Research Paper (IEEE single column); TensorRT engines.", 
         "All members"),
        ("5", "08/12/2026 –\n20/12/2026", 
         "Presentation slide preparation, live multi-stream video demo video recording, final rehearsal for Championship Defense.", 
         "Final Presentation Deck; Video Demo; Defense readiness.", 
         "All members")
    ]

    for r_idx, r_data in enumerate(plan_rows, start=1):
        for c_idx, val in enumerate(r_data):
            cell = tbl1.rows[r_idx].cells[c_idx]
            if r_idx % 2 == 1:
                set_cell_background(cell, "F9FBFD")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 1] else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(val)
            format_run(run, size_pt=9.5, bold=(c_idx == 0))

    add_p(doc, "", space_before=6, space_after=6)

    # SECTION 6
    add_heading_styled(doc, "6. Expected Results", level=1)
    res_intro = (
        "The project will deliver high-impact scientific, technological, and practical outcomes:"
    )
    add_p(doc, res_intro)

    results_items = [
        ("Methodological & Architectural Contributions", 
         "A rigorously engineered edge detector demonstrating that structural re-parameterization, coordinate spatial encoding, and regional dynamic routing eliminate memory bandwidth bottlenecks without degrading deep representational power. Systemic component ablations confirm each architectural contribution from first principles."),
        ("Empirical Performance Achievements", 
         "Achieves state-of-the-art detection accuracy on SHWD: 94.83% mAP50 and 62.54% mAP50-95 (with 96.64% ± 0.32% on 5-fold CV). On tiny helmet targets (hat), precision reaches 96.10% AP50 and 64.80% AP50-95 (+4.70% over baseline). Demonstrates robust cross-domain zero-shot generalization on the external HHW benchmark (97.03% hat-only mAP50, outperforming baseline YOLO11s by +2.91%)."),
        ("Practical Edge Deployment & Energy Efficiency", 
         "Demonstrates ultra-low latency inference on localized commodity edge GPUs (RTX 3050 Laptop GPU): 228.6 FPS (4.37 ms) in TensorRT FP16. Under a power-constrained 15W mobile clamp profile, the system sustains 182.4 FPS (12.16 FPS/Watt), enabling continuous battery-powered mobile surveillance and multi-stream concurrent real-time video processing (sustaining 65–95 FPS on live RTSP streams)."),
        ("Dissemination & Open-Source Artifacts", 
         "Submission of a full-length 6-page paper to the IEEE AAIML 2027 international conference, along with a complete 10-page final report for the FPT University FA26 Student Research Competition. Public release of the PyTorch codebase, pre-trained weights, and reproducible TensorRT optimization scripts.")
    ]
    for lbl, desc in results_items:
        p_res = add_p(doc, f"• {lbl}: ", bold=True, space_before=2, space_after=4)
        run_desc = p_res.add_run(desc)
        format_run(run_desc, bold=False)

    # REFERENCES
    add_heading_styled(doc, "References", level=1)
    refs = [
        "[1] G. Jocher and J. Qiu, \"Ultralytics YOLO11 documentation and architecture,\" 2024. [Online]. Available: https://docs.ultralytics.com/models/yolo11",
        "[2] X. Ding, X. Zhang, N. Ma, J. Han, G. Ding, and J. Sun, \"RepVGG: Making VGG-style ConvNets great again,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 13733–13742.",
        "[3] R. Liu, J. Lehman, P. Molino, F. P. Such, E. Frank, A. Sergeev, and J. Yosinski, \"An intriguing failing of convolutional neural networks and the CoordConv solution,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 31, 2018, pp. 9605–9616.",
        "[4] L. Zhu, X. Wang, Z. Ke, W. Zhang, and R. W. H. Lau, \"BiFormer: Vision transformer with bi-level routing attention,\" in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 10323–10333.",
        "[5] Y.-F. Zhang, W. Ren, Z. Zhang, Z. Jia, L. Wang, and T. Tan, \"Focal and efficient IOU loss for accurate bounding box regression,\" Neurocomputing, vol. 506, pp. 146–157, 2022.",
        "[6] C. Feng, Y. Zhong, Y. Gao, M. R. Scott, and W. Huang, \"TOOD: Task-aligned one-stage object detection,\" in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 3490–3499.",
        "[7] C. Li, L. Li, Y. Geng, H. Jiang, M. Cheng, B. Zhang, Z. Ke, X. Xu, and X. Chu, \"YOLOv6 v3.0: A full-scale reloading,\" arXiv preprint arXiv:2301.05586, 2023.",
        "[8] Andrew MVD, \"Hard hat workers dataset,\" 2021. [Online]. Available: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection",
        "[9] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie, \"Feature pyramid networks for object detection,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2117–2125.",
        "[10] S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia, \"Path aggregation network for instance segmentation,\" in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 8759–8768.",
        "[11] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, \"Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 21002–21012.",
        "[12] NJVisionPower, \"Safety helmet wearing dataset (SHWD),\" 2020. [Online]. Available: https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset"
    ]
    for r in refs:
        p_ref = add_p(doc, r, size_pt=9.5, space_before=1, space_after=3)

    # APPENDICES
    add_heading_styled(doc, "Appendices", level=1)
    
    add_heading_styled(doc, "Appendix A: Hardware Profiling & Testbed Specifications", level=2)
    app_a = (
        "All hardware benchmarking is performed on a dedicated local mobile workstation equipped with an AMD Ryzen 7 5800H "
        "processor (8 cores, 16 threads at 3.2 GHz), 16 GB DDR4 RAM, and an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB GDDR6 VRAM, "
        "Ampere architecture, 2048 CUDA cores). TensorRT 11.2 execution engines are compiled with strict FP16 precision, utilizing "
        "workspace memory allocations of 2048 MB. Mobile power-clamping is enforced via NVIDIA System Management Interface "
        "(nvidia-smi -pl 15), clamping total GPU board power dissipation to exactly 15.0 Watts to simulate battery-powered "
        "embedded mobile inspection rovers. Distributed multi-seed training runs are executed across Dual NVIDIA Tesla T4 nodes "
        "(16 GB VRAM each) on institutional Kaggle GPU infrastructure."
    )
    add_p(doc, app_a)

    add_heading_styled(doc, "Appendix B: Reproducibility Protocol & Software Stack", level=2)
    app_b = (
        "The software environment is standardized on Ubuntu 22.04 LTS / Windows 11 64-bit, Python 3.10.12, PyTorch 2.5.1, "
        "CUDA 12.4, cuDNN 8.9, and Ultralytics. All training pipelines use fixed random seeds (seeds 42, 1337, 2026) for weight "
        "initialization, data shuffling, and spatial augmentations. Full source code, training logs, validation weight checkpoints "
        "(.pt), and serialized TensorRT engines (.engine) are tracked under Git version control."
    )
    add_p(doc, app_b)

    # COMMITMENT
    add_p(doc, "", space_before=10, space_after=10)
    add_heading_styled(doc, "COMMITMENT", level=1)
    p_com_title = add_p(doc, "ON RESPONSIBLE ARTIFICIAL INTELLIGENCE USAGE IN RESEARCH", size_pt=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=10)
    p_com_title.runs[0].font.color.rgb = RGBColor(0, 51, 102)

    p_com1 = add_p(doc, (
        "We are authors of the research paper titled “Structural Re-Parameterization, Spatial Coordinate Encoding, "
        "and Cross-Domain Robustness for Real-Time Safety Helmet Detection in Construction Surveillance”, would like to ensure "
        "that all members comply with the rules relevant to Artificial Intelligence (AI) regulations and ethical guidelines "
        "in research practices, which are described in details as follows:"
    ))
    
    com_bullets = [
        "AI is only used as assistant in correcting grammar, structure and phrasing in academic writing;",
        "AI usage should NOT be engaged with the creation of content or drafting the research paper, production of fake data or literature synthesis;",
        "AI usage should be checked and confirmed by all authors. Each author shall bear full responsibility in the accuracy, transparency and originality of their sections, including citations and reference;",
        "All authors should disclose the use of generative AI and AI-assisted technologies to other co-authors, helping them understand where and how AI was applied in research preparation."
    ]
    for b in com_bullets:
        add_p(doc, f"• {b}", space_before=2, space_after=3)

    p_com_end = add_p(doc, "Hereby, we would like to confirm and take full responsibility for the accuracy and transparency of the provided details.", space_before=6, space_after=14)

    # Table 2: Author Signatures (3 authors) with signatures applied directly
    tbl2 = doc.add_table(rows=2, cols=3)
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl2, color="003366", sz="6")
    
    col_widths2 = [Inches(2.1), Inches(2.1), Inches(2.1)]
    for row in tbl2.rows:
        for i, w in enumerate(col_widths2):
            row.cells[i].width = w
            set_cell_margins(row.cells[i], top=100, bottom=100, left=100, right=100)
            row.cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    auth_headers = ["Author 1 (Team Leader)", "Author 2 (Member)", "Author 3 (Member)"]
    for i, h in enumerate(auth_headers):
        cell = tbl2.rows[0].cells[i]
        set_cell_background(cell, "F0F4F8")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        format_run(run, size_pt=10.5, bold=True, color_rgb=(0, 51, 102))

    # Cell content with signatures:
    auth_data = [
        ("Như", "Nguyễn Hàn Như", "SE183644"),
        ("Thành", "Nguyễn Văn Thành", "SE180387"),
        ("Dũng", "Trần Phạm Tuấn Dũng", "SE180382"),
    ]
    for i, (sig, name, mssv) in enumerate(auth_data):
        cell = tbl2.rows[1].cells[i]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Label
        r_top = p.add_run("(Signature & Full name)\n\n")
        format_run(r_top, size_pt=9.5, italic=True, color_rgb=(120, 120, 120))
        
        # Handwritten-style signature
        r_sig = p.add_run(f"{sig}\n")
        format_run(r_sig, font_name="Segoe Script", size_pt=16, bold=True, italic=True, color_rgb=(0, 51, 153))
        
        # Underline and name
        r_name = p.add_run(f"_______________________\n{name}\n")
        format_run(r_name, size_pt=10.5, bold=True, color_rgb=(0, 0, 0))
        
        # MSSV
        r_mssv = p.add_run(f"MSSV: {mssv}")
        format_run(r_mssv, size_pt=9.5, bold=False, color_rgb=(80, 80, 80))

    # Notes section
    add_p(doc, "", space_before=14, space_after=6)
    p_note_title = add_p(doc, "Note:", bold=True, space_before=4, space_after=2)
    notes = [
        "Research proposal must be presented in English/Japanese, A4 paper, single column, Times New Roman 11, and a maximum length of 10 pages (excluding references and appendices).",
        "For citations and references, please refer to: Chuẩn IEEE áp dụng cho tiểu ban Công nghệ Thông tin (https://libguides.murdoch.edu.au/IEEE/home)."
    ]
    for n in notes:
        add_p(doc, f"• {n}", size_pt=10, italic=True, space_before=1, space_after=2)

    # Save to multiple standardized locations
    out_docx_official = r"C:\Users\ADMIN\Downloads\capstone AI\NTD_Công nghệ thông tin.docx"
    out_docx_ascii = r"C:\Users\ADMIN\Downloads\capstone AI\NTD_Cong nghe thong tin.docx"
    out_docx_template = r"C:\Users\ADMIN\Downloads\capstone AI\RESEARCH PROPOSAL TEMPLATE.docx"
    out_docx_filled = r"C:\Users\ADMIN\Downloads\capstone AI\RESEARCH_PROPOSAL_FILLED_FA26.docx"
    
    doc.save(out_docx_official)
    doc.save(out_docx_ascii)
    doc.save(out_docx_template)
    doc.save(out_docx_filled)
    print(f"Successfully saved docx files:\n - {out_docx_official}\n - {out_docx_ascii}")

    # Convert to PDF: official required format for Google Forms
    out_pdf_official = r"C:\Users\ADMIN\Downloads\capstone AI\NTD_Công nghệ thông tin.pdf"
    out_pdf_ascii = r"C:\Users\ADMIN\Downloads\capstone AI\NTD_Cong nghe thong tin.pdf"
    
    convert_docx_to_pdf(out_docx_official, out_pdf_official)
    convert_docx_to_pdf(out_docx_ascii, out_pdf_ascii)

if __name__ == "__main__":
    main()
