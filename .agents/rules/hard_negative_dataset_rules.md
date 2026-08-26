# Hard-Negative Dataset Rules for Computer Vision Evaluation

When generating, collecting, or auditing `hard-negatives` datasets for evaluating False Positive Rates (FPR) in object detection models (e.g., YOLO, SHWD):

1. **Strict Target & Entity Absence (Zero-Person / Zero-Target Rule)**:
   - Hard negative images MUST NOT contain the target class (e.g. safety helmets) or entities naturally associated with the target class (e.g. workers/people).
   - Any detection of target objects in a hard negative dataset invalidates FPR auditing by turning FPs into TPs.

2. **Real-World Context Authenticity**:
   - Only use real-world photographic images from target operational contexts (e.g., CCTV feeds, construction sites, roadwork).
   - Strictly reject 3D vector renders, clipart, icon graphics, and studio product photography (e.g., kitchen plastic buckets).

3. **Targeted Distractor Selection**:
   - Focus on specific visual distractors with high geometric or color similarity to the target class (e.g., rounded yellow/orange buckets, traffic cones, hazard signs, lens glare in low light).

4. **Negative Search Queries & Automated Pre-Screening**:
   - Use negative search terms (e.g., `"empty construction site"`, `"no people"`, `"unmanned"`) during web scraping.
   - Run automated pre-screening using a standard object detector (e.g. YOLO COCO `person` detector) to automatically filter out images containing human figures or target objects prior to evaluation.
