from PIL import Image

img_path = 'paper_overleaf/figures/test_construction_site_workers_1080p.jpg'
img = Image.open(img_path)
width, height = img.size

# Crop out top black bar (approx 40 pixels out of 720)
top_crop = 40
cropped_img = img.crop((0, top_crop, width, height))

# Overwrite original test_construction_site_workers_1080p.jpg
cropped_img.save(img_path, quality=95)
cropped_img.save('paper_overleaf/figures/test_construction_site_workers_1080p_cropped.jpg', quality=95)
print("Successfully cropped and overwritten Fig. 1 image!")
