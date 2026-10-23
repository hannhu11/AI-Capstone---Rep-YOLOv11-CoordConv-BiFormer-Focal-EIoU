import requests

tokens = {
    "hannhu40022": "KGAT_e12378203b320b6808bb0a4f272aae4e",
    "mnu303": "KGAT_9d55f06fadc38903cb36f8842430875c"
}

headers = {"Authorization": "Bearer " + tokens["hannhu40022"]}
r = requests.get("https://www.kaggle.com/api/v1/kernels/status?userName=hannhu40022&kernelSlug=dataset-bo4-7class-unified-ppe-master", headers=headers)
print("status:", r.status_code, r.text)

r2 = requests.get("https://www.kaggle.com/api/v1/kernels/output?userName=hannhu40022&kernelSlug=dataset-bo4-7class-unified-ppe-master", headers=headers)
print("output:", r2.status_code, r2.text[:200])
