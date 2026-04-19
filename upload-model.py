from huggingface_hub import HfApi

# Upload model to hugging face

api = HfApi()

api.upload_folder(
    folder_path="C:\\Users\\Daniel\\Documents\\COMP560\\beit_finetuned\\checkpoint-470",
    repo_id="D123aniel/560_ViT_Finetuned",
    repo_type="model",
    ignore_patterns=[".env"]
)