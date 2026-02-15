import torch
import clip
from PIL import Image

# 1. Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
# vision transformer base model
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Load Image and txt
image = preprocess(Image.open("src/generated_image.png")
                   ).unsqueeze(0).to(device)
text = clip.tokenize(["a cat sitting on the sofa"]).to(device)

# 3. Get the embeddings
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# 4. Normalize the embeddings
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 5. Cosine Similarity
clip_score = (image_features @ text_features.T).item()
print(f"CLIP Score: {clip_score:.4f}")
