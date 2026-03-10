import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification

from .config import CHOICE_CLASSIFIER_OUTPUT_DIR, CONVNEXT_DIR, device


def choice_classify(grouped_masks, img_path):
    image_processor = AutoImageProcessor.from_pretrained(CONVNEXT_DIR)
    clf_model = ConvNextV2ForImageClassification.from_pretrained(CONVNEXT_DIR)
    clf_model.to(device)
    clf_model.eval()

    print("Classifier labels:", clf_model.config.id2label)

    if not grouped_masks:
        print("No masks to classify.")
    else:
        n = len(grouped_masks)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(6, 4 * n), squeeze=False)

        for idx, (gid, masked_img) in enumerate(grouped_masks):
            # Convert to PIL for the image processor
            pil_img = Image.fromarray(masked_img)

            inputs = image_processor(images=pil_img, return_tensors="pt").to(device)

            with torch.no_grad():
                logits = clf_model(**inputs).logits
                probs = logits.softmax(dim=-1)[0]
                pred_id = int(probs.argmax().item())
                pred_label = clf_model.config.id2label[pred_id]
                pred_prob = float(probs[pred_id])

            ax = axes[idx, 0]
            ax.imshow(masked_img)
            ax.set_title(f"Group {gid} → {pred_label} (p={pred_prob:.3f})")
            ax.axis("off")

            print(f"Group {gid}: predicted {pred_label} with p={pred_prob:.3f}")

        plt.tight_layout()
        path = os.path.join(CHOICE_CLASSIFIER_OUTPUT_DIR, os.path.basename(img_path))
        plt.savefig(path)

    ui_choice_groups = []  # list of dicts: {"gid":..., "img":..., "pred_prob":...}

    for gid, masked_img in grouped_masks:
        pil_img = Image.fromarray(masked_img)

        inputs = image_processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = clf_model(**inputs).logits
            probs = logits.softmax(dim=-1)[0]
            pred_id = int(probs.argmax().item())
            pred_label = clf_model.config.id2label[pred_id]
            pred_prob = float(probs[pred_id])

        if pred_label == "ui_choice":
            ui_choice_groups.append(
                {"gid": gid, "img": masked_img, "pred_prob": pred_prob}
            )

    print(f"Kept {len(ui_choice_groups)} / {len(grouped_masks)} groups as ui_choice.")
    return ui_choice_groups
