import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import supervision as sv
import pycocotools.mask as mask_util
import py360convert as p360

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision.draw.color import ColorPalette, Color
from utils.supervision_utils import CUSTOM_COLOR_MAP


def setup_logging(log_level: str) -> None:
    """
    Configure the logging format and level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_grounding_model(model_id: str, device: torch.device):
    """
    Load the Grounding DINO model and its processor.
    """
    logging.info(f"Loading Grounding DINO model '{model_id}'...")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model_kwargs = {}
        if device.type == 'cuda':
            # Determine preferred dtype for CUDA
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                logging.info("CUDA device supports bfloat16. Configuring model to load in bfloat16.")
                model_kwargs['torch_dtype'] = torch.bfloat16
            else:
                logging.info("CUDA device does not support bfloat16 or support check unavailable. Configuring model to load in float16.")
                model_kwargs['torch_dtype'] = torch.float16  # Fallback for broader compatibility

        if model_kwargs:  # If any dtype kwargs were set (i.e., on CUDA)
            try:
                model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, **model_kwargs).to(device)
                logging.info(f"Successfully loaded model with {model_kwargs.get('torch_dtype')} on CUDA.")
            except Exception as e_dtype:
                logging.warning(f"Failed to load model with {model_kwargs.get('torch_dtype')} on CUDA: {e_dtype}. Falling back to default precision.")
                model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        else:  # For CPU or if no specific dtype was chosen
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    except Exception as e:
        logging.error(f"Failed to load Grounding DINO model: {e}")
        sys.exit(1)
    return processor, model


def load_sam2_model(config_path: str, checkpoint: str, device: torch.device) -> SAM2ImagePredictor:
    """
    Build and return the SAM2 image predictor.
    """
    logging.info("Loading SAM2 model...")
    try:
        sam_model = build_sam2(config_path, checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam_model)
    except Exception as e:
        logging.error(f"Failed to load SAM2 model: {e}")
        sys.exit(1)
    return predictor


def run_grounding_dino(
    processor,
    model,
    image: Image.Image,
    text: str,
    box_threshold: float,
    text_threshold: float,
    device: torch.device,
):
    """
    Run Grounding DINO on the image and return detected boxes, labels, and scores.
    """
    logging.info(f"Running Grounding DINO with prompt '{text}'...")
    text = text.lower().strip()
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    # Determine autocast dtype based on device capabilities
    selected_autocast_dtype = torch.bfloat16  # Default for CPU or if CUDA checks fail in an unexpected way
    if device.type == 'cuda':
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            selected_autocast_dtype = torch.bfloat16
            logging.info("Using bfloat16 for CUDA autocast.")
        else:
            selected_autocast_dtype = torch.float16  # Fallback for CUDA if bfloat16 not supported
            logging.info("Using float16 for CUDA autocast (bfloat16 not supported/detected).")
    elif device.type == 'cpu':
        logging.info("Using bfloat16 for CPU autocast.")
        selected_autocast_dtype = torch.bfloat16

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=selected_autocast_dtype):
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )
    res = results[0]
    boxes = res["boxes"].cpu().numpy()
    labels = [str(l) for l in res["labels"]]
    scores = res["scores"].cpu().numpy().tolist()
    logging.info(f"Detected {len(boxes)} objects.")
    return boxes, labels, scores


def run_sam2(
    predictor: SAM2ImagePredictor,
    image_np: np.ndarray,
    boxes: np.ndarray,
    multimask: bool,
):
    """
    Run SAM2 to generate segmentation masks for the given boxes.
    """
    logging.info("Running SAM2 segmentation...")
    predictor.set_image(image_np)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=multimask,
    )
    # squeeze if needed
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    logging.info(f"Generated {len(masks)} masks.")
    return masks, scores, logits


def visualize_and_save(
    processed_image_np: np.ndarray,
    original_img_path: str,
    output_dir: Path,
    boxes: np.ndarray,
    labels: list,
    masks: np.ndarray,
    scores: list,
    dump_json: bool,
):
    """
    Visualize detections and masks, save RGBA transparent-background image and optional JSON.
    The visualization is based on processed_image_np.
    JSON metadata refers to original_img_path for the image path,
    but width/height are from processed_image_np.
    """
    # Convert the processed numpy array (which is likely RGB) to an RGBA PIL Image, then to numpy array.
    # This ensures it has an alpha channel and matches mask dimensions.
    orig_pil = Image.fromarray(processed_image_np).convert("RGBA")
    orig_np = np.array(orig_pil)
    # Dimensions for JSON and for mask operations are from the processed image
    img_height, img_width = orig_np.shape[:2]

    # combine all masks into one
    if masks.shape[0] == 0:  # No masks detected or passed
        combined = np.zeros((img_height, img_width), dtype=bool)
    else:
        combined = np.any(masks, axis=0)
        
    # set alpha: opaque inside masks, transparent outside
    alpha = (combined * 255).astype(np.uint8)
    orig_np[..., 3] = alpha

    # ensure output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(orig_np).save(str(output_dir / "objects_only.png"))
    logging.info(f"Saved transparent-mask RGBA to {output_dir / 'objects_only.png'}")

    # Dump JSON
    if dump_json:
        def to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        annos = []
        for lab, box, mask, score in zip(labels, boxes, masks, scores):
            annos.append({
                "class_name": lab,
                "bbox": box.tolist(),
                "segmentation": to_rle(mask),
                "score": float(score),
            })
        result = {
            "image_path": original_img_path,  # Use original image path for metadata
            "box_format": "xyxy",
            "img_width": img_width,       # Width of the processed image
            "img_height": img_height,     # Height of the processed image
            "annotations": annos,
        }
        with open(output_dir / "results.json", "w") as f:
            import json

            json.dump(result, f, indent=4)
        logging.info(f"Dumped JSON results to {output_dir / 'results.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grounded SAM2 Pipeline")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text-prompt", default="building.")
    parser.add_argument("--img-path", required=True)
    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs")

    # thresholds
    parser.add_argument("--box-threshold", type=float, default=0.4)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}")

    processor, grounding_model = load_grounding_model(args.grounding_model, device)
    sam2_predictor = load_sam2_model(args.sam2_model_config, args.sam2_checkpoint, device)

    # Load image
    try:
        original_image_pil = Image.open(args.img_path).convert("RGB")
    except FileNotFoundError:
        logging.error(f"Image not found: {args.img_path}")
        sys.exit(1)

    # preprocess: convert cylindrical panorama to equirectangular only
    img_np = np.array(original_image_pil)
    img_np = p360.c2e(img_np, cube_format='horizon', h=512, w=1024)

    # ensure output dir
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # save the converted (equirectangular) image
    preprocessed_image_path = output_dir_path / "preprocessed_image.png"
    Image.fromarray(img_np).save(str(preprocessed_image_path))
    logging.info(f"Saved preprocessed image to {preprocessed_image_path}")

    # reload from disk for both detection and SAM
    image_pil_processed = Image.open(preprocessed_image_path).convert("RGB")
    img_np = np.array(image_pil_processed)

    # Run detection on the preprocessed image
    boxes, labels, scores = run_grounding_dino(
        processor,
        grounding_model,
        image_pil_processed,
        args.text_prompt,
        args.box_threshold,
        args.text_threshold,
        device,
    )

    if boxes.shape[0] > 0:
        # Create Detections object for visualization
        # Use np.arange to assign a unique class_id to each detection for color mapping
        detections = sv.Detections(
            xyxy=boxes,
            confidence=np.array(scores),
            class_id=np.arange(len(boxes)) # Ensures different colors if CUSTOM_COLOR_MAP is a palette
        )

        # Prepare formatted labels for the LabelAnnotator
        # 'labels' contains class names, 'scores' contains confidence values
        annotator_labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(labels, scores)
        ]

        # Annotate the preprocessed image with detected boxes
        # Use a copy of img_np for annotation to keep original img_np clean for SAM
        annotated_image_np = img_np.copy()
        
        # Initialize BoxAnnotator
        box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP) if isinstance(CUSTOM_COLOR_MAP, list) else CUSTOM_COLOR_MAP, 
            thickness=2
        )
        annotated_image_np = box_annotator.annotate(
            scene=annotated_image_np, 
            detections=detections
        )

        # Initialize LabelAnnotator to draw labels (class name + score)
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(CUSTOM_COLOR_MAP) if isinstance(CUSTOM_COLOR_MAP, list) else CUSTOM_COLOR_MAP, # Background color for labels
            text_color=sv.Color.BLACK, # Text color
            text_scale=0.5,
            text_thickness=1
        )
        annotated_image_np = label_annotator.annotate(
            scene=annotated_image_np, 
            detections=detections, 
            labels=annotator_labels
        )

        # Save annotated image
        detected_boxes_image_path = output_dir_path / "preprocessed_detected_boxes.png"
        Image.fromarray(annotated_image_np).save(str(detected_boxes_image_path))
        logging.info(f"Saved preprocessed image with detected boxes and labels to {detected_boxes_image_path}")
    else:
        logging.info("No boxes detected by Grounding DINO. Skipping saving of image with detected boxes.")


    if boxes.shape[0] == 0:
        logging.info("No objects detected by Grounding DINO. SAM2 segmentation will be skipped.")
        # Create empty masks array with shape (0, img_np.shape[0], img_np.shape[1])
        masks = np.empty((0, img_np.shape[0], img_np.shape[1]), dtype=bool)
        # 'scores' from Grounding DINO are already appropriate (likely empty or matching empty boxes)
    else:
        # Run segmentation only if boxes were detected, using the preprocessed img_np
        masks, _, _ = run_sam2(
            sam2_predictor,
            img_np, # Use preprocessed numpy image
            boxes,
            args.multimask,
        )

    visualize_and_save(
        img_np, # Pass the preprocessed image numpy array
        args.img_path, # Pass the original image path for JSON metadata
        output_dir_path,
        boxes,
        labels,
        masks,
        scores,
        not args.no_dump_json,
    )

if __name__ == "__main__":
    main()
