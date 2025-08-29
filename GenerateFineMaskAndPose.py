import argparse
import os
import json
import logging
import sys

import cv2
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from modules.cloth_masker import AutoMasker
from Util.Logging import set_up_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument("--catvton_ckpt_path",type=str,default="zhengchong/CatVTON",help="The Path to the checkpoint of CatVTON")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the paired file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input directory")
    parser.add_argument("--input_cloth", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_mask", type=str, required=True, help="Path to output directory")
    parser.add_argument("--output_pose", type=str, required=True, help="Path to output directory")
    parser.add_argument("--debug", action='store_true', help="Debugging logging activated")

    return parser.parse_args()

def main():
    args = parse_args()

    DEBUG = args.debug
    level = "debug" if DEBUG else "info"
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    set_up_logging(console_log_output="stdout", console_log_level=level,
                   console_log_color=True, logfile_file=script_name + ".log",
                   logfile_log_level=level, logfile_log_color=False)

    FILE_PATH = args.file_path
    DIR_IN_IMAGE = args.input_image
    DIR_IN_CLOTH = args.input_cloth
    DIR_OUT_MASK = args.output_mask
    DIR_OUT_POSE = args.output_pose

    catvton_ckpt_path = snapshot_download(args.catvton_ckpt_path) if not os.path.exists(args.catvton_ckpt_path) else args.catvton_ckpt_path

    # Initialise Automasker object
    automasker = AutoMasker(densepose_ckpt=os.path.join(catvton_ckpt_path, "DensePose"),
                            schp_ckpt=os.path.join(catvton_ckpt_path, "SCHP"),
                            device='cuda')

    logging.info("autoMasker instance initialised")

    # Read paired list
    with open(FILE_PATH, 'r') as f: lines = f.readlines()
    # Main loop
    for line in lines:
        image_file, cloth_file = line.strip().split()
        full_image_file = os.path.join(DIR_IN_IMAGE, image_file)
        full_cloth_file = os.path.join(DIR_IN_CLOTH, cloth_file)
        full_json_file = os.path.splitext(full_cloth_file)[0] + '.json'

        # Parse JSON
        with open(full_json_file, "r") as f:record = json.load(f)
        mask_type = record.get('cloth_type')

        image = Image.open(full_image_file)

        mask_path = os.path.join(DIR_OUT_MASK, image_file.replace('.jpg', '_mask.png'))

        logging.info("%s | Generating mask for %s", image_file, mask_type)

        conditions = automasker(image, mask_type=mask_type)

        mask = conditions['mask']
        conditions['mask'].save(mask_path)

        logging.info("%s | Saved generated  MASK to %s", image_file, mask)

        densepose_path = os.path.join(DIR_OUT_POSE, image_file)

        logging.info("%s | Generating DensePose", image_file)

        densepose = automasker.densepose_processor(image, resize=1024, colormap=cv2.COLORMAP_PARULA)
        densepose.save(densepose_path)

        logging.info("%s | Saved generated DensePose to %s", image_file, densepose_path)

if __name__ == "__main__":
    main()
