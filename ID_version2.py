#!/usr/bin/env python
# coding: utf-8

from helpers import *

import json
import base64
from flask import Flask, jsonify, request
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/predict_image', methods=['POST'])
def predict_image():


    def OCR_pipeline(img_path=''):
        """
        Extracts ID number from an image using contour detection, perspective transformation,
        cropping, thresholding, and OCR.

        Args:
            img_path (str, optional): Path to the image. Defaults to ''.

        Returns:
            int: Extracted ID number (if found and 14 digits), otherwise None.

        Raises:
            Exception: If the extracted ID number is not 14 digits.
        """

        gray = img_to_gray(img_path)

        # Get homography between Gomhoreyet Masr template and ID
        M, match_boundingbox = findOrientation(gray)

        # If matches found, apply orientation
        if M is not None:
            gray, img_path, M = applyOrientation(gray, img_path, M)

            # Blur gray for smoothing the image for edge detection
            gray = blur_img(gray)

            # Adaptive threshold
            bin = adaptive_threshold(gray, 11, 2)

            # Edge detection
            edges = canny_edge_detector(bin, 200, 600)

            # Getting largest Contour
            contour_bbox, largest_contour = findLargestContour(edges)
            if contour_bbox is not None:

                cropped_img = output_img(img_path, largest_contour)
                cropped_gray = img_to_gray(cropped_img)

                selections, bboxes = detect_text_regions(cropped_gray)
                for i in selections:
                    x, y, x2, y2 = bboxes[i]
                    text_img = cropped_img[y:y2, x:x2]
                    text_bin = preproccess_OCR(text_img)
                    id_str = OCR_pytesseract(text_bin).replace(" ", "").replace("\n", "").rstrip()
                    if len(id_str) != 14:
                        continue
                    National_ID = int(id_str)
                    return National_ID
            else:
                print("No contours")
        else:
            print("No ID Card Found / No contours")

        return None  # Indicate no ID found

    try:
        # Check if images are present and valid
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({"status": 400, "msg": "Missing one or both images (image1, image2)"}), 400

        # Read images from form data and convert to NumPy arrays
        image1_file = request.files['image1']
        image1_data = np.frombuffer(image1_file.read(), np.uint8)
        image1_array = cv2.imdecode(image1_data, cv2.IMREAD_COLOR)

        image2_file = request.files['image2']
        image2_data = np.frombuffer(image2_file.read(), np.uint8)
        image2_array = cv2.imdecode(image2_data, cv2.IMREAD_COLOR)

        # Process images
        id = OCR_pipeline(image1_array)

        # Return response
        if id is None:
            return jsonify({"status": 200, "data": {"id": None}}), 200  # Indicate no ID found
        else:
            return jsonify({"status": 200, "data": {"id": id}}), 200

    except Exception as e:
        # Handle errors during processing
        return jsonify({"status": 500, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
