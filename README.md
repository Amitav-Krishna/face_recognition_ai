Face Recognition AI

This project uses AI and Python to detect whether a face is visible in multiple images. It extracts facial features using DeepFace, generates embeddings, and compares these embeddings to determine if faces in new images match known individuals.
Features

    Face Detection and Embedding: Uses DeepFace (Facenet) and OpenCV backend to extract 128-dimensional face embeddings from images.
    Image Collection: Recursively collects images from specified directories.
    Face Comparison: Compares embeddings using cosine similarity to identify matches.
    Handles Multiple People: Can be configured for multiple known identities (e.g., person_a, person_b).
    Error Handling: Skips images where no face is detected.

Requirements

    Python 3.x
    DeepFace
    numpy
    scipy

Install dependencies with:
bash

pip install deepface numpy scipy

Usage

    Prepare your data:
        Place known images of people in separate folders under data/, for example:
        Code

    data/
      person_a/
        img1.jpg
        img2.jpg
      person_b/
        img3.jpg

    Place test images in a test/ directory.

Run the script:
bash

    python face_feature_extraction.py

    Output:
        The script prints the paths of images being processed.
        For each test image, it prints "Match found!" if a match is found among known faces, or "No match." otherwise.

How it works

    The script first gathers all images in the data/person_a and data/person_b directories and extracts their face embeddings.
    It then gathers all test images in the test/ directory and extracts their embeddings.
    Each test embedding is compared to all known embeddings using cosine distance (threshold = 0.2 by default).
    If a match is found, it reports the test image as matched; otherwise, it reports no match.

Customization

    Add More People: Add more folders under data/ for new identities.
    Change Threshold: Adjust the threshold parameter in the is_match function for stricter/looser matching.
    Supported Formats: The script processes .jpg, .jpeg, and .png files by default.

Troubleshooting

    If DeepFace cannot detect a face in an image, the script will print a warning and skip that image.
    Ensure images are clear and well-lit for best results.

License

This project is open source and available under the MIT License.

Author: Amitav-Krishna
