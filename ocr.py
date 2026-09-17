import os

# Disable PaddlePaddle PIR
os.environ["FLAGS_enable_pir_api"] = "0"

# Disable oneDNN / MKLDNN
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR


OUTPUT_FOLDER = "ocr_output"
OUTPUT_TEXT = os.path.join(
    OUTPUT_FOLDER,
    "extracted_text.txt"
)


def extract_text(image_path):

    # ----------------------------------------------------------
    # CHECK INPUT IMAGE
    # ----------------------------------------------------------

    if not os.path.exists(image_path):

        print(
            f"OCR Error: Image not found: {image_path}"
        )

        return None

    # ----------------------------------------------------------
    # CREATE OCR OUTPUT FOLDER
    # ----------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # ----------------------------------------------------------
    # INITIALIZE OCR
    # ----------------------------------------------------------

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False
    )

    # ----------------------------------------------------------
    # RUN OCR
    # ----------------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING OCR")
    print("=" * 60)

    results = ocr.predict(image_path)

    extracted_lines = []

    # ----------------------------------------------------------
    # EXTRACT TEXT
    # ----------------------------------------------------------

    for result in results:

        try:

            data = result.json

            # Some PaddleOCR versions expose json as a method
            if callable(data):
                data = data()

        except Exception:

            data = None

        if isinstance(data, dict):

            res_data = data.get(
                "res",
                data
            )

            texts = res_data.get(
                "rec_texts",
                []
            )

            scores = res_data.get(
                "rec_scores",
                []
            )

            for i, text in enumerate(texts):

                text = str(text).strip()

                if not text:
                    continue

                # --------------------------------------------------
                # CONFIDENCE FILTER
                # --------------------------------------------------

                if i < len(scores):

                    try:

                        score = float(
                            scores[i]
                        )

                        if score < 0.30:
                            continue

                    except (ValueError, TypeError):
                        pass

                extracted_lines.append(text)

    # ----------------------------------------------------------
    # SAVE TEXT FILE
    # ----------------------------------------------------------

    with open(
        OUTPUT_TEXT,
        "w",
        encoding="utf-8"
    ) as file:

        for line in extracted_lines:

            file.write(
                line + "\n"
            )

    # ----------------------------------------------------------
    # DISPLAY RESULT
    # ----------------------------------------------------------

    print()
    print("=" * 60)
    print("EXTRACTED TEXT")
    print("=" * 60)

    if extracted_lines:

        for line in extracted_lines:
            print(line)

    else:

        print("No text detected.")

    print()
    print("=" * 60)
    print(
        f"OCR text saved to: {OUTPUT_TEXT}"
    )
    print("=" * 60)

    return extracted_lines