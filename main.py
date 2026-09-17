import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ocr import extract_text


IMAGE_PATH = "images/images1.jpg"
OUTPUT_FOLDER = "scanned"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "final_scanned_document.png")
MAX_SIZE = 1600


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return image


def resize_image(image):
    h, w = image.shape[:2]
    if max(h, w) <= MAX_SIZE:
        return image.copy()
    scale = MAX_SIZE / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def lighting_correction(gray):
    background = cv2.GaussianBlur(gray, (0, 0), 25)
    background = np.maximum(background, 1)
    corrected = cv2.divide(gray, background, scale=220)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def adaptive_threshold(gray):
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        7
    )


def morphology(threshold):
    # Remove only tiny isolated noise
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    result = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1
    )

    # Connect nearby parts of the document boundary
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=1
    )

    return result


def canny_edges(gray):
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )


def grabcut_document(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    rect = (
        int(w * 0.03),
        int(h * 0.01),
        int(w * 0.94),
        int(h * 0.96)
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        background_model,
        foreground_model,
        6,
        cv2.GC_INIT_WITH_RECT
    )

    mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    return mask


def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)

    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).reshape(-1)

    result[0] = points[np.argmin(total)]
    result[2] = points[np.argmax(total)]
    result[1] = points[np.argmin(difference)]
    result[3] = points[np.argmax(difference)]

    return result


def find_four_corners(contour):
    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approx) == 4:
        return order_points(
            approx.reshape(4, 2).astype(np.float32)
        )

    return None


def document_from_mask(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    image_area = mask.shape[0] * mask.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.05:
            continue

        corners = find_four_corners(contour)

        if corners is not None:
            candidates.append((area, corners))

    if candidates:
        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )
        return candidates[0][1]

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) >= image_area * 0.05:
        rectangle = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rectangle)
        return np.float32(box)

    return None


def document_from_edges(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    image_area = edges.shape[0] * edges.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.05:
            continue

        corners = find_four_corners(contour)

        if corners is not None:
            candidates.append((area, corners))

    if candidates:
        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )
        return candidates[0][1]

    return None

def find_document_candidate(binary, image_shape):

    h, w = image_shape[:2]
    image_area = h * w

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore small objects
        if area < image_area * 0.25:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        points = approx.reshape(
            4,
            2
        ).astype(np.float32)

        # ------------------------------------------------------
        # Area ratio
        # ------------------------------------------------------

        area_ratio = (
            cv2.contourArea(points)
            / image_area
        )

        # ------------------------------------------------------
        # Bounding rectangle
        # ------------------------------------------------------

        x, y, bw, bh = cv2.boundingRect(
            points.astype(np.int32)
        )

        if bw <= 0 or bh <= 0:
            continue

        rectangularity = (
            cv2.contourArea(points)
            / float(bw * bh)
        )

        # ------------------------------------------------------
        # Reject extremely thin rectangles
        # ------------------------------------------------------

        aspect_ratio = max(bw, bh) / min(bw, bh)

        if aspect_ratio > 6:
            continue

        # ------------------------------------------------------
        # Distance from image borders
        # ------------------------------------------------------

        left = np.min(points[:, 0])
        top = np.min(points[:, 1])
        right = np.max(points[:, 0])
        bottom = np.max(points[:, 1])

        border_distance = min(
            left,
            top,
            w - 1 - right,
            h - 1 - bottom
        )

        # ------------------------------------------------------
        # Border proximity
        # ------------------------------------------------------

        border_score = 1.0 - min(
            max(border_distance, 0)
            / (0.25 * min(h, w)),
            1.0
        )

        # ------------------------------------------------------
        # Size score
        # ------------------------------------------------------

        size_score = min(
            area_ratio / 0.90,
            1.0
        )

        # ------------------------------------------------------
        # Final score
        # ------------------------------------------------------

        score = (
            size_score * 0.60 +
            rectangularity * 0.20 +
            border_score * 0.20
        )

        candidates.append(
            (
                score,
                area_ratio,
                points
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_area_ratio, best_points = candidates[0]

    # ----------------------------------------------------------
    # IMPORTANT:
    # Do not accept an internal rectangle just because it has
    # four corners.
    # ----------------------------------------------------------

    if best_area_ratio < 0.40:
        return None

    return order_points(best_points)


def detect_document(image, edges, mask=None):

    h, w = image.shape[:2]
    image_area = h * w

    candidates = []

    # ==========================================================
    # CONVERT ORIGINAL IMAGE TO GRAYSCALE
    # ==========================================================

    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # ==========================================================
    # METHOD 1
    # CANNY EDGE DETECTION
    # ==========================================================

    candidate = find_document_candidate(
        edges,
        image.shape
    )

    if candidate is not None:

        area = cv2.contourArea(
            candidate.astype(np.float32)
        )

        candidates.append(
            (
                area / image_area,
                candidate,
                "Canny"
            )
        )

    # ==========================================================
    # METHOD 2
    # BRIGHT DOCUMENT DETECTION
    #
    # Useful for white receipts/bills on dark backgrounds.
    # ==========================================================

    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )

    _, bright = cv2.threshold(
        blurred,
        150,
        255,
        cv2.THRESH_BINARY
    )

    # Connect the white areas belonging to the bill
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    # Remove small objects
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    contours, _ = cv2.findContours(
        bright,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        # Bill should be reasonably large
        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        # Try several approximation strengths
        best_quad = None

        for epsilon_factor in [0.015, 0.02, 0.03, 0.04, 0.05]:

            approx = cv2.approxPolyDP(
                contour,
                epsilon_factor * perimeter,
                True
            )

            if len(approx) == 4:

                points = approx.reshape(
                    4,
                    2
                ).astype(np.float32)

                if cv2.isContourConvex(
                    points.astype(np.int32)
                ):
                    best_quad = points
                    break

        # ------------------------------------------------------
        # If we found four corners
        # ------------------------------------------------------

        if best_quad is not None:

            quad_area = cv2.contourArea(
                best_quad
            )

            area_ratio = (
                quad_area / image_area
            )

            if area_ratio < 0.15:
                continue

            x, y, bw, bh = cv2.boundingRect(
                best_quad.astype(np.int32)
            )

            if bw <= 0 or bh <= 0:
                continue

            rectangularity = (
                quad_area /
                float(bw * bh)
            )

            # Reject extremely thin shapes
            aspect_ratio = (
                max(bw, bh) /
                min(bw, bh)
            )

            if aspect_ratio > 6:
                continue

            # --------------------------------------------------
            # Brightness score
            # --------------------------------------------------

            mask_for_mean = np.zeros(
                gray.shape,
                dtype=np.uint8
            )

            cv2.fillPoly(
                mask_for_mean,
                [best_quad.astype(np.int32)],
                255
            )

            mean_brightness = cv2.mean(
                gray,
                mask=mask_for_mean
            )[0]

            brightness_score = min(
                mean_brightness / 255.0,
                1.0
            )

            # --------------------------------------------------
            # Center score
            # --------------------------------------------------

            center_x = np.mean(
                best_quad[:, 0]
            )

            center_y = np.mean(
                best_quad[:, 1]
            )

            image_center_x = w / 2
            image_center_y = h / 2

            distance = np.sqrt(
                (
                    (center_x - image_center_x) / w
                ) ** 2 +
                (
                    (center_y - image_center_y) / h
                ) ** 2
            )

            center_score = max(
                0,
                1.0 - distance * 2
            )

            # --------------------------------------------------
            # Final score
            # --------------------------------------------------

            score = (
                min(area_ratio / 0.70, 1.0) * 0.45
                +
                rectangularity * 0.20
                +
                brightness_score * 0.25
                +
                center_score * 0.10
            )

            candidates.append(
                (
                    score,
                    area_ratio,
                    best_quad,
                    "Bright document"
                )
            )

    # ==========================================================
    # METHOD 3
    # GRABCUT MASK
    # ==========================================================

    if mask is not None:

        mask_binary = np.where(
            mask > 0,
            255,
            0
        ).astype(np.uint8)

        kernel_mask = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (15, 15)
        )

        mask_binary = cv2.morphologyEx(
            mask_binary,
            cv2.MORPH_CLOSE,
            kernel_mask,
            iterations=2
        )

        candidate = find_document_candidate(
            mask_binary,
            image.shape
        )

        if candidate is not None:

            area = cv2.contourArea(
                candidate.astype(np.float32)
            )

            candidates.append(
                (
                    area / image_area,
                    area / image_area,
                    candidate,
                    "GrabCut"
                )
            )

    # ==========================================================
    # SELECT BEST CANDIDATE
    # ==========================================================

    if candidates:

        # Normalize candidate format
        normalized_candidates = []

        for item in candidates:

            if len(item) == 3:

                area_ratio, candidate, method = item

                score = area_ratio

            else:

                score, area_ratio, candidate, method = item

            normalized_candidates.append(
                (
                    score,
                    area_ratio,
                    candidate,
                    method
                )
            )

        normalized_candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        best_score, best_ratio, best_candidate, method = (
            normalized_candidates[0]
        )

        # ------------------------------------------------------
        # Accept only a meaningful document
        # ------------------------------------------------------

        if best_ratio >= 0.15:

            print(
                f"Document detected using: {method}"
            )

            print(
                f"Document area: {best_ratio * 100:.1f}%"
            )

            return order_points(
                best_candidate
            )

    # ==========================================================
    # FULL IMAGE FALLBACK
    # ==========================================================

    print(
        "No reliable document boundary detected."
    )

    print(
        "Using full image as fallback."
    )

    return np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ],
        dtype=np.float32
    )


def draw_boundary(image, corners):
    result = image.copy()
    points = np.int32(order_points(corners))

    cv2.polylines(
        result,
        [points],
        True,
        (0, 255, 0),
        5
    )

    for x, y in points:
        cv2.circle(
            result,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

    return result


def perspective_correct(image, corners):
    points = order_points(corners)
    tl, tr, br, bl = points

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = max(int(width_top), int(width_bottom))
    height = max(int(height_left), int(height_right))

    width = max(width, 300)
    height = max(height, 400)

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        points,
        destination
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


def final_scan(warped):
    # --------------------------------------------------
    # 1. Increase resolution
    # --------------------------------------------------
    warped = cv2.resize(
        warped,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------
    # 2. Grayscale
    # --------------------------------------------------
    gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # 3. Remove small camera noise
    # --------------------------------------------------
    gray = cv2.bilateralFilter(
        gray,
        7,
        35,
        35
    )

    # --------------------------------------------------
    # 4. Estimate large background
    # --------------------------------------------------
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        35
    )

    background = np.maximum(
        background,
        1
    )

    # --------------------------------------------------
    # 5. Correct uneven lighting
    # --------------------------------------------------
    normalized = cv2.divide(
        gray,
        background,
        scale=255
    )

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    # --------------------------------------------------
    # 6. Improve contrast
    # --------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=1.3,
        tileGridSize=(10, 10)
    )

    enhanced = clahe.apply(
        normalized
    )

    # --------------------------------------------------
    # 7. Gentle sharpening
    # --------------------------------------------------
    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.25,
        blur,
        -0.25,
        0
    )

    # --------------------------------------------------
    # 8. Remove very small noise
    # WITHOUT destroying text
    # --------------------------------------------------
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    cleaned = cv2.morphologyEx(
        sharpened,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )

    # --------------------------------------------------
    # 9. Improve white paper appearance
    # --------------------------------------------------
    cleaned = cv2.normalize(
        cleaned,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # --------------------------------------------------
    # IMPORTANT:
    # Keep FINAL output grayscale.
    #
    # Do NOT apply adaptiveThreshold here.
    # This preserves faint printed characters.
    # --------------------------------------------------

    return cleaned

def crop_white_border(image):
    h, w = image.shape[:2]

    margin_y = int(h * 0.015)
    margin_x = int(w * 0.015)

    return image[
        margin_y:h - margin_y,
        margin_x:w - margin_x
    ]


def display_results(
    original,
    gray,
    denoised,
    lighting,
    histogram,
    threshold,
    morph,
    edges,
    boundary,
    warped,
    scanned
):
    fig = plt.figure(
        figsize=(15, 10),
        constrained_layout=True
    )

    grid = fig.add_gridspec(
        3,
        4
    )

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    ax4 = fig.add_subplot(grid[0, 3])
    ax5 = fig.add_subplot(grid[1, 0])
    ax6 = fig.add_subplot(grid[1, 1])
    ax7 = fig.add_subplot(grid[1, 2])
    ax8 = fig.add_subplot(grid[1, 3])
    ax9 = fig.add_subplot(grid[2, 0])
    ax10 = fig.add_subplot(grid[2, 1])
    ax11 = fig.add_subplot(grid[2, 2:4])

    ax1.imshow(
        cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )
    )
    ax1.set_title("1. Original")
    ax1.axis("off")

    ax2.imshow(
        gray,
        cmap="gray"
    )
    ax2.set_title("2. Grayscale")
    ax2.axis("off")

    ax3.imshow(
        denoised,
        cmap="gray"
    )
    ax3.set_title("3. Noise Reduction")
    ax3.axis("off")

    ax4.imshow(
        lighting,
        cmap="gray"
    )
    ax4.set_title("4. Lighting Correction")
    ax4.axis("off")

    ax5.plot(
        histogram.ravel()
    )
    ax5.set_title("5. Intensity Histogram")
    ax5.set_xlabel("Intensity")
    ax5.set_ylabel("Frequency")

    ax6.imshow(
        threshold,
        cmap="gray"
    )
    ax6.set_title("6. Adaptive Thresholding")
    ax6.axis("off")

    ax7.imshow(
        morph,
        cmap="gray"
    )
    ax7.set_title("7. Morphological Processing")
    ax7.axis("off")

    ax8.imshow(
        edges,
        cmap="gray"
    )
    ax8.set_title("8. Canny Edge Detection")
    ax8.axis("off")

    ax9.imshow(
        cv2.cvtColor(
            boundary,
            cv2.COLOR_BGR2RGB
        )
    )
    ax9.set_title("9. Detected Document Boundary")
    ax9.axis("off")

    ax10.imshow(
        cv2.cvtColor(
            warped,
            cv2.COLOR_BGR2RGB
        )
    )
    ax10.set_title("10. Perspective Corrected")
    ax10.axis("off")

    ax11.imshow(
        scanned,
        cmap="gray"
    )
    ax11.set_title("11. Final Scanned Document")
    ax11.axis("off")

    plt.show()


def process_image():
    print("Input:", IMAGE_PATH)

    original = load_image(
        IMAGE_PATH
    )

    print(
        "Original image shape:",
        original.shape
    )

    original = resize_image(
        original
    )

    print(
        "Processing image shape:",
        original.shape
    )

    gray = grayscale(
        original
    )

    denoised = reduce_noise(
        gray
    )

    lighting = lighting_correction(
        denoised
    )

    histogram = cv2.calcHist(
        [lighting],
        [0],
        None,
        [256],
        [0, 256]
    )

    threshold = adaptive_threshold(
        lighting
    )

    morph = morphology(
        threshold
    )

    edges = canny_edges(
        lighting
    )

    # Use GrabCut + contours to separate the actual bill
    # from the hands and background.
    document_mask = grabcut_document(
        original
    )

    corners = detect_document(
        original,
        document_mask,
        edges
    )

    if corners is None:
        print()
        print("ERROR: Document boundary could not be detected.")
        print("Try another image or check the document visibility.")
        return

    print("Document boundary detected successfully.")

    boundary = draw_boundary(
        original,
        corners
    )

    warped = perspective_correct(
        original,
        corners
    )

    scanned = final_scan(
        warped
    )

    scanned = crop_white_border(
        scanned
    )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # Delete old generated files so scanned/
    # contains ONLY the current final scan.
    for filename in os.listdir(
        OUTPUT_FOLDER
    ):
        path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )
        if os.path.isfile(path):
            os.remove(path)

    if not cv2.imwrite(
        OUTPUT_PATH,
        scanned
    ):
        print("ERROR: Final image could not be saved.")
        return

    print(
        "Final scanned document saved:",
        OUTPUT_PATH
    )

    # ----------------------------------------------------------
    # DOCUMENT SCANNING COMPLETED
    # ----------------------------------------------------------

    print()
    print("=" * 60)
    print("DOCUMENT SCANNING COMPLETED")
    print("=" * 60)

    # ----------------------------------------------------------
    # RUN OCR ON FINAL SCANNED DOCUMENT
    # ----------------------------------------------------------

    extract_text(
        OUTPUT_PATH
    )

    # ----------------------------------------------------------
    # DISPLAY ALL PROCESSING RESULTS
    # ----------------------------------------------------------

    display_results(
        original,
        gray,
        denoised,
        lighting,
        histogram,
        threshold,
        morph,
        edges,
        boundary,
        warped,
        scanned
    )

if __name__ == "__main__":
    process_image()