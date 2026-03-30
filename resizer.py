# import os, sys
# import cv2

# directory = 'firered-leafgreen/shiny'

# for file in os.scandir(directory):
#     if file.is_file():
#         if '.png' in file.name:
#             print(file.name)
#             # print(file)
#             image = cv2.imread(file)
#             flipped = cv2.flip(image,1)
#             resize = cv2.resize(flipped, (400,400), interpolation=cv2.INTER_NEAREST)
            
#             cv2.imwrite('shiny/'+file.name,resize)


import os
import cv2
import numpy as np

directory = 'encounter/shiny'

def remove_background(image):
    """Flood-fill from all four corners to mask the background, then make it transparent."""
    # Convert to BGRA so we have an alpha channel to write into
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w = bgra.shape[:2]

    # Build a mask via flood-fill from each corner
    # The mask must be 2px larger than the image (OpenCV requirement)
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    corners = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]
    for cx, cy in corners:
        cv2.floodFill(
            image.copy(),   # floodFill needs a writable copy of the source
            fill_mask,
            (cx, cy),
            newVal=(0, 0, 0),       # replacement color (irrelevant, we only use the mask)
            loDiff=(10, 10, 10),    # tolerance — raise if edges are being missed
            upDiff=(10, 10, 10),
            flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),  # write 255 into mask
        )

    # Trim the 1px border OpenCV adds to the mask
    bg_mask = fill_mask[1:h + 1, 1:w + 1]

    # Anywhere the mask is filled = background → set alpha to 0
    bgra[:, :, 3] = np.where(bg_mask == 255, 0, 255).astype(np.uint8)
    return bgra


for file in os.scandir(directory):
    if file.is_file() and file.name.endswith('.png'):
        print(file.name)

        image = cv2.imread(file.path)           # read as BGR
        flipped = cv2.flip(image, 1)            # mirror horizontally
        # resized = cv2.resize(                   # scale up with no blur
        #     flipped, (400, 400),
        #     interpolation=cv2.INTER_NEAREST
        # )

        transparent = remove_background(flipped)
        cv2.imwrite('encounter/shiny/' + file.name, transparent)