import cv2
import numpy as np
import argparse
import os
from locations import Locations

parser = argparse.ArgumentParser(description="FRLG Shiny Hunter")
parser.add_argument('-method', type=str, default='walking', help="Method to use for hunting")   # walking, starter, staticGift, staticEncounter
parser.add_argument('-route', type=str, default='0', help="Route to use for hunting")
parser.add_argument('-target', type=str, default='0', help="Target National Dex number")
args = parser.parse_args()

method = args.method
route = args.route
target = int(args.target)

print(method, route, target)
routes = Locations.routes

# route = 3
if method == 'walking' or method == 'staticEncounter':
    sprite = 'capture'
    scanDir = ['encounter/shiny','encounter/original']
    cornerRange = {'x':[1080,1140], 'y':[70,200]}
else:
    sprite = 'summary'
    scanDir = ['summary/shiny','summary/original']
    cornerRange = {'x':[370,400], 'y':[220,250]}


def match_template_with_alpha(source_path: str, template_path: str, threshold: float = 0.8, roi: tuple = None):
    """
    Perform template matching using a transparent PNG template.
    The alpha channel is used as a mask so transparent areas are ignored.

    Args:
        source_path:  Path to the source image (the image to search within).
        template_path: Path to the template PNG (must have an alpha channel).
        threshold:    Match confidence threshold (0.0 – 1.0).
        roi:          Optional region of interest (x, y, w, h) to limit the search area.

    Returns:
        List of (x, y, w, h) tuples for each match found.
    """
    # Load source as BGR (no alpha needed)
    source = cv2.imread(source_path, cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"Source image not found: {source_path}")
    
    roi_x, roi_y = 0, 0
    if roi is not None:
        roi_x, roi_y, roi_w, roi_h = roi
        source = source[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        # cv2.imshow("ROI", source)  # Debug: Show the region of interest
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    # Load template with alpha channel intact
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        raise FileNotFoundError(f"Template image not found: {template_path}")
    if template.shape[2] != 4:
        raise ValueError("Template image does not have an alpha channel.")

    # Split into BGR and alpha mask
    tmpl_bgr  = template[:, :, :3]
    tmpl_mask = template[:, :,  3]   # 0 = transparent, 255 = opaque

    h, w = tmpl_bgr.shape[:2]

    # cv2.TM_CCORR_NORMED is the method that supports a mask argument
    result = cv2.matchTemplate(source, tmpl_bgr, cv2.TM_CCORR_NORMED, mask=tmpl_mask)

    # Find all locations above the threshold
    locations = np.where(result >= threshold)
    matches = []

    for pt in zip(*locations[::-1]):   # (col, row) → (x, y)
        matches.append((pt[0], pt[1], w, h))

    # Remove overlapping duplicates with non-maximum suppression
    matches = _nms(matches)
    return matches


def _nms(matches: list, overlap_thresh: float = 0.3) -> list:
    """Simple non-maximum suppression to remove duplicate detections."""
    if not matches:
        return []

    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in matches], dtype=float)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= overlap_thresh)[0] + 1]

    return [matches[i] for i in keep]


def draw_matches(source_path: str, matches: list, output_path: str = "output.png"):
    """Draw bounding boxes around matches and save the result."""
    img = cv2.imread(source_path)
    if img is None:
        raise FileNotFoundError(f"Source image not found for drawing: {source_path}")
    for x, y, w, h in matches:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (160+210, 210+24), (160+210+440, 210+24+386), (255, 0, 0), 2)
        cv2.rectangle(img, (370, 220), (400, 250), (0, 0, 255), 2)  # Debug: Highlight expected area
    cv2.imwrite(output_path, img)
    print(f"Saved result to {output_path}")

def checkMatches(matches, screen, sprite, corner, shinySprite: bool):
    print('Checking in function')
    for x, y, w, h in matches:
        print(f"  → x={x}, y={y}, w={w}, h={h}")
        # if 1080 <= x <= 1140 and 70 <= y <= 200:
        if corner['x'][0] <= x <= corner['x'][1] and corner['y'][0] <= y <= corner['y'][1]:
            if shinySprite:
                print(f'Shiny {sprite.name} has {len(matches)} matches with {screen.name}.')
                draw_matches(screen.path, matches, 'matches/shiny_'+sprite.name)
            else:
                print(f'{sprite.name} has {len(matches)} matches with {screen.name}.')
                draw_matches(screen.path, matches, 'matches/'+sprite.name)
        else:
            if shinySprite:
                print(f'Shiny {sprite.name} has {len(matches)} matches in wrong locations with {screen.name}.')
            else:
                print(f'{sprite.name} has {len(matches)} matches in wrong locations with {screen.name}.')
    
    return

# --- Example usage ---
if __name__ == "__main__":

    # img = cv2.imread("captures/ivysaur.png")
    # cv2.rectangle(img, (160+210, 210+24), (160+210+440, 210+24+386), (255, 0, 0), 2)
    # cv2.imwrite("roi.png", img)

    # if sprite == 'summary':

    #     for grab in os.scandir('captures'):
    #         for each in os.scandir('summary/shiny'):
    #             # print(each.path)
    #             matches = match_template_with_alpha(
    #                 source_path=grab.path,
    #                 template_path=each.path,  # PNG with transparency
    #                 threshold=0.95,
    #                 # roi=(160+210, 180+24, 440, 440)
    #             )
    #             if len(matches) > 0:
    #                 if len(matches) == 1:
    #                     for x, y, w, h in matches:
    #                         # print(f"  → x={x}, y={y}, w={w}, h={h}") 
    #                         if 370 <= x <= 400 and 220 <= y <= 250:
    #                             print(f'Shiny {each.name} has {len(matches)} matches with {grab.name}.')
    #                             draw_matches("summary.png", matches, 'matches/'+each.name)
    #                         else:
    #                             print(f'Shiny {each.name} has {len(matches)} matches in wrong locations with {grab.name}.')
    #                 else:
    #                     print(f'{each.name} has {len(matches)} matches with {grab.name}, but multiple matches found. Skipping drawing.')
            
    #         for each in os.scandir('summary/original'):
    #             # print(grab.path)
    #             # print(each.path)
    #             matches = match_template_with_alpha(
    #                 source_path=grab.path,
    #                 template_path=each.path,  # PNG with transparency
    #                 threshold=0.95,
    #                 # roi=(160+210, 210+24, 440, 400),
    #             )

    #             if len(matches) > 0:
    #                 if len(matches) == 1:
    #                     for x, y, w, h in matches:
    #                         print(f"  → x={x}, y={y}, w={w}, h={h}")
    #                         if 370 <= x <= 400 and 220 <= y <= 250:
    #                             print(f'{each.name} has {len(matches)} matches with {grab.name}.')
    #                             draw_matches(grab.path, matches, 'matches/'+each.name)
    #                         else:
    #                             print(f'{each.name} has {len(matches)} matches in wrong locations with {grab.name}.')
    #                 else:
    #                     print(f'{each.name} has {len(matches)} matches with {grab.name}, but multiple matches found. Skipping drawing.')

    # elif sprite == 'capture':
    print('Going through captures')
    for grab in os.scandir('captures'):
        if grab.name != 'ivysaur.png':
            continue
        for each in os.scandir(scanDir[0]):
            try: num = int(each.name.split('.')[0])
            except ValueError:
                continue
            if num not in routes[route]:
                continue
            if target == 0:
                matches = match_template_with_alpha(
                    source_path=grab.path,
                    template_path=each.path,  # PNG with transparency
                    threshold=0.95,
                    # roi=(160+210, 180+24, 440, 440)
                )
                if len(matches) > 0:
                    if len(matches) >= 1:
                        checkMatches(matches,grab,each,cornerRange,True)
            else:
                if num == target:
                    matches = match_template_with_alpha(
                        source_path=grab.path,
                        template_path=each.path,  # PNG with transparency
                        threshold=0.95,
                        # roi=(160+210, 180+24, 440, 440)
                    )
                    if len(matches) > 0:
                        if len(matches) >= 1:
                            checkMatches(matches,grab,each,cornerRange,True)
                else: continue
        
        for each in os.scandir(scanDir[1]):
            try: num = int(each.name.split('.')[0])
            except ValueError:
                continue
            if num not in routes[route]:
                continue
            matches = match_template_with_alpha(
                source_path=grab.path,
                template_path=each.path,  # PNG with transparency
                threshold=0.95,
                # roi=(160+210, 210+24, 440, 400),
            )

            if len(matches) > 0:
                if len(matches) >= 1:
                    checkMatches(matches,grab,each,cornerRange,False)