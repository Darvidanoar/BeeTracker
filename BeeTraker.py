import cv2
import math
import numpy as np
import sqlite3
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls
from scipy.optimize import linear_sum_assignment
from datetime import datetime, time


DB_PATH = "/home/david/Documents/RPi5/BeeTracker/hive_activity.db"
INTERVAL_SECONDS = 6 * 60  # 6 minutes
CUTOFF = time(19, 30)  # 7:30 PM

def init_db(db_path=DB_PATH):
    """Create the sqlite database/table if they don't already exist and
    return an open connection. Using WAL mode lets the file be safely
    read (e.g. for a dashboard) while this script is still writing to it."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hive_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            bees_exiting INTEGER NOT NULL,
            bees_returning INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def log_interval(conn, bees_exiting, bees_returning, timestamp=None):
    """Insert one interval's tally into the database."""
    timestamp = timestamp or datetime.now()
    conn.execute(
        "INSERT INTO hive_activity (timestamp, bees_exiting, bees_returning) VALUES (?, ?, ?)",
        (timestamp.strftime("%Y-%m-%d %H:%M:%S"), bees_exiting, bees_returning),
    )
    conn.commit()


class CentroidTracker:
    """
    Minimal centroid tracker: assigns a persistent ID to each detected
    object by matching new frame centroids to the closest ones from the
    previous frame. This is what lets us compute a *direction* per object
    instead of just a position, since direction requires knowing which
    blob in this frame corresponds to which blob in the last frame.
    """

    def __init__(self, max_disappeared=15, max_match_distance=100):
        self.next_object_id = 0
        self.objects = {}        # id -> (x, y) current centroid
        self.disappeared = {}    # id -> number of consecutive frames missing
        self.max_disappeared = max_disappeared
        self.max_match_distance = max_match_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        # No detections this frame: age out everyone
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Nothing tracked yet: register all detections as new objects
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Distance between every existing object and every new detection
        D = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids)[np.newaxis, :],
            axis=2,
        )

        # Globally optimal matching (Hungarian algorithm) rather than
        # greedy nearest-first. This matters most right after a merged
        # blob splits back into separate detections: two candidate IDs can
        # both be close to both new centroids, and greedy matching (which
        # locks in whichever pair happens to have the smallest distance
        # first) can swap the two IDs. Minimizing total assignment distance
        # across every pair at once avoids that swap far more often.
        row_ind, col_ind = linear_sum_assignment(D)

        used_rows, used_cols = set(), set()
        for row, col in zip(row_ind, col_ind):
            if D[row, col] > self.max_match_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        # Existing objects that found no match this frame: age them out
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # New detections that matched no existing object: register as new
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects


def rect_distance(box1, box2):
    """Gap between two (x, y, w, h) boxes: 0 if they overlap/touch,
    otherwise the Euclidean distance between their nearest edges."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
    dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
    return math.hypot(dx, dy)


def group_nearby_contours(contours, max_gap, fragment_max_area):
    """
    Cluster contours that sit within max_gap pixels of each other (by
    bounding-box distance) into groups, treating each group as one blob -
    but only when BOTH contours being joined are individually smaller
    than fragment_max_area (too small to plausibly be a whole bee on
    their own).

    This exists to undo a specific failure mode: as a bee shifts pose,
    the thin waist between its thorax and abdomen can be thinner than
    the erosion kernel used for mask cleanup, so the morphological
    "opening" briefly erases it and RETR_EXTERNAL reports the same bee
    as two separate, individually-undersized contours for a frame or
    two. Left alone, that mints a second tracker ID for the second
    piece, which then gets dropped when the blob reunites - and
    whichever ID survives the merge is a coin flip, so the bee can end
    up with a new ID after simply moving.

    The fragment_max_area gate matters because bees standing right next
    to each other at the hive entrance are common, and each one's own
    contour is already whole-bee-sized - merging those together just
    because they're close would recreate the "several bees look like
    one blob" problem this function is meant to avoid, only earlier in
    the pipeline where split_merged_blob never gets a chance to run on
    them individually.
    """
    n = len(contours)
    boxes = [cv2.boundingRect(c) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        if areas[i] >= fragment_max_area:
            continue
        for j in range(i + 1, n):
            if areas[j] >= fragment_max_area:
                continue
            if rect_distance(boxes[i], boxes[j]) <= max_gap:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(contours[i])
    return list(groups.values())


def estimate_bee_count(area, single_bee_area, merge_trigger_ratio=1.8):
    """How many bees a contour of this area probably contains, given the
    current running estimate of one bee's contour area.

    A single bee's own contour area naturally varies a lot with pose -
    legs splayed out or a slight motion-blur elongation can push it to
    1.3-1.5x its resting size - and its body already has two lobes
    (thorax/abdomen), so a watershed split can succeed on a single bee
    just as easily as on two touching ones. Naively rounding
    area/single_bee_area starts guessing "2" as soon as the ratio
    passes 1.5, which sits right in that normal single-bee range and
    was splitting ordinary bees into two IDs on almost every frame.
    Requiring a bigger jump (merge_trigger_ratio) before ever
    considering more than one bee avoids that false positive while
    still catching genuinely touching/overlapping bees, whose combined
    area lands well past it.
    """
    if single_bee_area <= 0:
        return 1
    ratio = area / single_bee_area
    if ratio < merge_trigger_ratio:
        return 1
    return max(2, round(ratio))


def split_merged_blob(mask_roi, offset, expected_count, min_region_area):
    """
    Split a blob mask that looks like `expected_count` touching/overlapping
    bees into that many separate boxes, using a distance-transform +
    watershed (the standard OpenCV technique for separating touching
    objects - the same one used for touching-coin/cell segmentation).

    mask_roi: binary (0/255) uint8 mask, cropped to the blob's bounding box.
    offset: (x, y) of that crop's top-left corner in the full frame, so
            returned boxes can be translated back to full-frame coordinates.
    min_region_area: a resulting piece smaller than this is treated as a
            leg/antenna artifact rather than a real second bee, and voids
            the whole split (a single lumpy bee - head/thorax/abdomen/legs -
            can otherwise produce more than one distance-transform peak and
            get chopped into fragments of one real bee).
    Returns a list of (x, y, w, h) boxes, or None if a confident split
    couldn't be found (caller should then fall back to one big box).
    """
    if expected_count <= 1:
        return None

    # Distance transform: pixels deep inside a bee are "brighter" than
    # pixels near its edge or near the touching seam with a neighbor.
    dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return None

    # Local maxima of the distance map = likely bee centers. Require a
    # fairly high fraction of the peak distance so shallow bumps from a
    # single bee's own body shape (not a real second bee) don't count.
    kernel = np.ones((9, 9), np.uint8)
    local_max = (dist == cv2.dilate(dist, kernel)) & (dist > 0.6 * dist.max())
    num_peaks, markers = cv2.connectedComponents(local_max.astype(np.uint8))

    # Wrong number of distinct peaks (too few OR too many): not confident
    # enough to split - safer to leave it as one box than guess wrong.
    if num_peaks - 1 != expected_count:
        return None

    markers = markers + 1
    # Outside the blob is "sure background" (label 1, left as-is). Inside
    # the blob but not at a peak is the ambiguous seam between bees -
    # that's what watershed needs to flood and assign, so mark it unknown.
    markers[(mask_roi > 0) & (local_max == 0)] = 0
    mask_bgr = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_bgr, markers)

    ox, oy = offset
    boxes = []
    for label in range(2, markers.max() + 1):
        region = np.uint8(markers == label) * 255
        region_area = cv2.countNonZero(region)
        # Any piece too small to plausibly be a whole bee (a leg, an
        # antenna, a sliver from an uneven split) invalidates the whole
        # split rather than being silently dropped - a bee missing its
        # body isn't a usable detection, and dropping it would also throw
        # off the expected_count vs len(boxes) bookkeeping upstream.
        if region_area < min_region_area:
            return None
        x, y, w, h = cv2.boundingRect(region)
        boxes.append((x + ox, y + oy, w, h))

    return boxes if len(boxes) == expected_count else None


def get_vertical_direction(prev_point, curr_point, min_movement=6):
    """
    Classify vertical movement between two points as 'Down', 'Up',
    or None. Horizontal movement is ignored entirely - this is used to
    filter for objects moving up<->down and ignore ones moving
    left/right or sitting still.
    """
    dy = curr_point[1] - prev_point[1]

    if abs(dy) < min_movement:
        return None

    return "Down" if dy > 0 else "Up"


def get_edge_zone(y, frame_height, zone_fraction=0.15):
    """
    Classify a y-coordinate as being in the 'top' edge zone, the
    'bottom' edge zone, or 'middle' (neither). Used to detect a full
    entry-to-exit crossing rather than every small direction change.
    """
    if y < frame_height * zone_fraction:
        return "top"
    if y > frame_height * (1 - zone_fraction):
        return "bottom"
    return "middle"


def main():
    # 1. Initialize Picamera2 for Camera Module 3
    picam = Picamera2()
    
    # Configure the camera configuration (Lower resolution = Higher FPS)
    
    zoom_height = 1500
    zoom_width = 3072 * zoom_height // 1728
    offset_x = (3072 - zoom_width) // 2
    offset_x = 1000
    offset_y = 500
    
    cam_config = picam.create_preview_configuration(main={"size": (1280, 900), "format": "RGB888"},
        controls={"AfMode": controls.AfModeEnum.Manual, "LensPosition": 8.4, "ScalerCrop": (offset_x, offset_y, zoom_width, zoom_height)})
    picam.configure(cam_config)
    
    # Start the camera stream
    picam.start()
    print("Camera initialized successfully.")
    

    # 2. Define HSV color range for tracking (Example: Bright Green)
    # Adjust these values to match the object you want to track
    lower_color = np.array([0,0,0])
    upper_color = np.array([180, 255, 50])

    # Tracker that assigns persistent IDs to blobs across frames, plus a
    # place to remember each ID's previous position so we can derive direction
    tracker = CentroidTracker(max_disappeared=15, max_match_distance=100)
    previous_positions = {}      # object_id -> (x, y) from the last frame
    last_edge_zone = {}          # object_id -> 'top' or 'bottom', the last edge zone it occupied

    # --- Persistence state: keeps a box/ID drawn for a little while ---
    # --- after an object stops moving, instead of vanishing the     ---
    # --- instant per-frame vertical movement drops below threshold. ---
    last_seen_box = {}           # object_id -> (x, y, w, h) most recent bounding box
    last_seen_direction = {}     # object_id -> last known direction ('Up'/'Down')
    last_moving_frame = {}       # object_id -> frame_count when it last registered movement
    frame_count = 0
    PERSISTENCE_FRAMES = 150      # how many frames to keep showing a stopped object (at 30fps)

    # Fraction of frame height that counts as the "top" / "bottom" entry-exit
    # zones. An object must be seen in one zone and later seen in the
    # opposite zone to register as a full crossing.
    EDGE_ZONE_FRACTION = 0.15

    # Running totals - incremented only on a full traversal (last seen in
    # the top zone, now seen in the bottom zone, or vice versa). Movement
    # that reverses direction before reaching the far zone doesn't count.
    top_to_bottom_count = 0
    bottom_to_top_count = 0

    # --- Merged-blob splitting ---
    # Running estimate of a single bee's contour area, used to guess how
    # many bees are inside an oversized contour so it can be split back
    # apart. Seeded with a fixed default (median single-bee contour area
    # measured from a reference frame, Bees.png, using this same
    # detection pipeline) instead of auto-calibrating from the first few
    # seconds of live detections - at the start of the day there are no
    # bees yet to calibrate from, so that window just expired with no
    # samples and splitting stayed permanently disabled.
    DEFAULT_SINGLE_BEE_AREA = 1858    # median contour area of 7 clean single-bee
                                       # detections in Bees.png at this camera's
                                       # zoom/crop
    single_bee_area = DEFAULT_SINGLE_BEE_AREA
    AREA_EMA_ALPHA = 0.05            # how fast the estimate adapts from the default
    MERGE_TRIGGER_RATIO = 1.8        # contour must be at least this many x the
                                      # estimate before it's even considered as
                                      # possibly more than one bee - comfortably
                                      # above the ~1.2-1.5x a single bee's own
                                      # contour can naturally reach from pose or
                                      # motion (see estimate_bee_count)
    MIN_SINGLE_SAMPLE_RATIO = 0.7    # a contour classified as a single bee but
                                      # smaller than this x the estimate is
                                      # probably a partial/edge-cropped view, not
                                      # a reliable sample of a whole bee's area -
                                      # excluded from the running estimate so it
                                      # can't drag it down over time
    MIN_SPLIT_REGION_RATIO = 0.5     # a split piece smaller than this x the estimate
                                      # is a leg/artifact, not a real second bee
    FRAGMENT_AREA_RATIO = 0.55       # a contour smaller than this x the estimate is
                                      # too small to be a whole bee on its own, so it's
                                      # a candidate to re-join a nearby same-size
                                      # fragment (see group_nearby_contours) - a
                                      # contour at or above this is treated as a
                                      # complete bee and never merged just for being
                                      # close to another one
    FRAGMENT_MERGE_GAP = 15          # pixels; how close two fragment-sized contours
                                      # must be to be considered pieces of one bee

    # --- Six-minute interval logging to SQLite ---
    # We log the *delta* since the last snapshot, not the running total,
    # so each database row represents "how many bees in this interval"
    # rather than a cumulative count.
    db_conn = init_db()
    interval_start_time = datetime.now()
    interval_start_exit_count = top_to_bottom_count
    interval_start_return_count = bottom_to_top_count

    try:
        while True:
            # 3. Capture frame as a NumPy array compatible with OpenCV
            frame = picam.capture_array("main")
            
            # Camera Module 3 outputs RGB/BGR directly via capture_array
            # Convert frame to HSV color space for stable color tracking
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 4. Threshold the image to isolate the targeted color
            # (lower/upper_color already selects low-brightness/black pixels,
            # so a single inRange call is all that's needed here)
            mask = cv2.inRange(hsv, lower_color, upper_color)
            
            # Morphological transformations to filter out tiny background
            # noise. An elliptical kernel and a net-neutral erode/dilate
            # (rather than dilate-more-than-erode) keeps bees that are
            # merely close together from being padded into touching one
            # another - the previous 1-erode/2-dilate combo grew the mask
            # outward on net, which made near-misses turn into actual
            # merges more often than necessary.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # 5. Find boundaries (contours) of all filtered objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Minimum pixel area threshold to prevent tracking tiny artifacts
            MIN_AREA = 500
            
            # Keep every contour big enough to be a real object
            valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

            # Re-join contours that are probably fragments of the same bee
            # (see group_nearby_contours) before deciding how many bees a
            # blob contains, so a momentary mask split doesn't get handed
            # to the tracker as a second object.
            groups = group_nearby_contours(
                valid_contours, FRAGMENT_MERGE_GAP, single_bee_area * FRAGMENT_AREA_RATIO
            )

            # Largest first, then an optional cap so a noisy frame doesn't
            # spam dozens of boxes
            groups.sort(key=lambda g: sum(cv2.contourArea(c) for c in g), reverse=True)
            MAX_OBJECTS = 33
            groups = groups[:MAX_OBJECTS]

            # Map each group's centroid to its bounding box so we can
            # look the box back up after the tracker assigns IDs. Groups
            # much larger than a single bee are assumed to be multiple
            # bees touching/overlapping and get split before being handed
            # to the tracker, instead of becoming one oversized box that
            # swallows several IDs.
            centroids = []
            boxes = {}
            for group in groups:
                group_boxes = [cv2.boundingRect(c) for c in group]
                x = min(bx for bx, by, bw, bh in group_boxes)
                y = min(by for bx, by, bw, bh in group_boxes)
                x2 = max(bx + bw for bx, by, bw, bh in group_boxes)
                y2 = max(by + bh for bx, by, bw, bh in group_boxes)
                w, h = x2 - x, y2 - y
                area = sum(cv2.contourArea(c) for c in group)

                expected_count = estimate_bee_count(area, single_bee_area, MERGE_TRIGGER_RATIO)

                split_boxes = None
                if expected_count > 1:
                    pad = 3
                    rx = max(x - pad, 0)
                    ry = max(y - pad, 0)
                    rw = min(x + w + pad, mask.shape[1]) - rx
                    rh = min(y + h + pad, mask.shape[0]) - ry
                    blob_mask = np.zeros((rh, rw), dtype=np.uint8)
                    cv2.drawContours(blob_mask, group, -1, 255, -1, offset=(-rx, -ry))
                    min_region_area = single_bee_area * MIN_SPLIT_REGION_RATIO
                    split_boxes = split_merged_blob(blob_mask, (rx, ry), expected_count, min_region_area)

                if split_boxes:
                    for (bx, by, bw, bh) in split_boxes:
                        cx = int(bx + bw / 2)
                        cy = int(by + bh / 2)
                        centroids.append((cx, cy))
                        boxes[(cx, cy)] = (bx, by, bw, bh)
                else:
                    center_x = int(x + (w / 2))
                    center_y = int(y + (h / 2))
                    centroids.append((center_x, center_y))
                    boxes[(center_x, center_y)] = (x, y, w, h)

                    if expected_count == 1 and area >= single_bee_area * MIN_SINGLE_SAMPLE_RATIO:
                        # Only refine the estimate from a contour that (a) was
                        # never even considered a possible merge, and (b) isn't
                        # a suspiciously small/partial view - so a failed split
                        # of a real multi-bee blob, or a bee cropped by the
                        # frame edge, can't drag the estimate away from what a
                        # whole bee actually looks like.
                        single_bee_area = (1 - AREA_EMA_ALPHA) * single_bee_area + AREA_EMA_ALPHA * area

            # 6. Update the tracker to get a stable ID -> centroid mapping
            tracked_objects = tracker.update(centroids)
            
            frame_height = frame.shape[0]
            current_positions = {}
            current_edge_zones = {}
            
            for object_id, centroid in tracked_objects.items():
                DRAW_BOX = True
                center_x, center_y = centroid
                current_positions[object_id] = centroid
                
                # --- Crossing count: based on entry zone -> exit zone, ---
                # --- independent of any single frame's direction        ---
                zone = get_edge_zone(center_y, frame_height, EDGE_ZONE_FRACTION)
                if zone != "middle":
                    prev_edge = last_edge_zone.get(object_id)
                    if prev_edge is not None and prev_edge != zone:
                        # Object was last seen in the opposite edge zone and
                        # has now reached this one - that's a full crossing
                        if prev_edge == "top" and zone == "bottom":
                            top_to_bottom_count += 1
                        elif prev_edge == "bottom" and zone == "top":
                            bottom_to_top_count += 1
                    current_edge_zones[object_id] = zone
                elif object_id in last_edge_zone:
                    # Still mid-crossing (in the middle) - remember which
                    # edge zone it came from so the eventual arrival at the
                    # far zone still counts
                    current_edge_zones[object_id] = last_edge_zone[object_id]
                
                # --- Display: draw objects that are currently moving       ---
                # --- vertically, AND objects that stopped recently, so the ---
                # --- box/ID stays visible for a short grace period instead ---
                # --- of disappearing the instant movement drops below      ---
                # --- threshold.                                            ---
                prev_point = previous_positions.get(object_id, centroid)
                v_direction = get_vertical_direction(prev_point, centroid)

                # A real detection this frame (not just a carried-over
                # position from a frame or two of no match) means `boxes`
                # has an entry keyed by this exact centroid.
                detected_this_frame = centroid in boxes
                x, y, w, h = boxes.get(centroid, (center_x - 25, center_y - 25, 50, 50))

                if v_direction is not None:
                    # Currently moving: refresh persistence state
                    last_seen_box[object_id] = (x, y, w, h)
                    last_seen_direction[object_id] = v_direction
                    last_moving_frame[object_id] = frame_count
                    is_stopped = False
                else:
                    # Not moving this frame: only keep drawing if it was
                    # moving recently (within the persistence window)
                    frames_since_moving = frame_count - last_moving_frame.get(object_id, -PERSISTENCE_FRAMES - 1)
                    if frames_since_moving > PERSISTENCE_FRAMES:
                        DRAW_BOX = False
                    if detected_this_frame:
                        # The bee is genuinely still right here, just not
                        # moving fast enough this frame to count as
                        # "moving" - keep last_seen_box in sync so a slow,
                        # gradual walk doesn't leave the box drifting
                        # behind the bee's real position (each individual
                        # frame's step can be under min_movement even
                        # while the bee steadily walks away over several
                        # seconds).
                        last_seen_box[object_id] = (x, y, w, h)
                    else:
                        # No detection at all this frame - fall back to
                        # the last known box rather than the generic
                        # placeholder so the box doesn't jump.
                        x, y, w, h = last_seen_box.get(object_id, (x, y, w, h))
                    is_stopped = True

                # Stopped objects are drawn in a different color (yellow)
                # so it's clear at a glance they're being held, not moving
                box_color = (0, 255, 255) if is_stopped else (0, 255, 0)
                direction_label = last_seen_direction.get(object_id, "?") if is_stopped else v_direction
                status = "Stopped" if is_stopped else direction_label

                # Draw visual tracking boundaries on live preview
                if DRAW_BOX:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Display ID, coordinates, and direction of travel
                label = f"ID {object_id}"
                if DRAW_BOX:
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            previous_positions = current_positions
            last_edge_zone = current_edge_zones

            # Drop persistence state for any object the tracker has fully
            # deregistered (gone past max_disappeared), so old IDs don't
            # linger in these dicts forever
            for stale_id in (last_seen_box.keys() - tracked_objects.keys()):
                last_seen_box.pop(stale_id, None)
                last_seen_direction.pop(stale_id, None)
                last_moving_frame.pop(stale_id, None)

            frame_count += 1
            
            # Draw the entry/exit zone boundaries for reference
            top_line_y = int(frame_height * EDGE_ZONE_FRACTION)
            bottom_line_y = int(frame_height * (1 - EDGE_ZONE_FRACTION))
            cv2.line(frame, (0, top_line_y), (frame.shape[1], top_line_y), (255, 0, 0), 1)
            cv2.line(frame, (0, bottom_line_y), (frame.shape[1], bottom_line_y), (255, 0, 0), 1)
            
            metadata = picam.capture_metadata()
            lens_pos = metadata.get("LensPosition",0.0)
            
            # Read the maximum sensor crop limits
            scaler_max = picam.camera_properties.get("ScalerCropMaximum")
            
            # Show how many objects are currently being tracked, plus the
            # running top-to-bottom / bottom-to-top crossing counts
            cv2.putText(frame, f"Objects tracked: {len(tracked_objects)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Return count: {bottom_to_top_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Exit count: {top_to_bottom_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # --- Every 6 minutes, log the interval's tally to SQLite ---
            now = datetime.now()
            elapsed = (now - interval_start_time).total_seconds()
            if elapsed >= INTERVAL_SECONDS:
                exits_this_interval = top_to_bottom_count - interval_start_exit_count
                returns_this_interval = bottom_to_top_count - interval_start_return_count
                log_interval(db_conn, exits_this_interval, returns_this_interval, timestamp=now)
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Logged interval - Exits: {exits_this_interval}, Returns: {returns_this_interval}")

                interval_start_time = now
                interval_start_exit_count = top_to_bottom_count
                interval_start_return_count = bottom_to_top_count

            # 7. Display the final tracked output window
            cv2.imshow("Camera Module 3 Object Tracking", frame)
            
            # Press 'q' key to break the tracking loop cleanly
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # If after the CUTOFF time, then exit
            now = datetime.now().time()
            if now >= CUTOFF:
                print("Cutoff time reached, exiting.")
                break
                
    finally:
        # Log whatever partial interval had accumulated before shutdown,
        # so a stopped/interrupted run doesn't silently lose data.
        final_exits = top_to_bottom_count - interval_start_exit_count
        final_returns = bottom_to_top_count - interval_start_return_count
        if final_exits or final_returns:
            log_interval(db_conn, final_exits, final_returns)
            print(f"Logged final partial interval - Exits: {final_exits}, Returns: {final_returns}")
        db_conn.close()

        # 8. Clean environment shutdown
        picam.stop()
        cv2.destroyAllWindows()
        print("Camera and windows closed cleanly.")

if __name__ == "__main__":
    main()
