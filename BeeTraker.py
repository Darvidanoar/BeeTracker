import cv2
import numpy as np
import sqlite3
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls
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

        # Greedily match smallest distances first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
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
    PERSISTENCE_FRAMES = 60      # how many frames to keep showing a stopped object (at 30fps)

    # Fraction of frame height that counts as the "top" / "bottom" entry-exit
    # zones. An object must be seen in one zone and later seen in the
    # opposite zone to register as a full crossing.
    EDGE_ZONE_FRACTION = 0.15

    # Running totals - incremented only on a full traversal (last seen in
    # the top zone, now seen in the bottom zone, or vice versa). Movement
    # that reverses direction before reaching the far zone doesn't count.
    top_to_bottom_count = 0
    bottom_to_top_count = 0

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
            
            # Morphological transformations to filter out tiny background noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # 5. Find boundaries (contours) of all filtered objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Minimum pixel area threshold to prevent tracking tiny artifacts
            MIN_AREA = 500
            
            # Keep every contour big enough to be a real object, largest first
            valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            
            # Optional cap so a noisy frame doesn't spam dozens of boxes
            MAX_OBJECTS = 33
            valid_contours = valid_contours[:MAX_OBJECTS]
            
            # Map each contour's centroid to its bounding box so we can
            # look the box back up after the tracker assigns IDs
            centroids = []
            boxes = {}
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = int(x + (w / 2))
                center_y = int(y + (h / 2))
                centroids.append((center_x, center_y))
                boxes[(center_x, center_y)] = (x, y, w, h)
            
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
                    DRAW_BOX = True
                    # Use the last known box/direction rather than the
                    # (possibly stale/absent) current detection so the box
                    # doesn't jump if the contour was momentarily lost
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
                if DRAW_BOX:
                    label = f"ID {object_id}"
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
