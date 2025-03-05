import cv2
import os

# Path to your gt file and image folder
gt_file = '2907/mot17/MOT17/train/MOT17-02-FRCNN/gt/gt.txt'
img_folder = '2907/mot17/MOT17/train/MOT17-02-FRCNN/img1/'

# Choose a frame to validate (e.g., frame number 1)
frame_to_check = 1

# Read gt.txt and filter entries for the chosen frame
gt_entries = []
with open(gt_file, 'r') as f:
    for line in f:
        # Assuming the line format: frame, id, x, y, w, h, score, class, visibility
        parts = line.strip().split(',')
        if len(parts) < 9:
            continue
        frame_num = int(parts[0])
        if frame_num == frame_to_check:
            obj_id = int(parts[1])
            x = int(float(parts[2]))
            y = int(float(parts[3]))
            w = int(float(parts[4]))
            h = int(float(parts[5]))
            score = float(parts[6])
            obj_class = int(parts[7])
            visibility = float(parts[8])
            gt_entries.append((obj_id, x, y, w, h, score, obj_class, visibility))

# Load the corresponding image (adjust the file naming as needed)
# Assuming images are named like '000001.jpg'
img_name = f"{frame_to_check:06d}.jpg"
img_path = os.path.join(img_folder, img_name)
img = cv2.imread(img_path)

if img is None:
    print(f"Image not found: {img_path}")
else:
    # Draw each GT bounding box on the image
    for entry in gt_entries:
        obj_id, x, y, w, h, score, obj_class, visibility = entry
        # Compute bottom-right corner
        x_max, y_max = x + w, y + h
        # Draw the bounding box (red)
        cv2.rectangle(img, (x, y), (x_max, y_max), (0, 0, 255), 2)
        # Put a label (ID or class)
        label = f"ID:{obj_id} Cls:{obj_class}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the image
    cv2.imshow('GT Validation', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

