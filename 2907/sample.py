import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import nn
from PIL import Image, ImageFilter
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            
            # Flip bbox coordinates
            if "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
                
        return image, target


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)
        
    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            # Convert to PIL for color jitter
            image_pil = F.to_pil_image(image)
            image_pil = self.transform(image_pil)
            image = F.to_tensor(image_pil)
        else:
            image = self.transform(image)
        return image, target


class RandomZoomOut(object):
    def __init__(self, fill=[0, 0, 0], min_scale=1.0, max_scale=1.5, prob=0.5):
        self.fill = fill
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
            
        orig_height, orig_width = image.shape[-2:] if isinstance(image, torch.Tensor) else (image.height, image.width)
        
        scale = random.uniform(self.min_scale, self.max_scale)
        new_height = int(orig_height * scale)
        new_width = int(orig_width * scale)
        
        if isinstance(image, torch.Tensor):
            # Create a new tensor with the fill color
            new_image = torch.ones((image.shape[0], new_height, new_width), 
                                  dtype=image.dtype) * torch.tensor(self.fill, 
                                  dtype=image.dtype).view(3, 1, 1)
            
            # Calculate paste coordinates
            top = (new_height - orig_height) // 2
            left = (new_width - orig_width) // 2
            
            # Paste the original image
            new_image[:, top:top+orig_height, left:left+orig_width] = image
            
            # Update boxes
            if "boxes" in target and len(target["boxes"]):
                boxes = target["boxes"]
                boxes[:, [0, 2]] += left
                boxes[:, [1, 3]] += top
                target["boxes"] = boxes
        else:
            # For PIL images
            new_image = Image.new("RGB", (new_width, new_height), tuple(self.fill))
            top = (new_height - orig_height) // 2
            left = (new_width - orig_width) // 2
            new_image.paste(image, (left, top))
            
            # Update boxes
            if "boxes" in target and len(target["boxes"]):
                boxes = target["boxes"]
                boxes[:, [0, 2]] += left
                boxes[:, [1, 3]] += top
                target["boxes"] = boxes
                
        return new_image, target


class RandomPhotometricDistort(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
            
        # Apply a series of photometric distortions
        distortions = []
        
        # Random brightness
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            distortions.append(lambda img: F.adjust_brightness(img, factor))
            
        # Random contrast
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            distortions.append(lambda img: F.adjust_contrast(img, factor))
            
        # Random saturation
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            distortions.append(lambda img: F.adjust_saturation(img, factor))
            
        # Random gaussian blur
        if random.random() < 0.2:
            radius = random.uniform(0.1, 2.0)
            distortions.append(lambda img: img.filter(ImageFilter.GaussianBlur(radius=radius)))
            
        random.shuffle(distortions)
        
        # Apply distortions
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL
            image_pil = F.to_pil_image(image)
            for d in distortions:
                image_pil = d(image_pil)
            image = F.to_tensor(image_pil)
        else:
            for d in distortions:
                image = d(image)
                
        return image, target


class RandomCrop(object):
    def __init__(self, min_scale=0.7, max_scale=1.0, prob=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
            
        orig_height, orig_width = image.shape[-2:] if isinstance(image, torch.Tensor) else (image.height, image.width)
        
        # Random crop size as a percentage of original size
        scale = random.uniform(self.min_scale, self.max_scale)
        crop_height = int(orig_height * scale)
        crop_width = int(orig_width * scale)
        
        # Random crop position
        top = random.randint(0, orig_height - crop_height)
        left = random.randint(0, orig_width - crop_width)
        
        if isinstance(image, torch.Tensor):
            # Crop the tensor
            image = image[:, top:top+crop_height, left:left+crop_width]
        else:
            # Crop the PIL image
            image = F.crop(image, top, left, crop_height, crop_width)
            
        # Update boxes
        if "boxes" in target and len(target["boxes"]):
            boxes = target["boxes"].clone()
            
            # Adjust boxes to crop area
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top
            
            # Clip boxes to crop boundaries
            boxes[:, 0].clamp_(min=0, max=crop_width)
            boxes[:, 1].clamp_(min=0, max=crop_height)
            boxes[:, 2].clamp_(min=0, max=crop_width)
            boxes[:, 3].clamp_(min=0, max=crop_height)
            
            # Remove boxes that are out of bounds or too small
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            
            # Update target with filtered boxes
            target["boxes"] = boxes[keep]
            for k in ["labels", "area", "iscrowd"]:
                if k in target:
                    target[k] = target[k][keep]
                
        return image, target
    
    


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    
    if train:
        # Data augmentation for training
        # Random color jitter with 50% probability
        transforms.append(ColorJitter(brightness=0.2, 
                                     contrast=0.2, 
                                     saturation=0.2, 
                                     hue=0.1))
        
        # Random horizontal flip with 50% probability
        transforms.append(RandomHorizontalFlip(0.5))
        
        # Random crop with 30% probability
        transforms.append(RandomCrop(min_scale=0.7, max_scale=1.0, prob=0.3))
        
        # Random zoom out with 30% probability
        transforms.append(RandomZoomOut(fill=[0, 0, 0], 
                                      min_scale=1.0, 
                                      max_scale=1.3, 
                                      prob=0.3))
        
        # Random photometric distortion with 30% probability
        transforms.append(RandomPhotometricDistort(prob=0.3))
    
    # Add normalization - using ImageNet mean and std
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]))
    
    return Compose(transforms)



# tracker_utils.py
import numpy as np
import torch
import cv2
from track import Sort

def detection_to_format(detections, threshold=0.5):
    """Convert detection model output to tracker input format"""
    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    
    # Filter by confidence threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    
    # Format for tracker: [x1, y1, x2, y2, score]
    dets = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1)
    return dets

def run_tracking_on_video(model, video_path, output_path=None, threshold=0.5, display=True):
    """
    Run object detection and tracking on a video
    
    Args:
        model: Detection model
        video_path: Path to input video
        output_path: Path to save output video (optional)
        threshold: Detection confidence threshold
        display: Whether to show the video during processing
        
    Returns:
        tracks_by_frame: Dictionary of tracks by frame
    """
    # Initialize tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output_path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    device = next(model.parameters()).device
    frame_count = 0
    tracks_by_frame = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB for model input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(rgb_frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
        
        # Run detection
        with torch.no_grad():
            detections = model(img_tensor)[0]
        
        # Convert detections to tracker format
        dets = detection_to_format(detections, threshold)
        
        # Update tracker
        tracks = tracker.update(dets)
        
        # Store tracks
        tracks_by_frame[frame_count] = tracks
        
        # Draw results
        for d in tracks:
            bbox = d[:4].astype(np.int32)
            track_id = int(d[4])
            score = d[5]
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw ID and score
            cv2.putText(frame, f"ID: {track_id}, {score:.2f}", 
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        # Display and save
        if display:
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        if output_path:
            out.write(frame)
            
        frame_count += 1
    
    # Clean up
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    return tracks_by_frame



# evaluation.py
import numpy as np
from collections import defaultdict

def compute_iou(boxA, boxB):
    """Compute IOU between two boxes in format [x1,y1,x2,y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_mota(gt_tracks, pred_tracks, iou_threshold=0.5):
    """
    Calculate Multi-Object Tracking Accuracy (MOTA)
    
    Args:
        gt_tracks: Ground truth tracks by frame {frame_id: [[x1,y1,x2,y2,track_id], ...]}
        pred_tracks: Predicted tracks by frame {frame_id: [[x1,y1,x2,y2,track_id,score], ...]}
        iou_threshold: IOU threshold for a match
        
    Returns:
        mota: MOTA score
        fn: Number of false negatives
        fp: Number of false positives
        id_switches: Number of ID switches
    """
    fn = 0  # False negatives
    fp = 0  # False positives
    id_switches = 0  # ID switches
    
    # Match track IDs between frames
    matches = {}  # gt_id -> pred_id mapping
    
    # Process each frame
    for frame_id in sorted(gt_tracks.keys()):
        gt_boxes = gt_tracks.get(frame_id, [])
        pred_boxes = pred_tracks.get(frame_id, [])
        
        # Create IOU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                iou_matrix[i, j] = compute_iou(gt[:4], pred[:4])
        
        # Match boxes based on IOU
        matched_gt = set()
        matched_pred = set()
        
        # Sort IOUs in descending order
        indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]
        
        for i, j in indices:
            if i in matched_gt or j in matched_pred:
                continue
                
            if iou_matrix[i, j] >= iou_threshold:
                matched_gt.add(i)
                matched_pred.add(j)
                
                gt_id = gt_boxes[i][4]
                pred_id = pred_boxes[j][4]
                
                # Check for ID switch
                if gt_id in matches and matches[gt_id] != pred_id:
                    id_switches += 1
                    
                matches[gt_id] = pred_id
        
        # Count false positives and negatives
        fp += len(pred_boxes) - len(matched_pred)
        fn += len(gt_boxes) - len(matched_gt)
    
    # Calculate MOTA
    total_gt = sum(len(boxes) for boxes in gt_tracks.values())
    mota = 1 - (fn + fp + id_switches) / max(1, total_gt)
    
    return mota, fn, fp, id_switches

def calculate_motp(gt_tracks, pred_tracks, iou_threshold=0.5):
    """
    Calculate Multi-Object Tracking Precision (MOTP)
    
    Args:
        gt_tracks: Ground truth tracks by frame {frame_id: [[x1,y1,x2,y2,track_id], ...]}
        pred_tracks: Predicted tracks by frame {frame_id: [[x1,y1,x2,y2,track_id,score], ...]}
        iou_threshold: IOU threshold for a match
        
    Returns:
        motp: MOTP score
    """
    total_iou = 0
    total_matches = 0
    
    # Process each frame
    for frame_id in sorted(gt_tracks.keys()):
        gt_boxes = gt_tracks.get(frame_id, [])
        pred_boxes = pred_tracks.get(frame_id, [])
        
        # Create IOU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                iou_matrix[i, j] = compute_iou(gt[:4], pred[:4])
        
        # Match boxes based on IOU
        matched_gt = set()
        matched_pred = set()
        
        # Sort IOUs in descending order
        indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]
        
        for i, j in indices:
            if i in matched_gt or j in matched_pred:
                continue
                
            if iou_matrix[i, j] >= iou_threshold:
                matched_gt.add(i)
                matched_pred.add(j)
                total_iou += iou_matrix[i, j]
                total_matches += 1
    
    # Calculate MOTP
    motp = total_iou / max(1, total_matches)
    
    return motp

def calculate_id_metrics(gt_tracks, pred_tracks, iou_threshold=0.5):
    """
    Calculate ID-related metrics (ID Precision, ID Recall, ID F1)
    
    Args:
        gt_tracks: Ground truth tracks by frame {frame_id: [[x1,y1,x2,y2,track_id], ...]}
        pred_tracks: Predicted tracks by frame {frame_id: [[x1,y1,x2,y2,track_id,score], ...]}
        iou_threshold: IOU threshold for a match
        
    Returns:
        idp: ID Precision
        idr: ID Recall
        idf1: ID F1 Score
    """
    # Extract all track IDs
    gt_ids = set()
    for boxes in gt_tracks.values():
        for box in boxes:
            gt_ids.add(box[4])
            
    pred_ids = set()
    for boxes in pred_tracks.values():
        for box in boxes:
            pred_ids.add(box[4])
    
    # Count true positives, false positives, false negatives by track ID
    tp_ids = defaultdict(int)  # Correctly identified detections by ID
    fp_ids = defaultdict(int)  # False detections by ID
    fn_ids = defaultdict(int)  # Missed detections by ID
    
    # Process each frame
    for frame_id in sorted(gt_tracks.keys()):
        gt_boxes = gt_tracks.get(frame_id, [])
        pred_boxes = pred_tracks.get(frame_id, [])
        
        # Create IOU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                iou_matrix[i, j] = compute_iou(gt[:4], pred[:4])
        
        # Match boxes based on IOU
        matched_gt = set()
        matched_pred = set()
        
        # Sort IOUs in descending order
        indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]
        
        for i, j in indices:
            if i in matched_gt or j in matched_pred:
                continue
                
            if iou_matrix[i, j] >= iou_threshold:
                matched_gt.add(i)
                matched_pred.add(j)
                
                gt_id = gt_boxes[i][4]
                pred_id = pred_boxes[j][4]
                
                tp_ids[gt_id] += 1
        
        # Count false positives and negatives
        for i, gt in enumerate(gt_boxes):
            if i not in matched_gt:
                fn_ids[gt[4]] += 1
                
        for j, pred in enumerate(pred_boxes):
            if j not in matched_pred:
                fp_ids[pred[4]] += 1
    
    # Calculate ID metrics
    tp_sum = sum(tp_ids.values())
    fp_sum = sum(fp_ids.values())
    fn_sum = sum(fn_ids.values())
    
    idp = tp_sum / max(1, tp_sum + fp_sum)  # ID Precision
    idr = tp_sum / max(1, tp_sum + fn_sum)  # ID Recall
    idf1 = 2 * idp * idr / max(1e-6, idp + idr)  # ID F1 Score
    
    return idp, idr, idf1


import os
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from collections import defaultdict

# Import our custom modules
# Assume we've saved our custom transforms.py, track.py, tracker_utils.py, and evaluation.py files

# First, let's create a function to extract frames from video sequences
def extract_frames_from_video(video_path, output_dir, frame_interval=1):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every n-th frame
        
    Returns:
        List of paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
            
        frame_count += 1
        
    cap.release()
    print(f"Extracted {len(extracted_frames)} frames from {video_path}")
    return extracted_frames

# Now let's use our model with tracking capabilities
def get_detection_model(num_classes):
    """
    Get a Faster R-CNN detection model with FPN
    
    Args:
        num_classes: Number of object classes to detect
        
    Returns:
        Detection model
    """
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Improve NMS settings
    model.roi_heads.nms_thresh = 0.3
    
    # Additional model customization
    # Lower detection threshold for higher recall
    model.roi_heads.score_thresh = 0.3
    
    return model

# Create a video-based dataset for inference
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, transforms=None, max_frames=None):
        self.video_path = video_path
        self.transforms = transforms
        self.max_frames = max_frames
        
        # Extract frames
        self.cap = cv2.VideoCapture(video_path)
        self.frames = []
        
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame)
            frame_count += 1
            
        self.cap.release()
        
    def __len__(self):
        return len(self.frames)
        
    def __getitem__(self, idx):
        # Convert to PIL Image
        img = Image.fromarray(self.frames[idx])
        
        if self.transforms:
            img = self.transforms(img)
            
        return img

# Create a visualization function for tracking results
def visualize_tracking_results(frame, tracks, output_path=None):
    """
    Visualize tracking results on a frame
    
    Args:
        frame: Input frame (numpy array, BGR format)
        tracks: Tracking results in format [[x1,y1,x2,y2,track_id,score], ...]
        output_path: Path to save the visualization (optional)
        
    Returns:
        Visualization frame
    """
    vis_frame = frame.copy()
    
    # Generate random colors for track IDs
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
    
    for track in tracks:
        bbox = track[:4].astype(np.int32)
        track_id = int(track[4])
        score = track[5]
        
        # Get color for this track ID
        color = tuple(map(int, colors[track_id % len(colors)]))
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw ID and score
        label = f"ID: {track_id}, {score:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(vis_frame, (bbox[0], bbox[1] - text_size[1] - 10), 
                     (bbox[0] + text_size[0], bbox[1]), color, -1)
        cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_frame)
        
    return vis_frame

# Function to run the complete tracking pipeline
def run_tracking_pipeline(model, video_path, output_dir, detection_threshold=0.5):
    """
    Run the complete tracking pipeline on a video
    
    Args:
        model: Detection model
        video_path: Path to input video
        output_dir: Directory to save results
        detection_threshold: Detection confidence threshold
        
    Returns:
        tracking_results: Dictionary of tracking results by frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracker
    from track import Sort
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    output_video_path = os.path.join(output_dir, "tracking_results.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process video
    device = next(model.parameters()).device
    frame_count = 0
    tracking_results = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB for model input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(rgb_frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
        
        # Run detection
        model.eval()
        with torch.no_grad():
            detections = model(img_tensor)[0]
        
        # Convert detections to tracker format
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        keep = scores >= detection_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        
        # Format for tracker: [x1, y1, x2, y2, score]
        dets = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1)
        
        # Update tracker
        tracks = tracker.update(dets)
        
        # Store tracks
        tracking_results[frame_count] = tracks
        
        # Visualize and save frame
        vis_frame = visualize_tracking_results(frame, tracks)
        out.write(vis_frame)
        
        # Save keyframes
        if frame_count % 20 == 0:
            keyframe_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(keyframe_path, vis_frame)
            
        frame_count += 1
        
        # Display progress
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")
    
    # Clean up
    cap.release()
    out.release()
    
    # Save tracking results
    results_path = os.path.join(output_dir, "tracking_results.npy")
    np.save(results_path, tracking_results)
    
    print(f"Tracking complete. Results saved to {output_dir}")
    return tracking_results

# Function to evaluate tracking performance
def evaluate_tracking_performance(tracking_results, ground_truth=None):
    """
    Evaluate tracking performance
    
    Args:
        tracking_results: Dictionary of tracking results by frame
        ground_truth: Dictionary of ground truth tracks by frame (optional)
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    from evaluation import calculate_mota, calculate_motp, calculate_id_metrics
    
    # If no ground truth is provided, we can only evaluate detection metrics
    if ground_truth is None:
        print("No ground truth provided. Computing detection statistics only.")
        
        # Count detections per frame
        detections_per_frame = [len(tracks) for tracks in tracking_results.values()]
        avg_detections = np.mean(detections_per_frame)
        
        # Count unique track IDs
        unique_ids = set()
        for tracks in tracking_results.values():
            for track in tracks:
                unique_ids.add(int(track[4]))
        
        # Compute average track length
        track_lengths = defaultdict(int)
        for frame_id, tracks in tracking_results.items():
            for track in tracks:
                track_id = int(track[4])
                track_lengths[track_id] += 1
        
        avg_track_length = np.mean(list(track_lengths.values())) if track_lengths else 0
        
        metrics = {
            "avg_detections_per_frame": avg_detections,
            "num_unique_tracks": len(unique_ids),
            "avg_track_length": avg_track_length
        }
    else:
        # Compute standard MOT metrics
        mota, fn, fp, id_switches = calculate_mota(ground_truth, tracking_results)
        motp = calculate_motp(ground_truth, tracking_results)
        idp, idr, idf1 = calculate_id_metrics(ground_truth, tracking_results)
        
        metrics = {
            "mota": mota,
            "motp": motp,
            "false_negatives": fn,
            "false_positives": fp,
            "id_switches": id_switches,
            "id_precision": idp,
            "id_recall": idr,
            "id_f1": idf1
        }
    
    # Print metrics
    print("\nTracking Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics

# Main execution flow for MOT17 dataset
def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load and prepare dataset
    dataset = MOT17ObjDetect('train', get_transform(train=True))
    dataset_test = MOT17ObjDetect('test', get_transform(train=False))
    
    # Create data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    # Create model
    model = get_detection_model(dataset.num_classes)
    model.to(device)
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train model
    num_epochs = 10
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate and save checkpoint
        if epoch % 2 == 0 or epoch == num_epochs:
            # Save checkpoint
            checkpoint_path = f"checkpoints/model_epoch_{epoch}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Evaluate on test set
            print(f"Evaluating after epoch {epoch}...")
            evaluate_and_write_result_files(model, data_loader_test)
    
    # Run tracking on a test video
    test_video = "test_video.mp4"  # Replace with actual test video
    tracking_results = run_tracking_pipeline(model, test_video, "tracking_output")
    
    # Evaluate tracking performance
    evaluate_tracking_performance(tracking_results)
    
    print("Training and evaluation complete!")

# If this file is run as a script, execute the main function
if __name__ == "__main__":
    main()