import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from tqdm import tqdm

checkpoints_path = r"2907/checkpoints"
steps_per_checkpoint = 100  # Save a checkpoint every 100 training steps

##############################################
# 1. DATA PREPROCESSING & AUGMENTATION
##############################################

class MOTDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for a (potentially small) subset of MOT17 or similar data.
    """
    def __init__(self, 
                 root_dir, 
                 start_frame=1, 
                 end_frame=300, 
                 transforms=None, 
                 min_visibility=0.3, 
                 only_person=True):
        super().__init__()
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "img1")
        self.ann_file = os.path.join(root_dir, "gt", "gt.txt")
        self.transforms = transforms
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.min_visibility = min_visibility
        self.only_person = only_person
        
        # Read the annotation file
        ann_cols = ["frame","id","x","y","w","h","conf","class","visibility"]
        df = pd.read_csv(self.ann_file, header=None, names=ann_cols)
        
        # Keep only frames in desired range
        df = df[(df["frame"] >= self.start_frame) & (df["frame"] <= self.end_frame)]
        
        # Filter out low-visibility or low-confidence boxes
        df = df[df["visibility"] >= self.min_visibility]
        df = df[df["conf"] == 1]  # keep only confident = 1
        
        # (Optional) Keep only 'person' class if needed
        if self.only_person:
            # Typically 'class=1' in MOT indicates pedestrian
            df = df[df["class"] == 1]
        
        # Group by frame
        self.frames_data = df.groupby("frame")
        
        # Sorted list of valid frames
        self.valid_frames = sorted(self.frames_data.groups.keys())

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, idx):
        frame_num = self.valid_frames[idx]
        # Build image path
        img_filename = f"{frame_num:06d}.jpg"
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Gather bounding boxes for this frame
        df_frame = self.frames_data.get_group(frame_num)
        
        boxes = []
        labels = []
        for _, row in df_frame.iterrows():
            xmin = row["x"]
            ymin = row["y"]
            xmax = xmin + row["w"]
            ymax = ymin + row["h"]
            boxes.append([xmin, ymin, xmax, ymax])
            # For MOT, label = 1 for person
            labels.append(1)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Additional fields
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "frame_num": torch.tensor([frame_num])  # helps with tracking logic
        }
        
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target

# Custom transforms for data augmentation & normalization
class CustomTransforms:
    def __init__(self, train=True):
        self.train = train
        # Basic transforms for all:
        self.to_tensor = T.ToTensor()
        # We'll define some random transforms for data augmentation (cropping, flipping, color jitter)
        self.augmentations = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomResizedCrop(size=(480, 640), scale=(0.8, 1.0)),  # random crop
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
        # Normalization
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __call__(self, image, target):
        # Convert PIL -> Tensor
        image = self.to_tensor(image)
        
        if self.train:
            # Augment only if training
            image = self.augmentations(image)
        
        # Basic normalization
        image = self.normalize(image)
        
        return image, target


##############################################
# 2. MODEL DEVELOPMENT (FASTER R-CNN)
##############################################

def get_faster_rcnn_model(num_classes=2):
    """
    Pretrained Faster R-CNN with ResNet-50 FPN as backbone.
    We replace the final predictor with (num_classes) outputs 
    (1 background + N classes).
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


##############################################
# 3. TEMPORAL CONSISTENCY + ADAPTIVE TRACKING
##############################################

class NaiveSORTTracker:
    """
    Minimalistic tracker using a distance-based or IoU-based approach 
    (inspired by SORT) with optional 'KalmanFilter' stubs or bounding-box smoothing.
    """
    def __init__(self, max_distance=100.0):
        self.next_id = 1
        self.tracks = {}  # id -> (bbox, velocity, last_frame)
        self.max_distance = max_distance

    def iou(self, boxA, boxB):
        """Compute IoU between two boxes (x1,y1,x2,y2)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, frame_num, detections):
        """
        detections: list of (bbox, score)
        Return: list of (track_id, bbox).
        """
        assigned = {}
        track_updates = {}
        used_detections = set()

        # For each active track, attempt to match with a detection
        for t_id, (t_box, velocity, last_fr) in self.tracks.items():
            best_iou = 0
            best_det_idx = None
            for i, (det_box, score) in enumerate(detections):
                if i in used_detections:
                    continue
                iou_val = self.iou(t_box, det_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det_idx = i
            # We set a threshold for iou to maintain track
            if best_iou > 0.3 and best_det_idx is not None:
                # update track
                assigned[t_id] = (best_det_idx, best_iou)
                used_detections.add(best_det_idx)

        # Now update existing tracks with assigned detections
        for t_id, (det_idx, iou_val) in assigned.items():
            det_box, score = detections[det_idx]
            # naive velocity update
            old_box = self.tracks[t_id][0]
            vx = (det_box[0] - old_box[0])  
            vy = (det_box[1] - old_box[1])
            new_velocity = (vx, vy)
            track_updates[t_id] = (det_box, new_velocity, frame_num)

        # Any detection not assigned => new track
        for i, (det_box, score) in enumerate(detections):
            if i not in used_detections:
                track_updates[self.next_id] = (det_box, (0,0), frame_num)
                self.next_id += 1

        # Overwrite old tracks with new updated ones
        self.tracks.update(track_updates)

        # Build output (id, bbox)
        results = []
        for t_id, (t_box, vel, lf) in self.tracks.items():
            if lf == frame_num:  # updated on this frame
                results.append((t_id, t_box))
        
        return results

##############################################
# 7. ADDING PERFORMANCE COMPARISON
##############################################

def compare_with_baseline(model, tracker, dataset, device, threshold=0.5):
    """
    Compare the current model and tracker with a baseline approach.
    
    For baseline, we'll use:
    1. A simple tracker without temporal consistency
    2. Raw Faster R-CNN detections without tracking
    
    Args:
        model: Trained Faster R-CNN model
        tracker: Our tracker implementation
        dataset: Validation dataset
        device: Device to run computations on
        threshold: Detection confidence threshold
        
    Returns:
        Dictionary of comparison metrics
    """
    # Initialize metrics
    metrics = {
        'our_approach': {
            'detections': 0,
            'true_positives': 0,
            'id_switches': 0
        },
        'no_tracking': {
            'detections': 0,
            'true_positives': 0,
        },
        'simple_iou_tracker': {
            'detections': 0,
            'true_positives': 0,
            'id_switches': 0
        }
    }
    
    # Create a simpler tracker for comparison (just using IoU without velocity)
    class SimpleIOUTracker:
        def __init__(self, iou_threshold=0.5):
            self.next_id = 1
            self.tracks = {}  # id -> bbox
            self.iou_threshold = iou_threshold
            
        def update(self, frame_num, detections):
            assigned = {}
            track_updates = {}
            used_detections = set()
            
            # Match existing tracks with new detections
            for t_id, t_box in self.tracks.items():
                best_iou = 0
                best_det_idx = None
                for i, (det_box, score) in enumerate(detections):
                    if i in used_detections:
                        continue
                    iou_val = iou(t_box, det_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_det_idx = i
                        
                if best_iou > self.iou_threshold and best_det_idx is not None:
                    det_box, _ = detections[best_det_idx]
                    track_updates[t_id] = det_box
                    used_detections.add(best_det_idx)
            
            # Any detection not assigned => new track
            for i, (det_box, score) in enumerate(detections):
                if i not in used_detections:
                    track_updates[self.next_id] = det_box
                    self.next_id += 1
                    
            # Update tracks
            self.tracks = track_updates
            
            # Build output (id, bbox)
            results = [(t_id, t_box) for t_id, t_box in self.tracks.items()]
            return results
    
    # Create a simple IoU tracker
    simple_tracker = SimpleIOUTracker()
    
    # Track identity switches for both trackers
    last_matched_ids = {'our_approach': {}, 'simple_iou_tracker': {}}
    
    # Process each frame
    for idx in tqdm(range(len(dataset)), desc="Comparing with baseline"):
        img, target = dataset[idx]
        frame_num = target["frame_num"].item()
        gt_boxes = target["boxes"].numpy()
        
        model.eval()
        with torch.no_grad():
            # Detect objects in the frame
            detection = model([img.to(device)])[0]
        
        # Process raw detections
        pred_boxes = detection["boxes"].cpu().numpy()
        scores = detection["scores"].cpu().numpy()
        
        # Filter by threshold
        high_conf_indices = np.where(scores > threshold)[0]
        pred_boxes = pred_boxes[high_conf_indices]
        scores = scores[high_conf_indices]
        
        # Create list of (bbox, score) for trackers
        detections = [(box, score) for box, score in zip(pred_boxes, scores)]
        
        # 1. Our approach: Update our tracker
        our_tracks = tracker.update(frame_num, detections)
        our_track_boxes = [box for _, box in our_tracks]
        our_track_ids = [tid for tid, _ in our_tracks]
        
        # 2. Simple IOU tracker: Update simple tracker
        simple_tracks = simple_tracker.update(frame_num, detections)
        simple_track_boxes = [box for _, box in simple_tracks]
        simple_track_ids = [tid for tid, _ in simple_tracks]
        
        # 3. No tracking: Just use raw detections
        no_track_boxes = pred_boxes
        
        # Count detections
        metrics['our_approach']['detections'] += len(our_track_boxes)
        metrics['simple_iou_tracker']['detections'] += len(simple_track_boxes)
        metrics['no_tracking']['detections'] += len(no_track_boxes)
        
        # Evaluate against ground truth
        for approach, boxes, ids in [
            ('our_approach', our_track_boxes, our_track_ids),
            ('simple_iou_tracker', simple_track_boxes, simple_track_ids)
        ]:
            # Match with ground truth
            for gt_idx, gt_box in enumerate(gt_boxes):
                best_iou = 0.5  # IoU threshold
                best_match = -1
                
                for j, box in enumerate(boxes):
                    iou_val = iou(gt_box, box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_match = j
                
                if best_match >= 0:
                    # We have a match (true positive)
                    metrics[approach]['true_positives'] += 1
                    
                    # Check for ID switch
                    gt_key = tuple(gt_box.tolist())
                    if gt_key in last_matched_ids[approach]:
                        if last_matched_ids[approach][gt_key] != ids[best_match]:
                            metrics[approach]['id_switches'] += 1
                    
                    # Update last matched ID
                    last_matched_ids[approach][gt_key] = ids[best_match]
        
        # For no tracking approach (just detections)
        for gt_box in gt_boxes:
            best_iou = 0.5
            for box in no_track_boxes:
                iou_val = iou(gt_box, box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    metrics['no_tracking']['true_positives'] += 1
                    break
    
    # Calculate comparative metrics
    for approach in metrics:
        metrics[approach]['precision'] = metrics[approach]['true_positives'] / max(1, metrics[approach]['detections'])
        metrics[approach]['recall'] = metrics[approach]['true_positives'] / max(1, len(dataset) * 5)  # Assuming ~5 objects per frame
        
        if approach != 'no_tracking':
            metrics[approach]['id_switch_rate'] = metrics[approach]['id_switches'] / len(dataset)
    
    return metrics

def compute_average_precision(detections, ground_truths, iou_threshold=0.5):
    """
    Compute Average Precision for a specific class.
    
    Args:
        detections: List of (boxes, scores) tuples for each image
        ground_truths: List of ground truth boxes for each image
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Average Precision score
    """
    # Total number of ground truth boxes across all images
    total_gt = sum(len(gt) for gt in ground_truths)
    
    if total_gt == 0:
        return 0.0
    
    # Flatten all detections across images
    all_scores = []
    all_matched = []
    
    # Keep track of which ground truth boxes have been matched
    gt_matched = []
    for i in range(len(ground_truths)):
        gt_matched.append(np.zeros(len(ground_truths[i]), dtype=bool))
    
    # Process each image
    for i in range(len(detections)):
        boxes, scores = detections[i]
        gt_boxes = ground_truths[i]
        
        for j in range(len(boxes)):
            all_scores.append(scores[j])
            
            # Check if this detection matches any ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for k in range(len(gt_boxes)):
                if gt_matched[i][k]:
                    continue  # Skip already matched ground truths
                    
                iou_val = iou(boxes[j], gt_boxes[k])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = k
            
            # Consider match if IoU > threshold
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_matched.append(1)
                gt_matched[i][best_gt_idx] = True
            else:
                all_matched.append(0)
    
    # Sort by confidence score
    indices = np.argsort(-np.array(all_scores))
    all_matched = np.array(all_matched)[indices]
    
    # Compute cumulative TP and FP
    tp = np.cumsum(all_matched)
    fp = np.cumsum(1 - all_matched)
    
    # Compute precision and recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / total_gt
    
    # Compute AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap

def iou(boxA, boxB):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute area of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-10)
    return iou


def compute_tracking_accuracy_and_id_switches(tracker_results, ground_truth):
    """
    Compute tracking accuracy and identity switch rate.
    
    Args:
        tracker_results: Dictionary mapping frame_num to list of (track_id, bbox)
        ground_truth: Dictionary mapping frame_num to list of gt_boxes
        
    Returns:
        tracking_accuracy: Overall tracking accuracy (MOTA-inspired metric)
        identity_switch_rate: Rate of ID switches per frame
    """
    # Initialize counters
    total_gt = 0
    total_matches = 0
    total_misses = 0
    total_false_positives = 0
    total_id_switches = 0
    
    # Dictionary to keep track of last matched track_id for each gt box
    last_matched_ids = {}
    
    # Process each frame
    common_frames = sorted(set(tracker_results.keys()) & set(ground_truth.keys()))
    
    for frame_num in common_frames:
        tracker_boxes = [box for _, box in tracker_results[frame_num]]
        tracker_ids = [tid for tid, _ in tracker_results[frame_num]]
        gt_boxes = ground_truth[frame_num]
        
        total_gt += len(gt_boxes)
        
        # Keep track of which ground truth and tracker boxes have been matched
        gt_matched = [False] * len(gt_boxes)
        tracker_matched = [False] * len(tracker_boxes)
        
        # Match ground truth with tracker boxes using IoU
        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0.5  # Minimum IoU threshold
            best_match = -1
            
            for j, tracker_box in enumerate(tracker_boxes):
                if tracker_matched[j]:
                    continue
                    
                iou_val = iou(gt_box, tracker_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_match = j
            
            if best_match >= 0:
                # We have a match
                gt_matched[i] = True
                tracker_matched[best_match] = True
                
                # Check if there's an ID switch
                gt_idx = tuple(gt_box.tolist())  # Convert to hashable tuple
                if gt_idx in last_matched_ids:
                    if last_matched_ids[gt_idx] != tracker_ids[best_match]:
                        total_id_switches += 1
                
                # Update last matched ID
                last_matched_ids[gt_idx] = tracker_ids[best_match]
                total_matches += 1
            else:
                total_misses += 1
        
        # Count false positives
        total_false_positives += sum(not x for x in tracker_matched)
    
    # Compute MOTA-inspired tracking accuracy
    tracking_accuracy = 1 - (total_misses + total_false_positives + total_id_switches) / max(1, total_gt)
    
    # Compute identity switch rate (ID switches per frame)
    identity_switch_rate = total_id_switches / len(common_frames) if common_frames else 0
    
    return tracking_accuracy, identity_switch_rate

@torch.no_grad()
def compute_map(model, data_loader, device, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP) at a given IoU threshold.
    
    Args:
        model: The detection model
        data_loader: DataLoader for the validation set
        device: Device to run computations on
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        mAP: Mean Average Precision score
    """
    model.eval()
    
    # Lists to store precision and recall values for each class
    all_detections = []
    all_ground_truths = []
    
    # Collect all predictions and ground truths
    for images, targets in tqdm(data_loader, desc="Computing mAP"):
        images = [img.to(device) for img in images]
        
        # Get model predictions
        outputs = model(images)
        
        for i, output in enumerate(outputs):
            pred_boxes = output["boxes"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            
            gt_boxes = targets[i]["boxes"].cpu().numpy()
            gt_labels = targets[i]["labels"].cpu().numpy()
            
            # Store predictions and ground truths for this image
            all_detections.append((pred_boxes, pred_scores, pred_labels))
            all_ground_truths.append((gt_boxes, gt_labels))
    
    # Compute AP for each class (in this case, just "person" class=1)
    class_ids = [1]  # Person class
    aps = []
    
    for class_id in class_ids:
        # Get detections and ground truths for this class
        class_detections = []
        class_ground_truths = []
        
        for i in range(len(all_detections)):
            pred_boxes, pred_scores, pred_labels = all_detections[i]
            gt_boxes, gt_labels = all_ground_truths[i]
            
            # Filter by class
            class_pred_indices = np.where(pred_labels == class_id)[0]
            class_gt_indices = np.where(gt_labels == class_id)[0]
            
            class_pred_boxes = pred_boxes[class_pred_indices]
            class_pred_scores = pred_scores[class_pred_indices]
            class_gt_boxes = gt_boxes[class_gt_indices]
            
            # Sort predictions by confidence score (descending)
            sorted_indices = np.argsort(-class_pred_scores)
            class_pred_boxes = class_pred_boxes[sorted_indices]
            class_pred_scores = class_pred_scores[sorted_indices]
            
            class_detections.append((class_pred_boxes, class_pred_scores))
            class_ground_truths.append(class_gt_boxes)
        
        # Compute precision and recall
        ap = compute_average_precision(class_detections, class_ground_truths, iou_threshold)
        aps.append(ap)
    
    # Return mean AP across all classes
    return np.mean(aps)


##############################################
# 5. TRAINING LOOP WITH TQDM + CHECKPOINTING
##############################################

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch_idx, global_step=0):
    """
    Train for one epoch. 
    'global_step' is incremented each batch for frequent checkpointing.
    Returns: (average_loss, updated_global_step)
    """
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch_idx+1} Training", leave=False)
    
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix({"loss": f"{losses.item():.4f}"})
        
        global_step += 1
        
        # ---------- Frequent checkpointing ----------
        if global_step % steps_per_checkpoint == 0:
            # We'll do a mid-epoch checkpoint
            save_checkpoint(epoch_idx, global_step, model, optimizer,
                            checkpoint_dir=checkpoints_path,
                            filename_prefix=f"checkpoint_step_{global_step}")
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, global_step


@torch.no_grad()
def validate_model(model, data_loader, device, epoch_idx):
    model.eval()
    val_pbar = tqdm(data_loader, desc=f"Epoch {epoch_idx+1} Validation", leave=False)
    total_detections = 0
    total_gt_boxes = 0
    for images, targets in val_pbar:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        for i, output in enumerate(outputs):
            total_detections += len(output["boxes"])
            total_gt_boxes += len(targets[i]["boxes"])
    if len(data_loader) > 0:
        val_pbar.set_postfix({"avg_detected_per_image": total_detections / len(data_loader)})
    return total_detections, total_gt_boxes


def save_checkpoint(epoch, global_step, model, optimizer, checkpoint_dir=checkpoints_path, filename_prefix="checkpoint"):
    """
    Saves a checkpoint with epoch, global_step, model state, and optimizer state.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{filename_prefix}.pth")
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, checkpoint_path)
    # Also save a 'last_checkpoint.pth' for easy resuming
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, last_ckpt_path)
    print(f"=> Checkpoint saved at {checkpoint_path} and {last_ckpt_path}")


def load_checkpoint_if_available(model, optimizer, checkpoint_dir=checkpoints_path):
    """
    Checks if 'last_checkpoint.pth' exists and loads it.
    Returns (start_epoch, global_step).
    """
    last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"=> Loaded checkpoint at epoch={ckpt['epoch']}, step={global_step} from {last_ckpt_path}")
        return start_epoch, global_step
    else:
        print("=> No existing checkpoint found. Starting from scratch.")
        return 0, 0


##############################################
# 6. MAIN SCRIPT
##############################################

def main():
    """
    Main driver function with frequent checkpointing and mid-epoch saves.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    
    dataset_root = r"2907/mot17/MOT17Det/train/MOT17-02"
    train_start, train_end = 1, 300
    val_start, val_end = 301, 350
    
    # Data sets
    dataset_train = MOTDataset(
        root_dir=dataset_root,
        start_frame=train_start,
        end_frame=train_end,
        transforms=CustomTransforms(train=True)
    )
    dataset_val = MOTDataset(
        root_dir=dataset_root,
        start_frame=val_start,
        end_frame=val_end,
        transforms=CustomTransforms(train=False)
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn
    )
    
    # Model
    model = get_faster_rcnn_model(num_classes=2).to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Load from checkpoint if available
    start_epoch, global_step = load_checkpoint_if_available(model, optimizer, checkpoints_path)
    
    # Training
    num_epochs = 5
    for epoch in range(start_epoch, num_epochs):
        train_loss, global_step = train_one_epoch(model, optimizer, train_loader, device, epoch, global_step)
        print(f"[Epoch {epoch+1}] Training loss: {train_loss:.4f}")
        
        # Validation step
        total_detections, total_gt_boxes = validate_model(model, val_loader, device, epoch)
        print(f"[Epoch {epoch+1}] Avg det/frame: {total_detections / len(val_loader):.2f}, GT/frame: {total_gt_boxes / len(val_loader):.2f}")
        
        # End-of-epoch checkpoint
        save_checkpoint(epoch, global_step, model, optimizer, checkpoint_dir=checkpoints_path,
                        filename_prefix=f"checkpoint_epoch_{epoch}")
    
    # Evaluate with proper mAP calculation
    print("\nCalculating mAP on validation set...")
    map_val = compute_map(model, val_loader, device)
    print(f"mAP @ IoU=0.5: {map_val:.4f}")
    
    # Tracking example
    print("\nPerforming tracking and evaluation...")
    tracker = NaiveSORTTracker()
    
    # We'll store results for each frame to measure tracking performance
    track_results = {}
    gt_dict = {}
    
    # Inference on the validation set frame by frame
    for idx in tqdm(range(len(dataset_val)), desc="Tracking"):
        img, target = dataset_val[idx]
        frame_num = target["frame_num"].item()
        
        model.eval()
        with torch.no_grad():
            detection = model([img.to(device)])[0]
        
        # Convert predictions to CPU numpy for the tracker
        pred_boxes = detection["boxes"].cpu().numpy()
        scores = detection["scores"].cpu().numpy()
        
        # Build list of (bbox, score)
        detections = []
        for b, s in zip(pred_boxes, scores):
            if s > 0.5:
                detections.append((b, s))
        
        # Update tracker
        assigned_tracks = tracker.update(frame_num, detections)
        track_results[frame_num] = assigned_tracks
        
        # Save ground truth
        gt_boxes_np = target["boxes"].numpy()
        gt_dict[frame_num] = gt_boxes_np
    
    # Compute tracking metrics
    tracking_acc, id_switch_rate = compute_tracking_accuracy_and_id_switches(track_results, gt_dict)
    print(f"Tracking Accuracy: {tracking_acc:.4f}")
    print(f"Identity Switch Rate: {id_switch_rate:.4f}")
    
    # Compare with baseline
    print("\nComparing with baseline approaches...")
    comparison_metrics = compare_with_baseline(model, tracker, dataset_val, device)
    
    # Print comparison results
    print("\nComparison Results:")
    print("---------------------------------------------------------------")
    print("Method                | Precision | Recall | ID Switch Rate")
    print("---------------------------------------------------------------")
    print(f"Our Approach          | {comparison_metrics['our_approach']['precision']:.4f} | {comparison_metrics['our_approach']['recall']:.4f} | {comparison_metrics['our_approach'].get('id_switch_rate', 0):.4f}")
    print(f"Simple IoU Tracker    | {comparison_metrics['simple_iou_tracker']['precision']:.4f} | {comparison_metrics['simple_iou_tracker']['recall']:.4f} | {comparison_metrics['simple_iou_tracker'].get('id_switch_rate', 0):.4f}")
    print(f"No Tracking           | {comparison_metrics['no_tracking']['precision']:.4f} | {comparison_metrics['no_tracking']['recall']:.4f} | N/A")
    print("---------------------------------------------------------------")
    
    print("Finished training, validation, tracking, and evaluation!")

if __name__ == "__main__":
    main()
