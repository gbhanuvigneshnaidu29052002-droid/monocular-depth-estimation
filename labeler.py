import os
import cv2
import pandas as pd

IMAGE_DIR = "."  # Folder containing your image files
CSV_FILE = "dataset.csv"  # Saved separately to keep your old labels safe

# Find all valid images in the directory
images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
images.sort()

# Load existing progress from dataset.csv if it exists
if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE)
    done_images = df_existing['filename'].tolist()
    labels_list = df_existing.to_dict('records')
else:
    done_images = []
    labels_list = []

print(f"Found {len(images)} images in directory. {len(done_images)} already logged in '{CSV_FILE}'.")

quadrants = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Left (BL)", "Bottom-Right (BR)"]
key_map = {ord('n'): 'N', ord('m'): 'M', ord('f'): 'F'}

for img_name in images:
    # Skip tracking metadata or output files
    if img_name in done_images or img_name in ["labels.csv", "dataset.csv"]:
        continue
        
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    h, w, _ = img.shape
    current_labels = {"filename": img_name}
    
    q_idx = 0
    while q_idx < 4:
        # Create a fresh copy to draw temporary crosshairs and text overlays
        display_img = img.copy()
        cv2.line(display_img, (w//2, 0), (w//2, h), (255, 0, 0), 2)
        cv2.line(display_img, (0, h//2), (w, h//2), (255, 0, 0), 2)
        
        # Display instructions dynamically at the top of the GUI window
        text = f"Assigning {quadrants[q_idx]}. Press: N (Near) | M (Middle) | F (Far)"
        cv2.putText(display_img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_img, f"Image: {img_name}", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Fast Depth Labeler", display_img)
        key = cv2.waitKey(0) & 0xFF
        
        if key in key_map:
            val = key_map[key]
            q_key = ["TL", "TR", "BL", "BR"][q_idx]
            current_labels[q_key] = val
            print(f"{img_name} -> {quadrants[q_idx]}: {val}")
            q_idx += 1
        elif key == 27:  # Press 'ESC' to exit and freeze your progress
            print("Exiting safely. Progress saved.")
            cv2.destroyAllWindows()
            exit()
        else:
            print("Invalid shortcut. Please press N, M, or F on your keyboard.")
            
    labels_list.append(current_labels)
    # Continual saving loop
    pd.DataFrame(labels_list).to_csv(CSV_FILE, index=False)

cv2.destroyAllWindows()
print(f"Success! All annotations are fully written into '{CSV_FILE}'.") 