import cv2
import pandas as pd
import os

folder = r"C:\Users\gbhan\COLLEGE NOTES ARRANGING\SECOND SEMESTER\Image-Processing and Computer Vision\TASKS\Task 4"
csv_path = os.path.join(folder, "labels.csv")

label_map = {ord('n'): 'N', ord('m'): 'M', ord('f'): 'F',
             ord('N'): 'N', ord('M'): 'M', ord('F'): 'F'}
regions = ['TL', 'TR', 'BL', 'BR']

images = sorted([f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')])
results = []

for idx, img_name in enumerate(images):
    img = cv2.imread(os.path.join(folder, img_name))
    if img is None:
        continue
    h, w = img.shape[:2]
    labels = []

    for region in regions:
        vis = img.copy()
        # Draw quadrant lines
        cv2.line(vis, (w//2, 0), (w//2, h), (0, 255, 0), 3)
        cv2.line(vis, (0, h//2), (w, h//2), (0, 255, 0), 3)
        # Highlight current region
        coords = {
            'TL': (0, 0, w//2, h//2),
            'TR': (w//2, 0, w, h//2),
            'BL': (0, h//2, w//2, h),
            'BR': (w//2, h//2, w, h)
        }
        x1, y1, x2, y2 = coords[region]
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # Labels
        cv2.putText(vis, "TL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.putText(vis, "TR", (w//2+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.putText(vis, "BL", (20, h//2+50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.putText(vis, "BR", (w//2+20, h//2+50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.putText(vis, f"[{idx+1}/{len(images)}] Label {region} → Press N / M / F",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

        cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Labeler", 900, 600)
        cv2.imshow("Labeler", vis)

        while True:
            key = cv2.waitKey(0)
            if key in label_map:
                labels.append(label_map[key])
                break
            elif key == 27:  # ESC to quit and save progress
                cv2.destroyAllWindows()
                pd.DataFrame(results, columns=['filename','TL','TR','BL','BR']).to_csv(csv_path, index=False)
                print("Saved progress and exited.")
                exit()

    results.append([img_name] + labels)
    print(f"[{idx+1}/{len(images)}] {img_name} → {labels}")

cv2.destroyAllWindows()
pd.DataFrame(results, columns=['filename','TL','TR','BL','BR']).to_csv(csv_path, index=False)
print(f"\n✅ Done! All {len(results)} images labeled. Saved to labels.csv")