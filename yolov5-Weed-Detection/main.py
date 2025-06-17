import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# Print input and output info
print("=== Model Inputs ===")
for i, input_tensor in enumerate(session.get_inputs()):
    print(f"Input {i}: name = {input_tensor.name}, shape = {input_tensor.shape}, type = {input_tensor.type}")

print("=== Model Outputs ===")
for i, output_tensor in enumerate(session.get_outputs()):
    print(f"Output {i}: name = {output_tensor.name}, shape = {output_tensor.shape}, type = {output_tensor.type}")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Class labels — your intended classes
class_names = ['herb paris', 'karela', 'small weed', 'grass', 'tori', 'horseweed', 'Bhindi', 'weed']

# Preprocess image
def preprocess(img_path, img_size=640):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {img_path}")
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img, img_resized

# Postprocess with NMS
def postprocess(predictions, conf_thres=0.25, iou_thres=0.45):
    preds = np.squeeze(predictions[0])  # shape: (N, C)
    print("Raw prediction shape:", preds.shape)

    if preds.shape[1] < 6:
        raise ValueError("Model output shape is unexpected. Must be (N, 6+)")

    boxes = preds[:, :4]
    objectness = preds[:, 4]
    class_probs = preds[:, 5:]

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(len(class_ids)), class_ids]
    scores = objectness * class_scores

    results = []
    for box, score, cls in zip(boxes, scores, class_ids):
        if score > conf_thres:
            cx, cy, w, h = box
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            results.append(([x, y, int(w), int(h)], float(score), int(cls)))

    print(f"Detections before NMS: {len(results)}")
    if not results:
        return []

    boxes_xywh = [r[0] for r in results]
    scores = [r[1] for r in results]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=conf_thres, nms_threshold=iou_thres)

    indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
    print(f"Detections after NMS: {len(indices)}")

    return [results[i] for i in indices]

# Main inference flow
img_path = "images/1.jpg"
img_input, img_display = preprocess(img_path)

# Run inference
predictions = session.run([output_name], {input_name: img_input})
results = postprocess(predictions)

# Draw results
for (x, y, w, h), score, cls in results:
    label = f"class {cls}: {score:.2f}"
    if cls < len(class_names):
        label = f"{class_names[cls]}: {score:.2f}"
    else:
        print(f"⚠️ Predicted class index {cls} exceeds class_names list (len={len(class_names)})")

    cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 0, 0), 2)
    cv2.putText(img_display, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Save and show result
cv2.imwrite("output.jpg", img_display)
cv2.imshow("Detections", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
