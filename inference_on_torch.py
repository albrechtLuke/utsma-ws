import os
import cv2
import time
import torch
from ultralytics import YOLO

model = YOLO('bestv11.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)

def prepare_images(image_paths):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640,640))
        image = image.transpose((2,0,1)) # W, H, C => C, H, W
        image = torch.from_numpy(image).to(device)
        image = image.float() / 255.0 # Normalise [0,1]
        images.append(image)
    return images

# Load Dataset
image_dir = 'datasets/valid/images'
image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
# print(image_paths)

# Only take a subset of images
IMAGE_NUMS = 128
images = prepare_images(image_paths[:IMAGE_NUMS])

def run_inference(model, images):
    times = []
    results = []
    for image in images:

        # Move image to GPU
        image = image.unsqueeze(0)
        image_gpu = image.to(device) # or .to('cuda')

        # Run inference with no gradient calculation
        start = time.time()
        with torch.no_grad():
            result = model(image_gpu)
        
        results.append(result)
        times.append(time.time() - start)

    return results, times

NUM_TRIALS = 10
avgs = []

for i in range(NUM_TRIALS):
    results, times = run_inference(model, images)
    avg = sum(times)/len(times)
    print("Avg inferenec time (s):", avg)
    avgs.append(avg)
        
mat = sum(avgs) / NUM_TRIALS
print("Mean avg inference time (s):", mat)
