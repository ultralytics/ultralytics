from PIL import Image
import os

os.makedirs('data/train/images', exist_ok=True)
os.makedirs('data/train/labels', exist_ok=True)

for i in range(1, 6):
    img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    img.save(f'data/train/images/{i}.jpg', 'JPEG')
    with open(f'data/train/labels/{i}.txt', 'w') as f:
        f.write('0 0.5 0.5 0.5 0.5\n')
