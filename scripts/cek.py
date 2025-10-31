import os

for cls in ['dark','medium','light']:
    count = len(os.listdir(f'dataset/raw/{cls}'))
    print(f"{cls}:{count}images")