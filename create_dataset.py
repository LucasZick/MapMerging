import numpy as np
import random
import scipy
import os
import gc

def generate_structured_map(size=200):
    map = np.zeros((size, size), dtype=int)
    
    def add_area(x, y, width, height):
        x_end = min(size, x + width)
        y_end = min(size, y + height)
        map[x:x_end, y:y_end] = 1
    
    def add_large_areas():
        for _ in range(random.randint(3, 8)):
            area_size = random.randint(size // 8, size // 4)
            x_start = random.randint(0, size - area_size)
            y_start = random.randint(0, size - area_size)
            add_area(x_start, y_start, area_size, area_size)

    def add_corridors():
        for _ in range(random.randint(10, 15)):
            corridor_width = random.randint(5, 10)
            start_x = random.randint(0, size - corridor_width)
            start_y = random.randint(0, size - corridor_width)
            length = size
            vertical = random.choice([True, False])
            if vertical:
                add_area(start_x, start_y, corridor_width, length)
            else:
                add_area(start_x, start_y, length, corridor_width)

    def add_rooms():
        for _ in range(random.randint(10, 20)):
            room_size = random.randint(10, 20)
            room_x = random.randint(10, size - 10)
            room_y = random.randint(10, size - 10)
            add_area(room_x, room_y, room_size, room_size)

    add_large_areas()
    add_corridors()
    add_rooms()

    return map

def generate_cut(map, angled=False, size=110):
    angle = random.uniform(0, 360) if angled else 0
    rows, cols = map.shape
    start_row = random.randint(0, rows - size)
    start_col = random.randint(0, cols - size)
    cut = map[start_row:start_row + size, start_col:start_col + size]
    if angled:
        cut = scipy.ndimage.rotate(cut, angle, reshape=False, mode='nearest')
    return cut, start_row, start_col, angle

def create_dataset_chunk(start_index, num_samples=10000):
    features = []
    targets = []
    
    for _ in range(num_samples):
        try:
            map = generate_structured_map()
            cut1, x1, y1, angle1 = generate_cut(map, angled=False)
            cut2, x2, y2, angle2 = generate_cut(map, angled=True)
        
            dx = x2 - x1
            dy = y2 - y1
            d_angle = angle2 - angle1
            
            features.append([cut1, cut2])
            targets.append([dx, dy, d_angle])
        except Exception as e:
            print(f"Error generating sample: {e}")
    
    features = np.array(features)
    targets = np.array(targets)
    
    # Save the chunk to disk
    np.save(f'dataset/features_{start_index}.npy', features)
    np.save(f'dataset/targets_{start_index}.npy', targets)
    
    # Clean up memory
    del features, targets
    gc.collect()

def create_combined_dataset(total_samples=100000, chunk_size=5000):
    num_chunks = total_samples // chunk_size
    for i in range(num_chunks):
        start_index = i * chunk_size
        print(f"Generating chunk {i + 1}/{num_chunks}...")
        create_dataset_chunk(start_index, num_samples=chunk_size)
    
# Criar o dataset combinado em um único arquivo
create_combined_dataset(total_samples=1000000, chunk_size=10000)
