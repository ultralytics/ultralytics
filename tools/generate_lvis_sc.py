import json

if __name__ == '__main__':
    path = '../datasets/lvis/annotations/lvis_v1_minival.json'
    
    with open(path, 'r') as f:
        data = json.load(f)

    for anno in data['annotations']:
        anno['category_id'] = 1
    
    with open('../datasets/lvis/annotations/lvis_v1_minival_sc.json', 'w') as f:
        json.dump(data, f)