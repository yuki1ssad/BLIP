import os
import xml.etree.ElementTree as ET

anns_path = '/data2/wxf2/datasets/OWOD_add_captions/VOC2007/Annotations/'
ann_file_list = os.listdir(anns_path)

count_all = 0
count_coco = 0
for ann_file in ann_file_list:
    count_all += 1
    base_name = os.path.basename(ann_file).replace('.xml', '') # img_id
    if len(base_name) == 12:
        count_coco += 1
        f = f'{anns_path}{ann_file}'
        tree = ET.parse(f)
        assert tree.findall("caption"), print(f'{ann_file} len is 12, but has no caption')
    else:
        f = f'{anns_path}{ann_file}'
        tree = ET.parse(f)
        if len(tree.findall("caption")):
            print(f'{ann_file} has caption')
                
        
print(f'total: {count_all}\ncoco: {count_coco}\nvoc:{count_all-count_coco}')