import os

def handle_folder_contents(ann_folder_path, image_id_captions_dict):
    if not os.path.isdir(ann_folder_path):
        print(f"Error: {ann_folder_path} is not a valid directory.")
        return

    file_list = os.listdir(ann_folder_path)

    for file_name in file_list:
        file_path = os.path.join(ann_folder_path, file_name)

        if os.path.isdir(file_path):
            # 处理子文件夹
            print(f"Subfolder: {file_name}")
            handle_folder_contents(file_path)
        elif os.path.isfile(file_path):
            # 处理文件
            base_name = os.path.basename(file_path).replace('.xml', '') # img_id
            if len(base_name) == 12:
                captions = get_caption(image_id_captions_dict, base_name) # captions of image[base_name]
                add_caption(file_path, captions, ann_folder_path)
                global modified_count
                modified_count += 1


def add_caption(xml_file_path, captions, ann_folder_path):
    import os
    import xml.etree.ElementTree as ET

    # 解析XML文件
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    for cap in captions:
        # 创建要添加的新元素
        tag_name = "caption"
        caption_tag = ET.SubElement(root, tag_name)
        caption_tag.text = cap

    # 构建新的 XML 文件路径
    xml_file_name = os.path.basename(xml_file_path)
    new_xml_file_path = os.path.join(ann_folder_path, xml_file_name)

    # 将修改后的XML保存到新文件中
    tree.write(new_xml_file_path)

    print(f"Modified ann and save successfully: {new_xml_file_path}")


def combine_caption(coco_annotation_dir):
    '''
    return:
        dict:{img_id:[list of captions about this img]}
    '''
    import json
    # First merge caption annotations from train and val
    # Load the train and val annotations
    json_train_file = '{:s}/captions_{:s}.json'.format(coco_annotation_dir, 'train2017')
    print("Loading training caption annotations from %s"%(format(json_train_file)))
    json_train = json.load(open(json_train_file, 'r'))

    json_val_file = '{:s}/captions_{:s}.json'.format(coco_annotation_dir, 'val2017')
    print("Loading validating caption annotations from %s"%(format(json_val_file)))
    json_val = json.load(open(json_val_file, 'r'))

    # Copy and sanity check
    assert(json_train['info']       == json_val['info'])
    assert(json_train['licenses']   == json_val['licenses'])
    
    json_all = json_train
    json_all['images']      = json_train['images'] + json_val['images'] # 123287 = 118287 + 5000
    json_all['annotations'] = json_train['annotations'] + json_val['annotations'] # 616767 = 591753 + 25014
    del json_train 
    del json_val

    # 构建标签和内容的字典
    image_id_captions_dict = {} # {img_id:[list of captions about this img]}
    for item in json_all['annotations']:
        image_id = str(item['image_id']).zfill(12) # coco文件名长度为12
        caption = item['caption']
        if image_id in image_id_captions_dict:
            image_id_captions_dict[image_id].append(caption)
        else:
            image_id_captions_dict[image_id] = [caption]
    return image_id_captions_dict

    
def get_caption(image_id_captions_dict, img_id):
    return image_id_captions_dict[img_id]
    
if __name__ == "__main__":
    # 指定文件夹路径
    
    coco_annotation_dir = "/data/wxf/datasets/coco/annotations"
    ann_folder_path = "/data2/wxf2/datasets/OWOD_add_captions/Annotations"
    
    modified_count = 0
    
    image_id_captions_dict = combine_caption(coco_annotation_dir)
    handle_folder_contents(ann_folder_path, image_id_captions_dict)
    
    
    print(f'{modified_count} ann file is modified.')
    
# 注：122218 ann file is modified.

