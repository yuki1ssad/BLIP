from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_demo_image(image_size, img_loc, device):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
    
    # img_loc = '/data2/wxf2/code/BLIP/test_imgs/000001.jpg' 
    raw_image = Image.open(img_loc).convert('RGB') 

    w,h = raw_image.size
    # display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

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

    print(f"Modified ann and save successfully: {new_xml_file_path}, add {len(captions)} captions.")
    
if __name__ == '__main__':
    from models.blip import blip_decoder
    import os
    import pickle
    # anns_path = '/data2/wxf2/code/BLIP/test/anns/'
    # imgs_path = '/data2/wxf2/code/BLIP/test/imgs/'
    anns_path = '/data2/wxf2/datasets/add_voc_captions/VOC2007/Annotations/'
    imgs_path = '/data2/wxf2/datasets/add_voc_captions/VOC2007/JPEGImages/'

    image_size = 384
    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    # model_loc = '/data2/wxf2/code/BLIP/pretrained/model_base_capfilt_large.pth'  
    # model_loc = '/data2/wxf2/code/BLIP/pretrained/model_base_caption_capfilt_large.pth' 
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth' 
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    # model = blip_decoder(pretrained=model_loc, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    
    assert os.path.isdir(anns_path)
    ann_file_list = os.listdir(anns_path)
    count = 0
    for ann_file in ann_file_list:
        base_name = os.path.basename(ann_file).replace('.xml', '') # img_id
        if len(base_name) != 12:
            img_loc = f'{imgs_path}/{base_name}.jpg'
            image = load_demo_image(image_size=image_size, img_loc=img_loc, device=device)
            with torch.no_grad():
                # beam search
                caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
                # nucleus sampling
                # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
            add_caption(os.path.join(anns_path, ann_file), caption, anns_path)
            count += 1
            
    print(f'total modified {count} files.')