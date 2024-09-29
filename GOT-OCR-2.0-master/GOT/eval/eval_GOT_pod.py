import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

from tqdm import tqdm
from PIL import Image
import json
import os
import requests
from PIL import Image
from io import BytesIO
import math

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
# logical_role_id2names = {0: 'background', 1: 'figure', 2: 'table', 3: 'equation', 4: 'caption', 5: 'footnote', 6: 'list-item', 7: 'footer', 8: 'header', 9: 'section', 10: 'paraline', 11: 'title', 12: 'others', 13: 'in_figure_others', 14: 'in_table_others', 15: 'in_formula_others'}

Doclaynet_category_mapping = {'Formula': 3, 'Table': 2, 'Page-footer': 7, 'others': 12, 'Text': 10, 'Picture': 1, 'Footnote': 5, 'Title': 11, 'Page-header': 8, 'List-item': 6, 'Section-header': 9, 'Caption': 4}

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()

    # vary old codes, no use
    image_processor = BlipImageEvalProcessor(image_size=1024)

    # image_processor_high = BlipImageEvalProcessor(image_size=1280)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    use_im_start_end = True

    # image_token_len = 400
    image_token_len = 256

    image_lists = os.listdir(args.image_path)

    print("Generate Results......")

    for image_name in tqdm(image_lists[2239:]):
        image_file_path = os.path.join(args.image_path, image_name)
        ll = 1
        image = load_image(image_file_path)
        image_1 = image.copy()
        # image_1 = image_1.resize((1024, 1024))

        # vary old code, NO USE
        image_tensor = image_processor_high(image_1)

        image_tensor_1 = image_processor_high(image_1)

        w, h = image.size

        qs =  DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len*ll + DEFAULT_IM_END_TOKEN  + '\n' +  'POD: '

        conv_mode = "mpt"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids,
                images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                do_sample=False,
                num_beams = 1,
                # temperature=0.2,
                no_repeat_ngram_size = 20,
                # encoder_repetition_penalty = 1.2,
                # penalty_alpha=0.2,
                # top_k=3,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria],
                use_cache=True,
                )
    
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        outputs_list = outputs.split('\n')
        pred_bboxes = []
        pred_logical_roles = []
        for out in outputs_list:
            try:
                bbox, logical_role = out.split('] ')
                bbox = bbox[1:].split(', ')
                # bbox = [int(i) for i in bbox]
                bbox = [int(i) / 1000 for i in bbox]
                bbox = [int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)]
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 0 or bbox[3] < 0:
                    continue
            except:
                continue
            pred_bboxes.append(bbox)
            pred_logical_roles.append(logical_role)
        if args.demo and len(os.listdir(os.path.join(args.out_path, 'viz'))) < 10:
            # visualize the bboxes and logical role in args.image_file: [117, 88, 328, 100] ['Section-header']
            import cv2
            image_viz = cv2.imread(image_file_path)
            logical_role2color = {"Section-header": (0, 255, 0), "Text": (0, 0, 255), "Table": (255, 0, 0), "Figure": (255, 255, 0), "Page-footer": (255, 0, 255), "Page-header": (0, 255, 255), "Formula": (130, 255, 255), "List-item": (165, 123, 132)}
            for bbox, logical_role in zip(pred_bboxes, pred_logical_roles):
                color = logical_role2color.get(logical_role, (0, 0, 0))
                image_viz = cv2.rectangle(image_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                image_viz = cv2.putText(image_viz, logical_role, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            viz_path = os.path.join(args.out_path, 'viz', image_name)
            cv2.imwrite(viz_path, image_viz)
            print("Visualized image is saved in {}".format(viz_path))
        
        # save the results
        result_json = []
        for bbox, logical_role in zip(pred_bboxes, pred_logical_roles):
            bbox_xywh = [float(bbox[0]), float(bbox[1]), float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1])]
            segmentation = [[bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]]
            score = 1.0
            category_id = Doclaynet_category_mapping.get(logical_role, 12)

            result_json.append({"bbox": bbox_xywh, "segmentation": segmentation, "score": score, "category_id": category_id, "file_name": image_name})
        
        det_json_path = os.path.join(args.out_path, 'det_json', image_name.split('.')[0] + '.json')
        with open(det_json_path, 'w') as f:
            json.dump(result_json, f, indent=4)
    print("Results are saved in {}".format(args.out_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--demo", action='store_true')
    args = parser.parse_args()
    print(args)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'viz'), exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'det_json'), exist_ok=True)
    eval_model(args)
