import os
import cv2
import torch
import numpy as np
import standard_text
from itertools import zip_longest
from PIL import Image
from torchvision import transforms
from paddleocr import PaddleOCR, draw_ocr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from predict import MyDilate
from model import Generator


class YULA:
    def __init__(self, cfg, bbox_scale_factor_x=1.1, bbox_scale_factor_y=1.25, dilate=True, slm=True, auto=True):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.auto = auto

        self.ocr_processor = PaddleOCR(lang='japan', ocr_version="PP-OCRv4")
        self.bbox_scale_factor_x = bbox_scale_factor_x
        self.bbox_scale_factor_y = bbox_scale_factor_y

        self.m2m100_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B").to(self.device)
        self.m2m100_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

        self.generator = self._load_generator()

        self.transform = transforms.Compose([
            transforms.Resize(self.cfg.data_shape),
            transforms.ToTensor(),
        ])
        self.std_text = standard_text.Std_Text(cfg.font_path)
        self.dilate = dilate
        self.slm = slm

    def _load_generator(self):
        model = torch.load(self.cfg.generator_path)
        model.eval()
        print(f'Model loaded: {self.cfg.generator_path}')
        return model

    def _ocr(self, img, img_path):
        for x in (1, 2, 3, 4):
            img = img.resize((img.width * x, img.height * x))
            result = self.ocr_processor.ocr(np.array(img), cls=False)[0]
            if result:
                return result, x
        raise ValueError(f"Failed to detect text on image: {img_path}")

    def _scale_bbox(self, bbox):
        cx = sum(p[0] for p in bbox) / 4
        cy = sum(p[1] for p in bbox) / 4
        return [(int(cx + (x - cx) * self.bbox_scale_factor_x), int(cy + (y - cy) * self.bbox_scale_factor_y)) for x, y
                in bbox]

    def _m2m100_translate(self, text, src_lang, target_lang):
        # uk - Ukrainian; ja - Japanese; en - English
        self.m2m100_tokenizer.src_lang = src_lang
        encoded = self.m2m100_tokenizer(text, return_tensors="pt").to(self.device)
        generated_tokens = self.m2m100_model.generate(**encoded, forced_bos_token_id=self.m2m100_tokenizer.get_lang_id(
            target_lang))
        return self.m2m100_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    @staticmethod
    def _get_crop_box(points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return [min(x), min(y), max(x), max(y)]

    def _get_generator_input(self, img, text):
        i_t = self.std_text.draw_text(text)
        i_t = Image.fromarray(np.uint8(i_t))
        i_t = self.transform(i_t).unsqueeze(0)  # adding batch_size=1 dimension
        i_s = self.transform(img).unsqueeze(0)  # adding batch_size=1 dimension
        return i_s.to(self.device), i_t.to(self.device)

    def _generate_adapted(self, img, text):
        if self.dilate:
            dilate = MyDilate()

        with torch.no_grad():
            i_s, i_t = self._get_generator_input(img, text)

            gen_o_b_ori, gen_o_b, gen_o_f, _, gen_o_mask_s, gen_o_mask_t = self.generator(i_t, i_s)

            gen_o_b_ori = gen_o_b_ori * 255
            gen_o_b = gen_o_b * 255
            gen_o_f = gen_o_f * 255

            o_mask_s = gen_o_mask_s[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            o_mask_t = gen_o_mask_t[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            o_b_ori = gen_o_b_ori[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            o_b = gen_o_b[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            o_f = gen_o_f[0].detach().to('cpu').numpy().transpose(1, 2, 0)

            ori_o_mask_s = o_mask_s
            if self.dilate:
                tmp_i_s = (i_s * 255)[0].detach().to('cpu').numpy().transpose(1, 2, 0)
                o_mask_s = dilate(o_mask_s)
                o_b = o_mask_s * o_b_ori + (1 - o_mask_s) * tmp_i_s

            if self.slm:
                alpha = 0.5
                o_f = o_mask_t * o_f + (1 - o_mask_t) * (alpha * o_b + (1 - alpha) * o_f)

            return o_f
            # cv2.imwrite(os.path.join(args.save_dir, name + '.' + suffix), o_f[:, :, ::-1])

    @staticmethod
    def _merge_images(original_img, adapted_img, bbox):
        original_img.paste(adapted_img, bbox)
        return original_img

    def _process_one_image(self, img_path):
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        filename = img_path.split('/')[-1]
        print(filename)
        print(img.size)
        img.show()
        ocr_results, factor = self._ocr(img, img_path)
        print(f"Resize factor: {factor}x")

        for i, ocr_result in enumerate(ocr_results):
            bbox = self._scale_bbox(ocr_result[0])
            ja_text = ocr_result[1][0]
            en_text = self._m2m100_translate(ja_text, "ja", "en")[0]
            uk_text = self._m2m100_translate(en_text, "en", "uk")[0]
            print(ja_text, en_text, uk_text)

            if not self.auto:
                override_text = input("Please enter your target text: ")
                if override_text:
                    uk_text = override_text
                else:
                    print("Choosing M2M100 translation")

            Image.fromarray(draw_ocr(img, [bbox])).show()

            crop_box = self._get_crop_box(bbox)
            cropped_img = img.crop(crop_box)

            print(i, "bbox", bbox)
            print(cropped_img.size)
            cropped_img.show()

            print("Adapting the image")

            adapted_img = self._generate_adapted(cropped_img, uk_text)
            adapted_img = Image.fromarray(np.uint8(adapted_img))
            print(adapted_img.size)
            adapted_img.show()

            resized_adapted_img = adapted_img.resize(cropped_img.size)
            print(resized_adapted_img.size)
            resized_adapted_img.show()

            merged_img = self._merge_images(img, resized_adapted_img, crop_box)
            merged_img.show()
            if not self.auto:
                merge = input("Merge? (y | n)")
                if merge.lower() in ('yes', 'y'):
                    img = merged_img
                else:
                    print("Skipping")
            print("\n\n")

    def pipeline(self, img_paths):
        img_paths = img_paths if isinstance(img_paths, list) else [img_paths]
        for img_path in img_paths:
            if img_path.endswith(".png") or img_path.endswith(".jpg"):
                self._process_one_image(img_path)
