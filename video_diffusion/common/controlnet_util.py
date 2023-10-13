import torch
import numpy as np
from einops import rearrange
import cv2
import PIL
from PIL import Image
import annotator
from annotator.midas import MidasDetector
from annotator.openpose import OpenposeDetector
from annotator.uniformer import UniformerDetector
from annotator.hed import HEDdetector, nms
import os
from video_diffusion.common.image_util import save_images_as_gif
from video_diffusion.pipelines.pipeline_tuneavideo_controlnet import MultiControlNetModel


def load_annotator_inputs(train_dataset, fn):
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=1,collate_fn=fn)
    
    x_batch = None
    for batch in train_dataloader:
        x_batch = batch["images"]   # values -1~1 / shape (1,3,8,512,512)
        break
    
    train_imgs = rearrange(
         x_batch[0].clone(),
         'c f h w -> f h w c'
    )   # values -1~1 / shape (8,3,512,512)

    x = np.array(train_imgs[0]) # values -1~1 / shape (512,512,3)

    ann_inputs = []
    for x in train_imgs[:]:
         image = (x+1) *127.5
         image = np.array(image).astype(np.uint8)
         ann_inputs.append(image)

    return ann_inputs


def annotator_inputs_to_controlnet_hints(annotator_inputs, control_types, save_detected_maps=False, save_dir=None):
    outs = []
    for control_type in control_types:
        images = []
        if control_type == "canny":
            for annotator_input in annotator_inputs:
                detected_map = cv2.Canny(annotator_input,100,200)
                detected_map = HWC3(detected_map)
                detected_map = Image.fromarray(detected_map)
                images.append(detected_map)

        elif control_type == "seg":
            with torch.no_grad():
                apply_model = UniformerDetector()
                for annotator_input in annotator_inputs:
                    detected_map = apply_model(annotator_input)
                    detected_map = Image.fromarray(detected_map)
                    images.append(detected_map)
                del apply_model


        elif control_type == "depth":
            with torch.no_grad():
                apply_model = MidasDetector()
                for annotator_input in annotator_inputs:
                    detected_map, _ = apply_model(annotator_input)
                    detected_map = HWC3(detected_map)
                    detected_map = Image.fromarray(detected_map)
                    images.append(detected_map)
                del apply_model

        elif control_type == "openpose":
            apply_model = OpenposeDetector()
            for annotator_input in annotator_inputs:
                detected_map, _ = apply_model(annotator_input)
                detected_map = HWC3(detected_map)
                detected_map = Image.fromarray(detected_map)
                images.append(detected_map)
            del apply_model

        elif control_type == "scribble":
            with torch.no_grad():
                apply_model = HEDdetector()
                for annotator_input in annotator_inputs:
                    detected_map = apply_model(annotator_input)
                    detected_map = HWC3(detected_map)
                    detected_map = nms(detected_map, 127, 3.0)
                    detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
                    detected_map[detected_map > 4] = 255
                    detected_map[detected_map < 255] = 0
                    detected_map = Image.fromarray(detected_map)
                    images.append(detected_map)
                del apply_model

        elif control_type == "hed":
            with torch.no_grad():
                apply_model = HEDdetector()
                for annotator_input in annotator_inputs:
                    detected_map = apply_model(annotator_input)
                    detected_map = HWC3(detected_map)
                    detected_map = Image.fromarray(detected_map)
                    images.append(detected_map)
                del apply_model
        else:
            raise NotImplementedError
        
        if save_detected_maps:
            save_dir = os.path.join(save_dir,"hints")
            os.makedirs(save_dir,exist_ok=True)
            save_gif_path = os.path.join(save_dir, f"{control_type}.gif")
            save_images_as_gif(images=images, save_path=save_gif_path)
            for i,img in enumerate(images):
                save_img_path = os.path.join(save_dir,f"{control_type}_{i}.png")
                img.save(save_img_path, "png")
        
        outs.append(images)

    if len(outs) == 1:
        outs = outs[0]

    return outs


def HWC3(x):
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def unlock_model_weights(model, trainable_modules, gradient_checkpointing):
    for name, module in model.named_modules():   
        if name.endswith(trainable_modules):
            for params in module.parameters():
                params.requires_grad = True
    
    if gradient_checkpointing:
        if isinstance(model, MultiControlNetModel):
            for net in model.nets:
                net.enable_gradient_checkpointing()
        else:
            model.enable_gradient_checkpointing()