import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw

from modules import processing, shared, sd_samplers, images, devices
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

def upscale(p, init_img, upscaler_index, tileSize, padding):
    scale_factor = max(p.width, p.height) // max(init_img.width, init_img.height)
    print(f"Canva size: {p.width}x{p.height}")
    print(f"Image size: {init_img.width}x{init_img.height}")
    print(f"Scale factor: {scale_factor}")

    upscaler = shared.sd_upscalers[upscaler_index]

    p.extra_generation_params["SD upscale overlap"] = padding
    p.extra_generation_params["SD upscale upscaler"] = upscaler.name

    initial_info = None
    seed = p.seed

    upscaled_img = init_img
    if upscaler.name == "None":
        return upscaled_img.resize((p.width, p.height), resample=Image.LANCZOS)

    if scale_factor > 4:
        iterations = math.ceil(scale_factor / 4)
    else:
        iterations = 1

    print(f"Total iterations: {iterations}")

    for i in range(iterations):
        if i + 1 == iterations:
            current_scale_factor = scale_factor - i * 4
        else:
            current_scale_factor = 4
        
        print(f"Upscaling iteration {i} with scale factor {current_scale_factor}")
        upscaled_img = upscaler.scaler.upscale(init_img, current_scale_factor, upscaler.data_path)

    return upscaled_img.resize((p.width, p.height), resample=Image.LANCZOS)

class Script(scripts.Script):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>")
        upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")
        tileSize = gr.Slider(minimum=256, maximum=2048, step=64, label='Tile size', value=512)
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=8)
        padding = gr.Slider(label='Padding', minimum=0, maximum=128, step=1, value=32)
        seam_pass_enabled = gr.Checkbox(label= "Seam pass enabled")
        seam_pass_width = gr.Slider(label='Seam pass width', minimum=0, maximum=128, step=1, value=16)
        seam_pass_denoise = gr.Slider(label='Seam pass denoise', minimum=0, maximum=1, step=0.01, value=0.25)
        seam_pass_padding = gr.Slider(label='Seam pass padding', minimum=0, maximum=128, step=1, value=32)

        return [info, tileSize, mask_blur, padding, seam_pass_enabled, seam_pass_width, seam_pass_denoise, seam_pass_padding, upscaler_index]

    def run(self, p, _, tileSize, mask_blur, padding, seam_pass_enabled, seam_pass_width, seam_pass_denoise, seam_pass_padding, upscaler_index):
        processing.fix_seed(p)
        p.extra_generation_params["SD upscale tileSize"] = tileSize
        p.mask_blur = mask_blur
        initial_info = None
        seed = p.seed

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)

        upscaled_img = upscale(p, init_img, upscaler_index, tileSize, padding)

        devices.torch_gc()

        p.inpaint_full_res = False

        rows = math.ceil(p.height / tileSize)
        cols = math.ceil(p.width / tileSize)

        seams = 0
        if seam_pass_enabled:
            seams = rows-1 + cols - 1

        result_images = []
        state.job_count = rows*cols + seams
        for yi in range(rows):
            for xi in range(cols):
                p.width = tileSize
                p.height = tileSize
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = padding
                mask = Image.new("L", (upscaled_img.width, upscaled_img.height), "black")
                draw = ImageDraw.Draw(mask)
                draw.rectangle((
                    xi * tileSize,
                    yi * tileSize,
                    xi * tileSize + tileSize,
                    yi * tileSize + tileSize
                    ), fill="white")


                p.init_images = [upscaled_img]
                p.image_mask = mask
                processed = processing.process_images(p)
                initial_info = processed.info
                if (len(processed.images)>0):
                    upscaled_img = processed.images[0]

        result_images.append(upscaled_img)

        if seam_pass_enabled:
            for xi in range(1, cols):
                p.width = seam_pass_width + seam_pass_padding*2
                p.height = upscaled_img.height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = padding
                mask = Image.new("L", (upscaled_img.width, upscaled_img.height), "black")
                draw = ImageDraw.Draw(mask)
                draw.rectangle((
                    xi * tileSize - seam_pass_width//2,
                    0,
                    xi * tileSize + seam_pass_width//2,
                    mask.height
                ), fill="white")

                p.init_images = [upscaled_img]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    upscaled_img = processed.images[0]
            for yi in range(1, rows):
                p.width = upscaled_img.width
                p.height = seam_pass_width + seam_pass_padding*2
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = padding
                mask = Image.new("L", (upscaled_img.width, upscaled_img.height), "black")
                draw = ImageDraw.Draw(mask)
                draw.rectangle((
                    0,
                    yi * tileSize - seam_pass_width//2,
                    mask.width,
                    yi * tileSize + seam_pass_width//2
                ), fill="white")

                p.init_images = [upscaled_img]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    upscaled_img = processed.images[0]


        result_images.append(upscaled_img)
        processed = Processed(p, result_images, seed, initial_info)

        return processed
