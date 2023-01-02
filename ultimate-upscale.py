import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw

from modules import processing, shared, sd_samplers, images, devices
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):
    def title(self):
        return "SD upscale 5"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>")
        tileSize = gr.Slider(minimum=256, maximum=2048, step=64, label='Tile size', value=512)
        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=8)
        padding = gr.Slider(label='Padding', minimum=0, maximum=128, step=1, value=32)
        seam_pass_enabled = gr.Checkbox(label= "Seam pass enabled")
        seam_pass_width = gr.Slider(label='Seam pass width', minimum=0, maximum=128, step=1, value=16)
        seam_pass_denoise = gr.Slider(label='Seam pass denoise', minimum=0, maximum=1, step=0.01, value=0.25)
        seam_pass_padding = gr.Slider(label='Seam pass padding', minimum=0, maximum=128, step=1, value=32)

        return [info, tileSize, mask_blur, padding, seam_pass_enabled, seam_pass_width, seam_pass_denoise, seam_pass_padding]

    def run(self, p, _, tileSize, mask_blur, padding, seam_pass_enabled, seam_pass_width, seam_pass_denoise, seam_pass_padding):
        processing.fix_seed(p)
        p.extra_generation_params["SD upscale tileSize"] = tileSize
        p.mask_blur = mask_blur
        initial_info = None
        seed = p.seed

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)

        upscaled_img = init_img.resize((p.width, p.height), resample=Image.LANCZOS)

        devices.torch_gc()

        p.inpaint_full_res = False

        rows = math.ceil(p.height / tileSize)
        cols = math.ceil(p.width / tileSize)

        seams = 0
        if seam_pass_enabled:
            seams = rows-1 + cols - 1

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
                    xi * tileSize - seam_pass_width//2,
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
                    xi * tileSize - seam_pass_width//2,
                    0,
                    xi * tileSize - seam_pass_width//2,
                    mask.height
                ), fill="white")

                p.init_images = [upscaled_img]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    upscaled_img = processed.images[0]


        result_images = []
        result_images.append(upscaled_img)
        processed = Processed(p, result_images, seed, initial_info)

        return processed
