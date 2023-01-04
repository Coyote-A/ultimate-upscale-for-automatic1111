import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageDraw

from modules import processing, shared, sd_samplers, images, devices
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from collections import namedtuple

def get_factor(num):
    if num == 1:
        return 2
    if num % 4 == 0:
        return 4
    if num % 3 == 0:
        return 3
    if num % 2 == 0:
        return 2
    return 0

def upscale(p, init_img, upscaler_index, padding):
    scale_factor = max(p.width, p.height) // max(init_img.width, init_img.height)
    print(f"Canva size: {p.width}x{p.height}")
    print(f"Image size: {init_img.width}x{init_img.height}")
    print(f"Scale factor: {scale_factor}")

    upscaler = shared.sd_upscalers[upscaler_index]

    p.extra_generation_params["SD upscale overlap"] = padding
    p.extra_generation_params["SD upscale upscaler"] = upscaler.name

    upscaled_img = init_img
    if upscaler.name == "None":
        return upscaled_img.resize((p.width, p.height), resample=Image.LANCZOS)
    
    current_scale = 1
    iteration = 0
    current_scale_factor = get_factor(scale_factor)
    while current_scale_factor == 0:
        scale_factor += 1
        current_scale_factor = get_factor(scale_factor)
    while current_scale < scale_factor:
        iteration += 1
        current_scale_factor = get_factor(scale_factor // current_scale)
        current_scale = current_scale * current_scale_factor
        if current_scale_factor == 0:
            break
        print(f"Upscaling iteration {iteration} with scale factor {current_scale_factor}")
        upscaled_img = upscaler.scaler.upscale(upscaled_img, current_scale_factor, upscaler.data_path)

    return upscaled_img.resize((p.width, p.height), resample=Image.LANCZOS)


def redraw_image(p, upscaled_img, rows, cols, tile_size, padding):
    initial_info = None
    for yi in range(rows):
        for xi in range(cols):
            p.width = tile_size
            p.height = tile_size
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = padding
            mask = Image.new("L", (upscaled_img.width, upscaled_img.height), "black")
            draw = ImageDraw.Draw(mask)
            draw.rectangle((
                xi * tile_size,
                yi * tile_size,
                xi * tile_size + tile_size,
                yi * tile_size + tile_size
            ), fill="white")

            p.init_images = [upscaled_img]
            p.image_mask = mask
            processed = processing.process_images(p)
            initial_info = processed.info
            if (len(processed.images) > 0):
                upscaled_img = processed.images[0]

    return upscaled_img, initial_info


def redraw_middle_offset_image(p, upscaled_img, rows, cols, tile_size, padding, seams_fix_denoise, seams_fix_mask_blur):
    initial_info = None
    gradient = Image.linear_gradient("L")

    row_gradient = Image.new("L", (tile_size, tile_size), "black")
    row_gradient.paste(gradient.resize((tile_size, tile_size//2), resample=Image.BICUBIC), (0, 0))
    row_gradient.paste(gradient.rotate(180).resize((tile_size, tile_size//2), resample=Image.BICUBIC), (0, tile_size//2))
    col_gradient = Image.new("L", (tile_size, tile_size), "black")
    col_gradient.paste(gradient.rotate(90).resize((tile_size//2, tile_size), resample=Image.BICUBIC), (0, 0))
    col_gradient.paste(gradient.rotate(270).resize((tile_size//2, tile_size), resample=Image.BICUBIC), (tile_size//2, 0))

    p.denoising_strength = seams_fix_denoise
    p.mask_blur = seams_fix_mask_blur

    for yi in range(rows-1):
        for xi in range(cols):
            p.width = tile_size
            p.height = tile_size
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = padding
            mask = Image.new("L", (upscaled_img.width, upscaled_img.height), "black")
            mask.paste(row_gradient, (xi*tile_size, yi*tile_size + tile_size//2))

            p.init_images = [upscaled_img]
            p.image_mask = mask
            processed = processing.process_images(p)
            initial_info = processed.info
            if (len(processed.images) > 0):
                upscaled_img = processed.images[0]

    for yi in range(rows):
        for xi in range(cols-1):
            p.width = tile_size
            p.height = tile_size
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = padding
            mask = Image.new("L", (upscaled_img.width, upscaled_img.height), "black")
            mask.paste(col_gradient, (xi*tile_size+tile_size//2, yi*tile_size))

            p.init_images = [upscaled_img]
            p.image_mask = mask
            processed = processing.process_images(p)
            initial_info = processed.info
            if (len(processed.images) > 0):
                upscaled_img = processed.images[0]

    return upscaled_img, initial_info


def seam_draw(p, upscaled_img, seams_fix_width, seams_fix_padding, seams_fix_denoise, tile_size, cols, rows):
    p.denoising_strength = seams_fix_denoise
    p.mask_blur = 0

    gradient = Image.linear_gradient("L")
    mirror_gradient = Image.new("L", (256, 256), "black")
    mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
    mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

    row_gradient = mirror_gradient.resize((upscaled_img.width, seams_fix_width), resample=Image.BICUBIC)
    col_gradient = mirror_gradient.rotate(90).resize((seams_fix_width, upscaled_img.height), resample=Image.BICUBIC)

    for xi in range(1, cols):
        p.width = seams_fix_width + seams_fix_padding * 2
        p.height = upscaled_img.height
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = seams_fix_padding
        mask = Image.new("L", (upscaled_img.width, upscaled_img.height), "black")
        mask.paste(col_gradient, (xi * tile_size - seams_fix_width // 2, 0))

        p.init_images = [upscaled_img]
        p.image_mask = mask
        processed = processing.process_images(p)
        if (len(processed.images) > 0):
            upscaled_img = processed.images[0]
    for yi in range(1, rows):
        p.width = upscaled_img.width
        p.height = seams_fix_width + seams_fix_padding * 2
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = seams_fix_padding
        mask = Image.new("L", (upscaled_img.width, upscaled_img.height), "black")
        mask.paste(row_gradient, (0, yi * tile_size - seams_fix_width // 2))

        p.init_images = [upscaled_img]
        p.image_mask = mask
        processed = processing.process_images(p)
        if (len(processed.images) > 0):
            upscaled_img = processed.images[0]
    return upscaled_img

class Script(scripts.Script):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        seams_fix_types = [
            "None",
            "Band pass", 
            "Half tile offset pass"
        ]
        
        info = gr.HTML(
            "<p style=\"margin-bottom:0.75em\">Will upscale the image to selected with and height</p>")
        gr.HTML("<p style=\"margin-bottom:0.75em\">Redraw options:</p>")
        with gr.Row():
            redraw_enabled = gr.Checkbox(label="Enabled", value=True)
            tile_size = gr.Slider(minimum=256, maximum=2048, step=64, label='Tile size', value=512)
            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=8)
            padding = gr.Slider(label='Padding', minimum=0, maximum=128, step=1, value=32)
        upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers],
                                value=shared.sd_upscalers[0].name, type="index")
        gr.HTML("<p style=\"margin-bottom:0.75em\">Seams fix:</p>")
        with gr.Row():
            seams_fix_type = gr.Dropdown(label="Type", choices=[k for k in seams_fix_types], type="index", value=next(iter(seams_fix_types)))
            seams_fix_denoise = gr.Slider(label='Denoise', minimum=0, maximum=1, step=0.01, value=0.35, visible=False, interactive=True)
            seams_fix_width = gr.Slider(label='Width', minimum=0, maximum=128, step=1, value=64, visible=False, interactive=True)
            seams_fix_mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=False, interactive=True)
            seams_fix_padding = gr.Slider(label='Padding', minimum=0, maximum=128, step=1, value=16, visible=False, interactive=True)
        gr.HTML("<p style=\"margin-bottom:0.75em\">Save options:</p>")
        with gr.Row():
            save_upscaled_image = gr.Checkbox(label="Upscaled", value=True)
            save_seams_fix_image = gr.Checkbox(label="Seams fix", value=False)

        def select_fix_type(fix_index):
            all_visible = fix_index != 0
            mask_blur_visible = fix_index == 2
            width_visible = fix_index == 1

            return [gr.update(visible=all_visible),
                    gr.update(visible=width_visible),
                    gr.update(visible=mask_blur_visible),
                    gr.update(visible=all_visible)]

        seams_fix_type.change(
            fn=select_fix_type,
            inputs=seams_fix_type,
            outputs=[seams_fix_denoise, seams_fix_width, seams_fix_mask_blur, seams_fix_padding]
        )

        return [info, tile_size, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
                upscaler_index, save_upscaled_image, redraw_enabled, save_seams_fix_image, seams_fix_mask_blur, 
                seams_fix_type]

    def run(self, p, _, tile_size, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding, 
            upscaler_index, save_upscaled_image, redraw_enabled, save_seams_fix_image, seams_fix_mask_blur, 
            seams_fix_type):
        processing.fix_seed(p)
        p.extra_generation_params["SD upscale tileSize"] = tile_size
        p.mask_blur = mask_blur
        seed = p.seed

        initial_info = None

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)

        # Upscaling
        upscaled_img = upscale(p, init_img, upscaler_index, padding)

        # Drawing
        devices.torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True

        p.inpaint_full_res = False

        rows = math.ceil(p.height / tile_size)
        cols = math.ceil(p.width / tile_size)

        print(f"Tiles amount: {rows * cols}")
        print(f"Grid: {rows}x{cols}")
        print(f"Seam path: {seams_fix_type}")

        seams = 0
        if seams_fix_type > 0:
            seams = rows - 1 + cols - 1
        state.job_count = ((rows * cols) if redraw_enabled else 0) + (seams if seams_fix_type == 1 else 0) + (
            (rows * (cols - 1) + (rows - 1) * cols) if seams_fix_type == 2 else 0)

        result_images = []
        result_image = upscaled_img
        if redraw_enabled:
            result_image, initial_info = redraw_image(p, upscaled_img, rows, cols, tile_size, padding)
        result_images.append(result_image)
        if save_upscaled_image:
            images.save_image(result_image, p.outpath_samples, "", seed, p.prompt, opts.grid_format, info=initial_info, p=p)

        if seams_fix_type == 2:
            print(f"Starting offset pass drawing")
            result_image, initial_info = redraw_middle_offset_image(p, result_image, rows, cols, tile_size, seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur)
            result_images.append(result_image)
            if save_seams_fix_image:
                images.save_image(result_image, p.outpath_samples, "", seed, p.prompt, opts.grid_format, info=initial_info, p=p)

        if seams_fix_type == 1:
            print(f"Starting bands pass drawing")
            result_image = seam_draw(p, result_image, seams_fix_width, seams_fix_padding, seams_fix_denoise, tile_size, cols, rows)
            result_images.append(result_image)
            if save_seams_fix_image:
                images.save_image(result_image, p.outpath_samples, "", seed, p.prompt, opts.grid_format, info=initial_info, p=p)

        processed = Processed(p, result_images, seed, initial_info if initial_info is not None else "")

        return processed
