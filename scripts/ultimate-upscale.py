import math
import copy
import time
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from modules import processing, shared, images, devices, scripts
from modules.processing import StableDiffusionProcessing
from modules.processing import Processed
from modules.shared import opts, state
from enum import Enum
from hashlib import md5
from collections import namedtuple


class USDUGrid():
    def __init__(self, image, padding, tile_width, tile_height, mask_blur):
        self.image = image
        self.padding = padding
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tiles = []
        self.rows_c = math.ceil(self.image.height / self.tile_height)
        self.cols_c = math.ceil(self.image.width / self.tile_width)
        self.mask_blur = mask_blur

    def add_row(self, row):
        self.tiles.append(row)

    def calc_crop_region(self, xi, yi):
        x1 = xi * self.tile_width - self.padding
        y1 = yi * self.tile_height - self.padding
        x2 = (xi + 1) * self.tile_width + self.padding
        y2 = (yi + 1) * self.tile_height + self.padding
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > self.image.width:
            x2 = self.image.width
        if y2 > self.image.height:
            y2 = self.image.height
        return x1, y1, x2, y2

    def split_grid(self):
        for yi in range(self.rows_c):
            row = USDUGridRow()
            for xi in range(self.cols_c):
                crop_region = self.calc_crop_region(xi, yi)
                x1, y1, x2, y2 = crop_region
                tile = self.image.crop(crop_region)
                col = USDUGridCol()
                col.add_tile(tile, (xi, yi), (x1, y1, x2-x1, y2-y1))
                row.add_col(col)
            self.add_row(row)

    def combine_grid(self):
        start_at = time.time()
        image = self.image
        for row in self.tiles:
            for col in row.cols:
                if (col.mask is None):
                    continue
                xi, yi = col.pos
                x, y, w, h = col.paste_to
                m_image = Image.new('RGB', (image.width, image.height))
                s_at = time.time()
                m_image.paste(col.mask.filter(ImageFilter.GaussianBlur(self.mask_blur)), (x,y))
                e_at = time.time()
                print(f"Gauss: {e_at - s_at}")
                np_mask = np.array(m_image)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                mask_for_overlay = Image.fromarray(np_mask)
                image_masked = Image.new('RGBa', (m_image.width, m_image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask_for_overlay.convert('L')))
                image_masked = image_masked.convert('RGBA')
                if col.paste_to is not None:
                    x, y, w, h = col.paste_to
                    base_image = Image.new('RGBA', (image_masked.width, image_masked.height))
                    image = images.resize_image(1, col.tile, w, h)
                    # image.save(f"F:/tt/o{col.pos}.png")
                    base_image.paste(image, (x, y))
                    image = base_image
                    # image.save(f"F:/tt1/o{col.pos}.png")

                image = image.convert('RGBA')
                image.alpha_composite(image_masked)
                image = image.convert('RGB')
        end_at = time.time()
        print(f"Combine time: {end_at - start_at}")
        return image

class USDUGridRow():
    def __init__(self):
        self.cols = []

    def add_col(self, col):
        self.cols.append(col)

class USDUGridCol():
    def __init__(self):
        self.tile = None
        self.pos = None
        self.paste_to = None
        self.tile_width = None
        self.tile_height = None
        self.mask = None
    
    def add_tile(self, tile, pos, paste_to):
        self.tile = tile
        self.pos = pos
        self.paste_to = paste_to
        self.tile_width = self.tile.width
        self.tile_height = self.tile.height

    def apply_overlay(self, image):
        if self.tile is None:
            return image

        return image

class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class USDUpscaler():

    def __init__(self, p, image, upscaler_index:int, save_redraw, save_seams_fix, tile_width, tile_height, padding) -> None:
        self.p:StableDiffusionProcessing = p
        self.image:Image = image
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))
        self.upscaler = shared.sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.redraw.padding = padding
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(p.height / self.redraw.tile_height)
        self.cols = math.ceil(p.width / self.redraw.tile_width)

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Check upscaler is not empty
        if self.upscaler.name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            return
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Upscaling iteration {index+1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(self.image, value, self.upscaler.data_path)
        # Resize image to set values
        self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)

    def setup_redraw(self, redraw_mode, mask_blur):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.p.mask_blur = mask_blur
        self.redraw.max_batch_size = self.p.batch_size

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self):
        images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt, opts.grid_format, info=self.initial_info, p=self.p)

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0
        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (self.cols - 1)

        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = self.upscaler.name
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = self.redraw.tile_width
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = self.redraw.tile_height
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = self.p.mask_blur
        self.p.extra_generation_params["Ultimate SD upscale padding"] = self.redraw.padding

    def process(self):
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.redraw.save:
            self.save_image()

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if self.seams_fix.save:
                self.save_image()
        state.end()

class USDURedraw():

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = width
        p.height = height
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi, padding, cols, rows, tile_width, tile_height, mask_blur):
        
        # x1 = 0
        # y1 = 0
        # x2 = self.tile_width
        # y2 = self.tile_height

        x1 = math.ceil(padding / 2) if xi > 0 else 0
        y1 = math.ceil(padding / 2) if yi > 0 else 0
        x2 = tile_width - math.ceil(padding / 2) if xi < (cols - 1) else tile_width
        y2 = tile_height - math.ceil(padding / 2) if yi < (rows - 1) else tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def chess_process_processing(self, p, image, rows, cols, polar, tiles):
        
        grid = USDUGrid(image, self.padding, self.tile_width, self.tile_height, p.mask_blur)
        grid.split_grid()
        print(len(grid.tiles))
        if len(grid.tiles) == 0:
            return image
        tiles_processing_data = {}
        for row in grid.tiles:
            for col in row.cols:
                xi, yi = col.pos
                if (tiles[yi][xi] == polar):
                    coords = self.calc_rectangle(xi, yi, self.padding, cols, rows, col.tile.width, col.tile.height, p.mask_blur)
                    idx = ''.join([str(value) for value in (col.tile.width, col.tile.height)]).join([str(value) for value in coords])
                    if tiles_processing_data.get(idx) == None:
                        tiles_processing_data[idx] = [coords, [], [], []]
                    tiles_processing_data[idx][1].append(col.tile)
                    tiles_processing_data[idx][2].append(xi)
                    tiles_processing_data[idx][3].append(yi)
        max_batch_size = self.max_batch_size

        v = 0
        for idxf, tile_data in tiles_processing_data.items():
            kk = 0
            for tile in tile_data[1]:
                kk += 1
            v += 1

        for idxf, tile_data in tiles_processing_data.items():
            mask, draw = self.init_draw(p, tile_data[1][0].width, tile_data[1][0].height)
            draw.rectangle(tile_data[0], fill="white")
            p.image_mask = mask
            batch_count = math.ceil(len(tile_data[1]) / max_batch_size)
            for i in range(batch_count):
                p.batch_size = max_batch_size if len(tile_data[1]) > max_batch_size * (i + 1) else len(tile_data[1]) - max_batch_size * i
                work_images = []
                begin_index = 0 if i == 0 else max_batch_size * (i)
                end_index = i * max_batch_size + p.batch_size
                for j in range(begin_index, end_index):
                    work_images.append(tile_data[1][j])
                p.init_images = work_images
                processed = processing.process_images(p)
                k = 0
                for j in range(begin_index, end_index):
                    row_index = tile_data[3][j]
                    col_index = tile_data[2][j]
                    grid.tiles[row_index].cols[col_index].tile = processed.images[k]
                    grid.tiles[row_index].cols[col_index].mask = copy.deepcopy(mask)
                    k += 1
        return grid.combine_grid()  

    def chess_process(self, p, image, rows, cols):
        tiles = []
        # calc tiles colors
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)
        image = self.chess_process_processing(p, image, rows, cols, True, tiles)
        image = self.chess_process_processing(p, image, rows, cols, False, tiles)
        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols)
        if self.mode == USDUMode.CHESS:
            return self.chess_process(p, image, rows, cols)

class USDUHalfTile():
    
    def __init__(self, tile_width, tile_height, image, padding, rows, cols) -> None:
        self.tile_width = tile_width
        self.tille_height = tile_height
        self.image = image
        self.padding = padding
        self.initial_info = None
        self.rows = rows
        self.cols = cols
    
    def init_draw(self, p, denoise, mask_blur):
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64
        p.denoising_strength = denoise
        p.mask_blur = mask_blur
        p.width = self.tile_width
        p.height = self.tile_height
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding

    def setup_gradient(self):
        gradient = Image.linear_gradient("L")

        self.row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        self.row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height//2), resample=Image.BICUBIC), (0, 0))
        self.row_gradient.paste(gradient.rotate(180).resize(
                (self.tile_width, self.tile_height//2), resample=Image.BICUBIC),
                (0, self.tile_height//2))
        self.col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        self.col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (0, 0))
        self.col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (self.tile_width//2, 0))
        
    def process(self, p, denoise, mask_blur):
        for yi in range(rows-1):
            for xi in range(cols):
                if state.interrupted:
                    break
                
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi*self.tile_width, yi*self.tile_height + self.tile_height//2))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]
        self.init_draw(p, denoise, mask_blur)
        self.setup_gradient()


class USDUSeamsFix():

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):

        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height//2), resample=Image.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize(
                (self.tile_width, self.tile_height//2), resample=Image.BICUBIC),
                (0, self.tile_height//2))
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (self.tile_width//2, 0))

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        

        for yi in range(rows):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi*self.tile_width+self.tile_width//2, yi*self.tile_height))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.BICUBIC)
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        #p.mask_blur = 0
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(gradient, (xi*self.tile_width + self.tile_width//2,
                                      yi*self.tile_height + self.tile_height//2))

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, cols, rows):

        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.BICUBIC)

        for xi in range(1, rows):
            if state.interrupted:
                    break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]
        for yi in range(1, cols):
            if state.interrupted:
                    break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        if USDUSFMode(self.mode) == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image

class Script(scripts.Script):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        target_size_types = [
            "From img2img2 settings",
            "Custom size",
            "Scale from image size"
        ]

        seams_fix_types = [
            "None",
            "Band pass",
            "Half tile offset pass",
            "Half tile offset pass + intersections"
        ]

        redrow_modes = [
            "Linear",
            "Chess",
            "None"
        ]

        info = gr.HTML(
            "<p style=\"margin-bottom:0.75em\">Will upscale the image depending on the selected target size type</p>")

        with gr.Row():
            target_size_type = gr.Dropdown(label="Target size type", choices=[k for k in target_size_types], type="index",
                                  value=next(iter(target_size_types)))

            custom_width = gr.Slider(label='Custom width', minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_height = gr.Slider(label='Custom height', minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_scale = gr.Slider(label='Scale', minimum=1, maximum=16, step=0.01, value=2, visible=False, interactive=True)

        gr.HTML("<p style=\"margin-bottom:0.75em\">Redraw options:</p>")
        with gr.Row():
            upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers],
                                value=shared.sd_upscalers[0].name, type="index")
        with gr.Row():
            redraw_mode = gr.Dropdown(label="Type", choices=[k for k in redrow_modes], type="index", value=next(iter(redrow_modes)))
            tile_width = gr.Slider(minimum=0, maximum=2048, step=64, label='Tile width', value=512)
            tile_height = gr.Slider(minimum=0, maximum=2048, step=64, label='Tile height', value=0)
            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=8)
            padding = gr.Slider(label='Padding', minimum=0, maximum=128, step=1, value=32)
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
            mask_blur_visible = fix_index == 2 or fix_index == 3
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

        def select_scale_type(scale_index):
            is_custom_size = scale_index == 1
            is_custom_scale = scale_index == 2

            return [gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_scale),
                    ]

        target_size_type.change(
            fn=select_scale_type,
            inputs=target_size_type,
            outputs=[custom_width, custom_height, custom_scale]
        )

        return [info, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
                upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
                seams_fix_type, target_size_type, custom_width, custom_height, custom_scale]

    def run(self, p, _, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
            upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
            seams_fix_type, target_size_type, custom_width, custom_height, custom_scale):

        # Init
        processing.fix_seed(p)
        devices.torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        # p.inpainting_fill = 1

        seed = p.seed

        # Init image
        init_img = p.init_images[0]
        if init_img == None:
            return Processed(p, [], seed, "Empty image")
        init_img = images.flatten(init_img, opts.img2img_background_color)

        #override size
        if target_size_type == 1:
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
            p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(p, init_img, upscaler_index, save_upscaled_image, save_seams_fix_image, tile_width, tile_height, padding)
        upscaler.upscale()
        
        # Drawing
        upscaler.setup_redraw(redraw_mode, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(p, result_images, seed, upscaler.initial_info if upscaler.initial_info is not None else "")

