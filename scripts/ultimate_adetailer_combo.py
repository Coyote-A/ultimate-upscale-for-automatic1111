import importlib.util
import math
from copy import copy
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from PIL import Image

from modules import images, paths, scripts, shared
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    process_images,
)

# ADetailer bits
try:
    from adetailer import get_models, mediapipe_predict, ultralytics_predict
    from adetailer.args import MASK_MERGE_INVERT
    from adetailer.mask import mask_preprocess
except Exception:
    get_models = mediapipe_predict = ultralytics_predict = None
    MASK_MERGE_INVERT = ["None", "Merge", "Merge and Invert"]
    mask_preprocess = None


def copy_extra_params(extra_params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extra_params.items() if not callable(v)}


def load_ultimate_module():
    """Dynamically load ultimate-upscale.py (file name has a dash)."""
    module_path = Path(__file__).parent / "ultimate-upscale.py"
    spec = importlib.util.spec_from_file_location("ultimate_upscale_dyn", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Ultimate SD Upscale script not found.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ultimate_module = None


def get_ultimate_module():
    global _ultimate_module
    if _ultimate_module is None:
        _ultimate_module = load_ultimate_module()
    return _ultimate_module


def resolve_target_size(image: Image.Image, scale: float) -> tuple[int, int]:
    w = math.ceil(image.width * scale / 64) * 64
    h = math.ceil(image.height * scale / 64) * 64
    return w, h


class Script(scripts.Script):
    """Combo: ADetailer -> Ultimate SD Upscale (txt2img/img2img)."""

    def title(self):
        return "Ultimate + ADetailer Combo"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        modes = [
            ("Refinar com ADetailer e depois fazer upscale", "refine_upscale"),
            ("Tiles guiados pela detecção", "guided_tiles"),
        ]

        with gr.Accordion("Ultimate + ADetailer", open=False):
            enable = gr.Checkbox(label="Ativar combo", value=False)
            mode = gr.Dropdown(
                label="Modo",
                choices=[m[0] for m in modes],
                value=modes[0][0],
                type="value",
            )
            scale = gr.Slider(
                label="Escala final",
                minimum=1.0,
                maximum=4.0,
                step=0.1,
                value=2.0,  # A10: 2x como base; subir se quiser 3-4x
            )

            # ADetailer
            ad_model_choices = self._list_ad_models()
            ad_model = gr.Dropdown(
                label="Modelo ADetailer",
                choices=ad_model_choices,
                value=ad_model_choices[0] if ad_model_choices else "None",
                type="value",
            )
            ad_confidence = gr.Slider(
                label="Confiança do detector",
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=0.4,  # A10: pode usar threshold um pouco maior
            )
            ad_dilate = gr.Slider(
                label="Dilatar/Erodir máscara",
                minimum=-32,
                maximum=32,
                step=1,
                value=6,  # A10: mais margem para refino
            )
            ad_mask_blur = gr.Slider(
                label="Blur da máscara (inpaint)",
                minimum=0,
                maximum=32,
                step=1,
                value=10,  # A10: feather mais suave em alta resolução
            )
            ad_merge = gr.Radio(
                label="Mesclar/Inverter máscaras",
                choices=MASK_MERGE_INVERT,
                value=MASK_MERGE_INVERT[0],
                type="value",
            )
            ad_denoise = gr.Slider(
                label="Denoise para refino",
                minimum=0.05,
                maximum=1.0,
                step=0.05,
                value=0.35,  # A10: range equilibrado para rosto/mão
            )

            # Ultimate
            upscaler_index = gr.Radio(
                label="Modelo de upscale (sd_upscalers: 4x-Nickel, R-ESRGAN, etc.)",
                choices=[x.name for x in shared.sd_upscalers],
                value=shared.sd_upscalers[0].name if shared.sd_upscalers else "",
                type="index",
                info="Escolha o upscaler instalado (ESRGAN/SwinIR/4x*).",
            )
            tile_width = gr.Slider(
                label="Tile width",
                minimum=64,
                maximum=2048,
                step=64,
                value=768,  # A10: tiles maiores fora das áreas
            )
            tile_height = gr.Slider(
                label="Tile height (0 = usar width)",
                minimum=0,
                maximum=2048,
                step=64,
                value=0,
            )
            padding = gr.Slider(
                label="Padding (redraw)",
                minimum=0,
                maximum=256,
                step=4,
                value=64,  # A10: mais padding para costura suave
            )
            mask_blur = gr.Slider(
                label="Mask blur (redraw)",
                minimum=0,
                maximum=64,
                step=1,
                value=12,  # A10: feather moderado nas bordas dos tiles
            )
            redraw_mode = gr.Dropdown(
                label="Modo de redraw",
                choices=["Linear", "Chess", "None"],
                value="Linear",
                type="value",
                info="Linear percorre todos os tiles; Chess alterna tiles para reduzir costuras; None desativa redraw.",
            )
        return [
            enable,
            mode,
            scale,
            ad_model,
            ad_confidence,
            ad_dilate,
            ad_mask_blur,
            ad_merge,
            ad_denoise,
            upscaler_index,
            tile_width,
            tile_height,
            padding,
            mask_blur,
            redraw_mode,
        ]

    def _list_ad_models(self):
        if get_models is None:
            return ["None"]
        model_dir = Path(paths.models_path) / "adetailer"
        extra_dirs = shared.opts.data.get("ad_extra_models_dir", "")
        dirs = [model_dir, *extra_dirs.split("|")] if extra_dirs else [model_dir]
        models = list(get_models(*dirs, huggingface=False).keys())
        models.append("None")
        return models

    def _detect(self, image: Image.Image, model: str, conf: float):
        if (
            model in (None, "", "None")
            or mediapipe_predict is None
            or ultralytics_predict is None
        ):
            return None

        if model.lower().startswith("mediapipe"):
            return mediapipe_predict(model, image, conf)

        model_dir = Path(paths.models_path) / "adetailer"
        extra_dirs = shared.opts.data.get("ad_extra_models_dir", "")
        dirs = [model_dir, *extra_dirs.split("|")] if extra_dirs else [model_dir]
        mapping = get_models(*dirs, huggingface=False)
        if model not in mapping:
            return None
        model_path = mapping[model]
        try:
            return ultralytics_predict(model_path, image, conf)
        except Exception:
            return None

    def _inpaint_masks(
        self,
        p,
        image: Image.Image,
        masks: list[Image.Image],
        mask_blur: int,
        denoise: float,
    ) -> Image.Image:
        if not masks:
            return image

        refined = image
        for mask in masks:
            i2i = StableDiffusionProcessingImg2Img(
                init_images=[refined],
                resize_mode=0,
                denoising_strength=denoise,
                mask=mask,
                mask_blur=mask_blur,
                inpainting_fill=1,
                inpaint_full_res=True,
                inpaint_full_res_padding=0,
                inpainting_mask_invert=0,
                sd_model=p.sd_model,
                outpath_samples=p.outpath_samples,
                outpath_grids=p.outpath_grids,
                prompt=p.prompt,
                negative_prompt=p.negative_prompt,
                styles=p.styles,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=p.sampler_name,
                batch_size=1,
                n_iter=1,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                width=refined.width,
                height=refined.height,
                restore_faces=getattr(p, "restore_faces", False),
                tiling=p.tiling,
                extra_generation_params=copy_extra_params(p.extra_generation_params),
                do_not_save_samples=True,
                do_not_save_grid=True,
            )
            i2i._ultimate_adetailer_disabled = True
            i2i.scripts = p.scripts
            i2i.script_args = p.script_args
            processed = process_images(i2i)
            if processed.images:
                refined = processed.images[0]
        return refined

    def _ultimate_upscale(
        self,
        p,
        image: Image.Image,
        target_w: int,
        target_h: int,
        upscaler_index: int,
        tile_w: int,
        tile_h: int,
        padding: int,
        mask_blur: int,
        redraw_mode_label: str,
    ) -> Image.Image:
        ultimate = get_ultimate_module()
        p_up = StableDiffusionProcessingImg2Img(
            init_images=[image],
            resize_mode=0,
            denoising_strength=p.denoising_strength if hasattr(p, "denoising_strength") else 0.35,
            mask=None,
            mask_blur=mask_blur,
            inpainting_fill=1,
            inpaint_full_res=True,
            inpaint_full_res_padding=padding,
            inpainting_mask_invert=0,
            sd_model=p.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=p.prompt,
            negative_prompt=p.negative_prompt,
            styles=p.styles,
            seed=p.seed,
            subseed=p.subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=p.sampler_name,
            batch_size=1,
            n_iter=1,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            width=target_w,
            height=target_h,
            restore_faces=getattr(p, "restore_faces", False),
            tiling=p.tiling,
            extra_generation_params=copy_extra_params(p.extra_generation_params),
            do_not_save_samples=True,
            do_not_save_grid=True,
        )
        p_up._ultimate_adetailer_disabled = True
        p_up.scripts = p.scripts
        p_up.script_args = p.script_args

        upscaler = ultimate.USDUpscaler(
            p_up,
            image,
            upscaler_index,
            save_redraw=False,
            save_seams_fix=False,
            tile_width=tile_w,
            tile_height=tile_h,
        )
        upscaler.upscale()
        redraw_enum = ultimate.USDUMode.LINEAR
        if redraw_mode_label == "Chess":
            redraw_enum = ultimate.USDUMode.CHESS
        elif redraw_mode_label == "None":
            redraw_enum = ultimate.USDUMode.NONE
        upscaler.setup_redraw(redraw_mode=redraw_enum.value, padding=padding, mask_blur=mask_blur)
        upscaler.setup_seams_fix(padding=0, denoise=0.4, mask_blur=0, width=0, mode=ultimate.USDUSFMode.NONE.value)
        upscaler.process()
        return upscaler.result_images[-1] if upscaler.result_images else image

    def postprocess_image(
        self,
        p,
        pp,
        enable,
        mode_label,
        scale,
        ad_model,
        ad_confidence,
        ad_dilate,
        ad_mask_blur,
        ad_merge,
        ad_denoise,
        upscaler_index,
        tile_width,
        tile_height,
        padding,
        mask_blur,
        redraw_mode,
    ):
        if getattr(p, "_ultimate_adetailer_disabled", False):
            return
        if not enable:
            return

        mode = (
            "refine_upscale"
            if "Refinar" in mode_label
            else "guided_tiles"
        )

        base = images.flatten(pp.image, shared.opts.img2img_background_color)

        pred = self._detect(base, ad_model, ad_confidence)
        masks = []
        if pred and getattr(pred, "masks", None) and mask_preprocess is not None:
            masks = mask_preprocess(
                pred.masks,
                kernel=int(ad_dilate),
                x_offset=0,
                y_offset=0,
                merge_invert=ad_merge,
            )

        # Mode 2: refino antes do upscale
        if mode == "refine_upscale":
            base = self._inpaint_masks(p, base, masks, ad_mask_blur, ad_denoise)

        target_w, target_h = resolve_target_size(base, scale)

        # Mode 4: tiles guiados -> tiles menores se houve detecção
        guided_tile_w = tile_width
        guided_tile_h = tile_height
        if mode == "guided_tiles" and masks:
            # Heurística: tiles menores se há máscaras
            guided_tile_w = max(128, min(tile_width, base.width // 3))
            guided_tile_h = max(128, min(tile_height or tile_width, base.height // 3))

        final_img = self._ultimate_upscale(
            p,
            base,
            target_w,
            target_h,
            upscaler_index,
            guided_tile_w,
            guided_tile_h,
            int(padding),
            int(mask_blur),
            redraw_mode,
        )
        pp.image = final_img
