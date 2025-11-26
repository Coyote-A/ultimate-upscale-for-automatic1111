# Ultimate + ADetailer Combo (A1111)

Extensão para o AUTOMATIC1111 que combina:
- Ultimate SD Upscale (upscale em tiles com fix de costuras).
- ADetailer (detecção de rosto/mão/etc. e inpaint focado).

Não usa Hires fix: gera direto na resolução-alvo definida na UI da extensão.

## Instalação
1. Coloque esta pasta em `stable-diffusion-webui/extensions/`.
2. Certifique-se de ter o Ultimate SD Upscale (já incluído em `scripts/ultimate-upscale.py`).
3. Tenha um modelo do ADetailer em `models/adetailer` (YOLO/YOLO-World) ou use mediapipe (funciona sem baixar modelo).
4. Reinicie o WebUI.

## Uso (txt2img e img2img)
1. Abra o acordeão “Ultimate + ADetailer” e marque “Ativar combo”.
2. Escolha o modo:
   - “Refinar com ADetailer e depois fazer upscale”: detecta → inpaint nas máscaras → upscale global.
   - “Tiles guiados pela detecção”: detecta → tiles menores nas áreas detectadas → upscale global.
3. Defina escala final (ex.: 2.0) e upscaler.
4. Ajuste detector (modelo, confiança, dilatar/blur) e denoise do refino.
5. Ajuste tiles/padding/blur do Ultimate.
6. Gere normalmente (sem Hires fix).

## Parâmetros principais
- Ativar combo: liga/desliga a extensão.
- Modo: “Refinar → Upscale” ou “Tiles guiados”.
- Escala final: fator de upscale direto.
- Detector: modelo ADetailer ou mediapipe; confiança; dilatar/erodir; blur; merge/invert.
- Denoise do refino: força de inpaint nas áreas detectadas.
- Ultimate: upscaler, tile width/height, padding e mask blur (redraw).

## Notas
- Se nada for detectado, cai no upscale normal.
- Para economizar VRAM, use mediapipe ou um modelo pequeno; tiles maiores fora das áreas ajudam em GPUs menores.
- Metadados seguem do job original; versões intermediárias não são salvas automaticamente.
