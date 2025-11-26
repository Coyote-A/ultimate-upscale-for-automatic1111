# Ultimate + ADetailer Combo (A1111)

Extensão “always on” para txt2img/img2img no AUTOMATIC1111 que une:
- **Ultimate SD Upscale**: upscale em tiles com correção de costuras.
- **ADetailer**: detecção (rostos, mãos, etc.) e inpaint focado.

Gera direto na resolução final (sem Hires fix). Se a extensão estiver ativa, o pipeline dispara automaticamente após o txt2img/img2img.

## Instalação
1. Copie esta pasta para `stable-diffusion-webui/extensions/`.
2. Já incluído: `scripts/ultimate-upscale.py`. Mantenha-o em `scripts/`.
3. Modelos ADetailer: coloque `.pt` em `models/adetailer` (YOLO/YOLO-World) ou use mediapipe (não requer download).
4. Reinicie o WebUI.

## Uso rápido
1) Abra o acordeão “Ultimate + ADetailer” e marque **Ativar combo**.  
2) Escolha o **Modo**:
   - **Refinar com ADetailer e depois fazer upscale**: detecta → inpaint nas máscaras → upscale global.
   - **Tiles guiados pela detecção**: detecta → usa máscaras para tiles menores nas áreas detectadas → upscale global.
3) Defina a **Escala final** (ex.: 2.0) e o **Modelo de upscale** (ESRGAN/SwinIR/4x* instalados).  
4) Ajuste o **Detector** (modelo, confiança, dilatar/erodir, blur, merge/invert) e o **Denoise do refino**.  
5) Ajuste o Ultimate: **tile width/height**, **padding**, **mask blur** e **modo de redraw** (Linear/Chess/None).  
6) Gere normalmente (sem Hires fix).

## Parâmetros principais
- **Ativar combo**: liga/desliga a extensão.
- **Modo**: “Refinar → Upscale” ou “Tiles guiados”.
- **Escala final**: fator direto de upscale.
- **Detector ADetailer**: modelo YOLO/mediapipe; confiança; dilatar/erodir; blur; merge/invert.
- **Denoise do refino**: força de inpaint nas áreas detectadas.
- **Ultimate**: upscaler (4x-Nickel/R-ESRGAN/SwinIR/etc.), tile width/height, padding, mask blur, modo de redraw (Linear/Chess/None).

## Perfil sugerido para A10 (24 GB)
- Escala 2–3× direto; detecção em alta.  
- Tiles fora das máscaras: ~768; dentro (tiles guiados): 320–384.  
- Padding 64 e mask blur 10–12 no redraw.  
- Denoise de refino 0.3–0.45; dilatação 4–8 px e blur 8–12 nas máscaras.  
- Redraw: Linear (mais previsível) ou Chess (melhor nas costuras) conforme teste.

## Notas
- Se nada for detectado, cai no upscale normal.
- Para economizar VRAM: mediapipe ou modelo leve; tiles maiores fora das áreas.  
- Metadados seguem do job original; versões intermediárias não são salvas automaticamente.
