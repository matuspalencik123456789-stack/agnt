# ComfyUI: AI modelka z 2 fotiek (telo z fotky 1 + tvár z fotky 2)

| Súbor | Kedy použiť |
|---|---|
| `workflows/01_modelka_telo+tvar_flux2klein_BFS_16GB.json` | **Hlavný pre Mac so 16 GB.** fp8 model, fotky 0.75 MP. |
| `workflows/01_modelka_telo+tvar_flux2klein_BFS.json` | To isté pre 24 GB+ (bf16 model, 1 MP). Len core uzly. |
| `stiahni_modely_mac.sh` | Stiahne modely do správnych priečinkov a vloží workflowy do ComfyUI. |
| `workflows/02_modelka_telo+tvar_reactor_zaloha.json` | Záloha s modelmi, ktoré už máš (ReActor + Flux.1 dev). Slabšia kvalita, licencia len nekomerčná. |
| `workflows/api/*.api.json` | Na skriptovanie cez `/prompt` API. Do UI ich nepreťahuj. |
| `build_workflows.py` | Generátor. Po úprave spusti `python3 comfyui/build_workflows.py`. |

Do ComfyUI: pretiahni `.json` (bez `.api`) do okna prehliadača.

## Spustenie na Macu so 16 GB RAM (krok za krokom)

1. Stiahni skript (Terminal):
   ```bash
   curl -L -o ~/stiahni_modely_mac.sh https://raw.githubusercontent.com/matuspalencik123456789-stack/agnt/claude/lucid-goldberg-86v1xi/comfyui/stiahni_modely_mac.sh
   ```
2. Spusti ho s cestou k ComfyUI (priečinok, v ktorom je `main.py`). Stiahne ~12.8 GB a vloží workflowy do ComfyUI:
   ```bash
   bash ~/stiahni_modely_mac.sh ~/ComfyUI
   ```
3. Spusti ComfyUI:
   ```bash
   cd ~/ComfyUI && source .venv/bin/activate && python main.py
   ```
4. Otvor http://127.0.0.1:8188 → klávesa **W** (panel Workflows) → **modelka_telo+tvar_16GB**.
5. Uzol **FOTKA 1 – TELO** → *choose file to upload* → fotka s telom. Uzol **FOTKA 2 – TVÁR** → fotka tváre.
6. **Run** (alebo Cmd+Enter). Prvé spustenie trvá dlhšie, lebo sa načítavajú modely.
   Výsledok je v `ComfyUI/output/modelka/`.

Keď niečo nejde:
- **Červený uzol / „Value not in list“** → súbor nie je v správnom priečinku alebo má iný názov. Stlač **R** (refresh) alebo reštartuj ComfyUI.
- **„MPS backend out of memory“** → v uzloch „→ 0.75 MP“ daj 0.6 a zavri ostatné aplikácie. Až potom skús
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python main.py`. Pozor: táto premenná vypne limit pamäte a Mac môže zamrznúť.

## 01 – FLUX.2 Klein 4B + BFS Head Swap LoRA

Model je editačný a dostane obe fotky naraz: *Picture 1 = telo*, *Picture 2 = hlava*.
Hlavu nenalepí ako koláž, ale dogeneruje ju. Svetlo, krk, tón pokožky a pózu hlavy
preberie z fotky 1. Tvár, vlasy, farbu očí a nos preberie z fotky 2.

### Modely na stiahnutie (Flux.1 súbory sa tu nedajú použiť)

| Súbor | Veľkosť | Priečinok v `ComfyUI/models/` |
|---|---|---|
| [flux-2-klein-4b.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/flux-2-klein-4b.safetensors) (bf16, odporúčané na Mac) | 7.8 GB | `diffusion_models/` |
| alebo [flux-2-klein-4b-fp8.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors) (pri 16 GB RAM) | 4.1 GB | `diffusion_models/` |
| [qwen_3_4b.safetensors](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) | ~8 GB | `text_encoders/` |
| [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/vae/flux2-vae.safetensors) | ~0.3 GB | `vae/` |
| [bfs_head_v1_flux-klein_4b.safetensors](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_4b.safetensors) | malý | `loras/` |

Ak použiješ fp8 súbor, prepni ho v uzle *Model: FLUX.2 Klein 4B*.

### Nastavenia (z oficiálnej šablóny ComfyUI pre Klein 4B distilled)
- 4 kroky, `euler`, CFG 1.0, negatív `ConditioningZeroOut`.
- Obe fotky sa zmenšia na ~1 MP (násobky 16). Výstup má pomer strán fotky 1.
- Prompt je oficiálny BFS `head_swap:` prompt. Začiatok nemeň. Na koniec môžeš doplniť výraz tváre.

### Ladenie
| Problém | Čo skúsiť |
|---|---|
| Tvár nie je dosť podobná fotke 2 | LoRA 1.1–1.3; ostrejšia fotka 2; orež fotku 2 na hlavu + vlasy |
| Mení sa telo / oblečenie / pozadie | LoRA 0.8–0.9 |
| Zlý výraz tváre | doplň do promptu napr. `smiling softly, mouth closed` |
| Viditeľný šev na krku / iný tón pleti | vyber fotku 2 s podobným svetlom; skús iný seed |
| Málo RAM / pády na Macu | fp8 súbor; `megapixels` 0.75 v oboch „→ 1 MP“ uzloch |

### Fotky – čo funguje najlepšie
- **Fotka 1 (telo):** tvár je viditeľná a nie je príliš malá (polcelok je lepší než celá postava z diaľky).
- **Fotka 2 (tvár):** ostrá, rovnomerné svetlo, bez okuliarov a rúk pred tvárou, podobný uhol hlavy ako na fotke 1.
- Na konzistentnú AI modelku používaj stále tú istú fotku 2. Najlepšie je vygenerovať si vlastnú AI tvár.

## 02 – záloha: ReActor (inswapper_128) + Flux.1 dev doladenie
ReActor vymení tvár. Potom Flux.1 dev pri denoise 0.20 vráti textúru pokožky, lebo inswapper
generuje tvár len v 128 px a výsledok je rozmazaný. Použi ho, kým nemáš stiahnuté modely pre 01,
alebo ako „identity lock“ na výsledok z 01.

## Mac (Apple Silicon) – na čo myslieť
- Na MPS nie je natívny výpočet vo fp8. ComfyUI fp8 váhy buď prevedie na fp16/bf16 (ak je dosť pamäte),
  alebo ich nechá vo fp8 a prevádza po vrstvách. Na Macu teda fp8 neušetrí toľko ako na NVIDIA. Preto je odporúčaný bf16 súbor.
- Klein 4B + Qwen3 4B potrebujú spolu ~16 GB váh. Pri 16 GB RAM to pôjde pomaly so swapom,
  od 24 GB rozumne, od 32 GB pohodlne.
- Väčšia verzia Klein 9B (`bfs_head_v1_flux-klein_9b_*`) dáva lepšie výsledky, ale potrebuje ~2× viac pamäte
  a jej licencia je nekomerčná.

## Licencie a pravidlá
- FLUX.2 Klein 4B: Apache-2.0 (komerčne OK). BFS LoRA: MIT. `inswapper_128`: len nekomerčné.
- Tvár z fotky 2 musí byť tvoja vlastná AI tvár, tvoja tvár alebo tvár človeka, ktorý dal súhlas.
  Nepoužívaj tváre známych osôb ani reálnych ľudí bez súhlasu. Nerob intímny obsah reálnych ľudí.
  Platformy (Instagram, TikTok…) vyžadujú označenie AI obsahu.

## Zdroje
- BFS Best Face Swap (model card, docs, workflowy): https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap
- Oficiálne ComfyUI šablóny Flux.2 Klein: https://github.com/Comfy-Org/workflow_templates
- FLUX.2 Klein 4B (Apache-2.0): https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Prehľad metód 2026 (Civitai): https://civitai.com/articles/31600/the-definitive-guide-to-face-swapping-in-2026
