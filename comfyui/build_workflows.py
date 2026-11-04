"""Generate the ComfyUI workflow JSON files in this folder.

Run:  python3 comfyui/build_workflows.py

Writes UI-format workflows (drag & drop into ComfyUI) into comfyui/workflows/
and API-format twins (for the /prompt endpoint / scripting) into
comfyui/workflows/api/.
Only core ComfyUI nodes are used in workflow 01; workflow 02 additionally
needs ComfyUI-ReActor.
"""

import json
import os
import uuid

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")

# Widget names per node type, in the order ComfyUI serialises widgets_values.
# "*control*" marks the extra control_after_generate value the frontend adds
# after seed widgets.
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type", "device"],
    "VAELoader": ["vae_name"],
    "LoadImage": ["image", "*upload*"],
    "ImageScaleToTotalPixels": ["upscale_method", "megapixels", "resolution_steps"],
    "GetImageSize": [],
    "VAEEncode": [],
    "VAEDecode": [],
    "CLIPTextEncode": ["text"],
    "ConditioningZeroOut": [],
    "ReferenceLatent": [],
    "EmptyFlux2LatentImage": ["width", "height", "batch_size"],
    "Flux2Scheduler": ["steps", "width", "height"],
    "KSamplerSelect": ["sampler_name"],
    "RandomNoise": ["noise_seed", "*control*"],
    "CFGGuider": ["cfg"],
    "SamplerCustomAdvanced": [],
    "SaveImage": ["filename_prefix"],
    "PreviewImage": [],
    "ImageStitch": ["direction", "match_image_size", "spacing_width", "spacing_color"],
    "FluxGuidance": ["guidance"],
    "KSampler": ["seed", "*control*", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "ReActorFaceSwap": [
        "enabled", "swap_model", "facedetection", "face_restore_model",
        "face_restore_visibility", "codeformer_weight", "detect_gender_input",
        "detect_gender_source", "input_faces_index", "source_faces_index",
        "console_log_level",
    ],
    "MarkdownNote": ["text"],
}

# Socket inputs (non-widget) per node type, in slot order, with their types.
SOCKETS = {
    "LoraLoaderModelOnly": [("model", "MODEL")],
    "ImageScaleToTotalPixels": [("image", "IMAGE")],
    "GetImageSize": [("image", "IMAGE")],
    "VAEEncode": [("pixels", "IMAGE"), ("vae", "VAE")],
    "VAEDecode": [("samples", "LATENT"), ("vae", "VAE")],
    "CLIPTextEncode": [("clip", "CLIP")],
    "ConditioningZeroOut": [("conditioning", "CONDITIONING")],
    "ReferenceLatent": [("conditioning", "CONDITIONING"), ("latent", "LATENT")],
    "CFGGuider": [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
    "SamplerCustomAdvanced": [("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                              ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
    "SaveImage": [("images", "IMAGE")],
    "PreviewImage": [("images", "IMAGE")],
    "ImageStitch": [("image1", "IMAGE"), ("image2", "IMAGE")],
    "FluxGuidance": [("conditioning", "CONDITIONING")],
    "KSampler": [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                 ("latent_image", "LATENT")],
    "ReActorFaceSwap": [("input_image", "IMAGE"), ("source_image", "IMAGE"),
                        ("face_model", "FACE_MODEL"), ("face_boost", "FACE_BOOST")],
}

OUTPUTS = {
    "UNETLoader": [("MODEL", "MODEL")],
    "LoraLoaderModelOnly": [("MODEL", "MODEL")],
    "CLIPLoader": [("CLIP", "CLIP")],
    "DualCLIPLoader": [("CLIP", "CLIP")],
    "VAELoader": [("VAE", "VAE")],
    "LoadImage": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    "ImageScaleToTotalPixels": [("IMAGE", "IMAGE")],
    "GetImageSize": [("width", "INT"), ("height", "INT"), ("batch_size", "INT")],
    "VAEEncode": [("LATENT", "LATENT")],
    "VAEDecode": [("IMAGE", "IMAGE")],
    "CLIPTextEncode": [("CONDITIONING", "CONDITIONING")],
    "ConditioningZeroOut": [("CONDITIONING", "CONDITIONING")],
    "ReferenceLatent": [("CONDITIONING", "CONDITIONING")],
    "EmptyFlux2LatentImage": [("LATENT", "LATENT")],
    "Flux2Scheduler": [("SIGMAS", "SIGMAS")],
    "KSamplerSelect": [("SAMPLER", "SAMPLER")],
    "RandomNoise": [("NOISE", "NOISE")],
    "CFGGuider": [("GUIDER", "GUIDER")],
    "SamplerCustomAdvanced": [("output", "LATENT"), ("denoised_output", "LATENT")],
    "SaveImage": [],
    "PreviewImage": [],
    "ImageStitch": [("IMAGE", "IMAGE")],
    "FluxGuidance": [("CONDITIONING", "CONDITIONING")],
    "KSampler": [("LATENT", "LATENT")],
    "ReActorFaceSwap": [("SWAPPED_IMAGE", "IMAGE"), ("FACE_MODEL", "FACE_MODEL"),
                        ("ORIGINAL_IMAGE", "IMAGE")],
    "MarkdownNote": [],
}

VIRTUAL = {"MarkdownNote"}  # frontend-only nodes, not part of the API prompt


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.groups = []
        self._nid = 0
        self._lid = 0

    def add(self, ntype, widgets=None, pos=(0, 0), size=(320, 120), title=None, color=None,
            bgcolor=None, **links):
        """Add a node. widgets: dict name->value. links: input_name=(node, out_slot)."""
        self._nid += 1
        widgets = dict(widgets or {})
        names = WIDGETS[ntype]
        node = {
            "id": self._nid,
            "type": ntype,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": self._nid - 1,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {"Node name for S&R": ntype} if ntype not in VIRTUAL else {},
            "widgets_values": [],
        }
        if title:
            node["title"] = title
        if color:
            node["color"] = color
        if bgcolor:
            node["bgcolor"] = bgcolor
        # widgets_values
        for w in names:
            if w == "*control*":
                node["widgets_values"].append(widgets.pop("*control*", "randomize"))
            elif w == "*upload*":
                node["widgets_values"].append("image")
            else:
                if w not in widgets:
                    raise KeyError(f"{ntype}: missing widget value {w}")
                node["widgets_values"].append(widgets.pop(w))
        if widgets:
            raise KeyError(f"{ntype}: unknown widgets {list(widgets)}")
        # socket inputs
        for name, typ in SOCKETS.get(ntype, []):
            node["inputs"].append({"name": name, "type": typ, "link": None})
        # outputs
        for i, (name, typ) in enumerate(OUTPUTS[ntype]):
            node["outputs"].append({"name": name, "type": typ, "links": [], "slot_index": i})
        self.nodes.append(node)
        for inp, (src, slot) in links.items():
            self.connect(src, slot, node, inp)
        return node

    def connect(self, src, slot, dst, inp):
        typ = src["outputs"][slot]["type"]
        idx = next((i for i, x in enumerate(dst["inputs"]) if x["name"] == inp), None)
        if idx is None:
            # widget converted to input (e.g. width/height fed by GetImageSize)
            if inp not in WIDGETS[dst["type"]]:
                raise KeyError(f"{dst['type']} has no input {inp}")
            dst["inputs"].append({"name": inp, "type": typ, "widget": {"name": inp}, "link": None})
            idx = len(dst["inputs"]) - 1
        self._lid += 1
        dst["inputs"][idx]["link"] = self._lid
        src["outputs"][slot]["links"].append(self._lid)
        self.links.append([self._lid, src["id"], slot, dst["id"], idx, typ])

    def group(self, title, members, color, pad=40):
        xs = [n["pos"][0] for n in members]
        ys = [n["pos"][1] for n in members]
        xe = [n["pos"][0] + n["size"][0] for n in members]
        ye = [n["pos"][1] + n["size"][1] for n in members]
        self.groups.append({
            "id": len(self.groups) + 1,
            "title": title,
            "bounding": [min(xs) - pad, min(ys) - pad - 40, max(xe) - min(xs) + 2 * pad,
                         max(ye) - min(ys) + 2 * pad + 40],
            "color": color,
            "font_size": 24,
            "flags": {},
        })

    def ui_json(self):
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(self.links))),
            "revision": 0,
            "last_node_id": self._nid,
            "last_link_id": self._lid,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": 0.6, "offset": [80, 120]}},
            "version": 0.4,
        }

    def api_json(self):
        link_by_id = {l[0]: l for l in self.links}
        prompt = {}
        for n in self.nodes:
            if n["type"] in VIRTUAL or n["mode"] != 0:
                continue
            inputs = {}
            vals = list(n["widgets_values"])
            for w in WIDGETS[n["type"]]:
                v = vals.pop(0)
                if w.startswith("*"):
                    continue
                inputs[w] = v
            for inp in n["inputs"]:
                if inp["link"] is not None:
                    l = link_by_id[inp["link"]]
                    inputs[inp["name"]] = [str(l[1]), l[2]]
            entry = {"class_type": n["type"], "inputs": inputs}
            if n.get("title"):
                entry["_meta"] = {"title": n["title"]}
            prompt[str(n["id"])] = entry
        return prompt


BLUE, GREEN, PURPLE, ORANGE, GREY = "#3f789e", "#3f9e5a", "#8a5ab4", "#b06634", "#444444"

BFS_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and "
    "background. remove the head from Picture 1 completely and replace it with the head from "
    "Picture 2, strictly preserving the hair, eye color, and nose structure of Picture 2. copy the "
    "direction of the eye, head rotation, and micro-expressions from Picture 1. Keep the body, "
    "clothing, pose, framing, and background intact.\n\n"
    "The final image is a real, unretouched photograph of a real person, shot on a full-frame camera "
    "with a 50mm lens. Natural skin texture with visible pores, fine facial hair, slight skin "
    "unevenness and natural shine, individual strands of hair, realistic eyes with natural "
    "catchlights. The skin tone and lighting on the face match the neck and body exactly. Sharp "
    "focus, true-to-life colors, subtle film grain."
)

NOTE_01 = """# Fúzia: TELO z fotky 1 + TVÁR z fotky 2
**FLUX.2 Klein 4B + BFS Head Swap LoRA** (len core uzly, žiadne custom nodes)

## Ako použiť
1. **FOTKA 1 – TELO**: postava, póza, oblečenie, pozadie, svetlo → toto zostane.
2. **FOTKA 2 – TVÁR**: ostrá fotka hlavy (tvár + vlasy), ideálne orezaná na hlavu a plecia.
3. *Queue* (Ctrl/Cmd+Enter). Seed je *randomize* → každé spustenie = nová varianta.
4. Výsledok vpravo; porovnanie *telo | výsledok* v náhľade.

**Poradie fotiek NEMENIŤ** – LoRA je trénovaná *body-first* (Picture 1 = telo, Picture 2 = hlava).

## Modely (stiahnuť raz)
| Súbor | Priečinok |
|---|---|
| [flux-2-klein-4b.safetensors](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors) (7.8 GB, bf16 – odporúčané na Mac) | `models/diffusion_models/` |
| ⚠️ `flux-2-klein-4b-fp8` na Macu **nepoužívaj** – s LoRA padá (MPS nevie fp8) | – |
| [qwen_3_4b.safetensors](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) | `models/text_encoders/` |
| [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/vae/flux2-vae.safetensors) | `models/vae/` |
| [bfs_head_v1_flux-klein_4b.safetensors](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_4b.safetensors) | `models/loras/` |

Tvoj `flux1-dev-fp8` + `clip_l` + `t5xxl` + `ae` sa tu **nepoužívajú** (Flux.1 ≠ Flux.2).

## Ladenie
- **Tvár nie je dosť podobná** → LoRA strength 1.1–1.3; lepšia/ostrejšia fotka 2; orež fotku 2 na hlavu.
- **Mení sa telo/pozadie** → LoRA strength 0.8–0.9.
- **Iný výraz tváre** → na koniec promptu doplň napr. `smiling softly, mouth closed`.
- **Kvalita** → megapixels 1.0 (default). Vyššie = pomalšie a väčšia spotreba RAM.
- **Viac variant naraz** → `batch_size` v *Empty Flux 2 Latent* (na Macu radšej 1).

## Pravidlá
Používaj len tváre, na ktoré máš súhlas (vlastná AI postava, ty, osoba so súhlasom).
Žiadne verejne známe osoby, žiadne intímne zábery reálnych ľudí. Klein 4B je Apache-2.0 (komerčne OK),
BFS LoRA je MIT.
"""

NOTE_16GB = """# ⚠️ Verzia pre Mac so 16 GB RAM
- Model: **flux-2-klein-4b.safetensors** (bf16, 7.8 GB), fotky zmenšené na **0.75 MP**.
- fp8 verzia na Macu padá (*Float8_e4m3fn … MPS*), nepoužívaj ju.
- Pred spustením zavri Chrome/Safari s veľa kartami a ostatné ťažké aplikácie.
- Pri chybe *MPS backend out of memory* daj v oboch „→ 0.75 MP“ uzloch 0.6.

"""

NEG_PROMPT = (
    "cartoon, anime, illustration, drawing, painting, 3d render, cgi, digital art, plastic skin, "
    "airbrushed skin, overly smooth skin, wax figure, doll face, oversaturated colors, harsh contrast, "
    "blurry, low quality, jpeg artifacts, deformed face, asymmetric eyes, visible seam on the neck"
)

NOTE_BASE = """# Realistická verzia: FLUX.2 Klein 4B **BASE** (nedestilovaný)
Rovnaká fúzia ako workflow 01, ale s pomalším modelom, ktorý dáva prirodzenejšiu pokožku
a **berie negatívny prompt** (CFG > 1).

## Model navyše (stiahnuť raz, 7.8 GB)
[flux-2-klein-base-4b.safetensors](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/diffusion_models/flux-2-klein-base-4b.safetensors) → `models/diffusion_models/`
(ostatné súbory sú rovnaké ako vo workflow 01). Nepoužívaj `-fp8` verziu – na Macu s LoRA padá.

## Nastavenia (oficiálna šablóna ComfyUI pre Klein 4B base: euler, 20 krokov, CFG 5)
- **Kroky 20** (16 = rýchlejšie, 30 = o niečo detailnejšie).
- **CFG 4** – nižšie (3–3.5) = prirodzenejšie farby, vyššie (5) = presnejšie drží prompt, ale „AI“ kontrast.
- **Negatívny prompt** tu funguje – dopíš doň, čo nechceš vidieť.
- Fotky 0.75 MP (16 GB Mac). Pri *out of memory* daj 0.6.

## Rýchlosť
Base robí pri CFG > 1 dva prechody modelom na krok → 20 krokov ≈ **10× dlhšie** než 4-krokový workflow 01.
Rýchlo si vo workflow 01 over, ktorá dvojica fotiek funguje (podobnosť, póza), a až tú pusti tu.
Seed sa medzi 01 a 03 neprenáša – iný model = iný obrázok z rovnakého seedu.

---

"""

NOTE_02 = """# Záložný workflow: ReActor + Flux.1 dev doladenie
Funguje **len s modelmi, ktoré už máš** (inswapper_128 + flux1-dev-fp8).
Kvalitou je pod workflow 01 (inswapper má rozlíšenie tváre len 128 px), ale je rýchly.

## Ako použiť
1. **FOTKA 1 – TELO** a **FOTKA 2 – TVÁR** ako v 01.
2. ReActor vymení tvár → Flux.1 dev ju pri nízkom denoise (0.20) „pretextúruje“, aby nebola rozmazaná.
3. Denoise 0.15–0.30: vyššie = ostrejšia koža, ale tvár sa začne vzďaľovať od fotky 2.

## Tipy
- `face_restore_model`: ak máš `codeformer-v0.1.0.pth` alebo `GFPGANv1.4.pth` v `models/facerestore_models/`, vyber ho (visibility 0.6–0.8).
- Ak tvoj `flux1-dev-fp8.safetensors` leží v `models/checkpoints/` (all-in-one ~17 GB), tento workflow ho nevidí –
  presuň/nalinkuj ho do `models/diffusion_models/` len ak je to *UNET-only* verzia (~12 GB). All-in-one verziu načítaj cez *Load Checkpoint*.
- Ako „identity lock“: na vstup TELO daj výsledok z workflow 01, ak chceš tvár ešte bližšie k fotke 2.

**Licencia:** `inswapper_128` je od InsightFace len na **nekomerčné** použitie. Na monetizovaný AI influencer účet použi workflow 01.
"""


def build_01(unet_name="flux-2-klein-4b.safetensors", mp=1.0, note=None, base=False):
    g = Graph()
    # --- models
    unet = g.add("UNETLoader", {"unet_name": unet_name, "weight_dtype": "default"},
                 pos=(0, 0), size=(360, 90),
                 title="Model: FLUX.2 Klein 4B BASE (nedestilovaný)" if base else "Model: FLUX.2 Klein 4B")
    lora = g.add("LoraLoaderModelOnly", {"lora_name": "bfs_head_v1_flux-klein_4b.safetensors",
                                         "strength_model": 1.0},
                 pos=(0, 140), size=(360, 90), title="BFS Head Swap LoRA (sila 0.8–1.3)", model=(unet, 0))
    clip = g.add("CLIPLoader", {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"},
                 pos=(0, 280), size=(360, 110), title="Text encoder: Qwen3 4B")
    vae = g.add("VAELoader", {"vae_name": "flux2-vae.safetensors"}, pos=(0, 440), size=(360, 60),
                title="VAE: Flux 2")
    g.group("1 · Modely", [unet, lora, clip, vae], BLUE)

    # --- photos
    body = g.add("LoadImage", {"image": "telo.png"}, pos=(460, 0), size=(340, 420),
                 title="FOTKA 1 – TELO (póza, oblečenie, pozadie)", color="#232", bgcolor="#353")
    face = g.add("LoadImage", {"image": "tvar.png"}, pos=(460, 480), size=(340, 420),
                 title="FOTKA 2 – TVÁR / HLAVA (identita)", color="#223", bgcolor="#335")
    body_s = g.add("ImageScaleToTotalPixels", {"upscale_method": "lanczos", "megapixels": mp,
                                               "resolution_steps": 16},
                   pos=(860, 0), size=(300, 110), title=f"Telo → {mp:g} MP", image=(body, 0))
    face_s = g.add("ImageScaleToTotalPixels", {"upscale_method": "lanczos", "megapixels": mp,
                                               "resolution_steps": 16},
                   pos=(860, 480), size=(300, 110), title=f"Tvár → {mp:g} MP", image=(face, 0))
    size = g.add("GetImageSize", {}, pos=(860, 170), size=(300, 90), title="Rozmer výstupu = rozmer tela",
                 image=(body_s, 0))
    body_l = g.add("VAEEncode", {}, pos=(860, 320), size=(300, 50), title="Encode telo",
                   pixels=(body_s, 0), vae=(vae, 0))
    face_l = g.add("VAEEncode", {}, pos=(860, 640), size=(300, 50), title="Encode tvár",
                   pixels=(face_s, 0), vae=(vae, 0))
    g.group("2 · Vstupné fotky", [body, face, body_s, face_s, size, body_l, face_l], GREEN)

    # --- prompt + references
    pos_c = g.add("CLIPTextEncode", {"text": BFS_PROMPT}, pos=(1260, 0), size=(420, 260),
                  title="Prompt (BFS head_swap – nemeň začiatok)", clip=(clip, 0))
    if base:
        neg_c = g.add("CLIPTextEncode", {"text": NEG_PROMPT}, pos=(1260, 320), size=(420, 180),
                      title="Negatívny prompt (čo NECHCEŠ vidieť)", color="#322", bgcolor="#533", clip=(clip, 0))
    else:
        neg_c = g.add("ConditioningZeroOut", {}, pos=(1260, 320), size=(300, 30),
                      title="Negatív (prázdny, CFG 1)", conditioning=(pos_c, 0))
    pos_r1 = g.add("ReferenceLatent", {}, pos=(1740, 0), size=(260, 50), title="+ Picture 1 (telo)",
                   conditioning=(pos_c, 0), latent=(body_l, 0))
    pos_r2 = g.add("ReferenceLatent", {}, pos=(1740, 100), size=(260, 50), title="+ Picture 2 (tvár)",
                   conditioning=(pos_r1, 0), latent=(face_l, 0))
    neg_r1 = g.add("ReferenceLatent", {}, pos=(1740, 320), size=(260, 50), title="neg + Picture 1",
                   conditioning=(neg_c, 0), latent=(body_l, 0))
    neg_r2 = g.add("ReferenceLatent", {}, pos=(1740, 420), size=(260, 50), title="neg + Picture 2",
                   conditioning=(neg_r1, 0), latent=(face_l, 0))
    g.group("3 · Prompt + referencie (poradie: telo, potom tvár)",
            [pos_c, neg_c, pos_r1, pos_r2, neg_r1, neg_r2], PURPLE)

    # --- sampler
    latent = g.add("EmptyFlux2LatentImage", {"width": 1024, "height": 1024, "batch_size": 1},
                   pos=(2100, 0), size=(300, 110), title="Empty Flux 2 Latent (batch_size = počet variant)",
                   width=(size, 0), height=(size, 1))
    sched = g.add("Flux2Scheduler", {"steps": 20 if base else 4, "width": 1024, "height": 1024},
                  pos=(2100, 160), size=(300, 110),
                  title="Kroky (base: 16–30)" if base else "Kroky (distilled Klein: 4)",
                  width=(size, 0), height=(size, 1))
    sampler = g.add("KSamplerSelect", {"sampler_name": "euler"}, pos=(2100, 320), size=(300, 60))
    noise = g.add("RandomNoise", {"noise_seed": 20261103, "*control*": "randomize"}, pos=(2100, 430),
                  size=(300, 90), title="Seed")
    guider = g.add("CFGGuider", {"cfg": 4.0 if base else 1.0}, pos=(2100, 570), size=(300, 100),
                   title="CFG 4 (base: 3–5)" if base else "CFG 1.0 (distilled)",
                   model=(lora, 0), positive=(pos_r2, 0), negative=(neg_r2, 0))
    samp = g.add("SamplerCustomAdvanced", {}, pos=(2460, 0), size=(280, 130), noise=(noise, 0),
                 guider=(guider, 0), sampler=(sampler, 0), sigmas=(sched, 0), latent_image=(latent, 0))
    dec = g.add("VAEDecode", {}, pos=(2460, 190), size=(280, 50), samples=(samp, 0), vae=(vae, 0))
    g.group("4 · Generovanie", [latent, sched, sampler, noise, guider, samp, dec], ORANGE)

    # --- output
    save = g.add("SaveImage", {"filename_prefix": "modelka/headswap_base" if base else "modelka/headswap"},
                 pos=(2840, 0), size=(460, 620),
                 title="VÝSLEDOK", images=(dec, 0))
    stitch = g.add("ImageStitch", {"direction": "right", "match_image_size": True, "spacing_width": 8,
                                   "spacing_color": "white"},
                   pos=(2840, 680), size=(460, 130), title="Porovnanie telo | výsledok",
                   image1=(body_s, 0), image2=(dec, 0))
    prev = g.add("PreviewImage", {}, pos=(2840, 870), size=(460, 360), title="Náhľad porovnania",
                 images=(stitch, 0))
    g.group("5 · Výstup", [save, stitch, prev], GREY)

    g.add("MarkdownNote", {"text": note or NOTE_01}, pos=(0, 1020), size=(1160, 900), title="NÁVOD (čítaj prvé)",
          color="#432", bgcolor="#653")
    return g


def build_02():
    g = Graph()
    body = g.add("LoadImage", {"image": "telo.png"}, pos=(0, 0), size=(340, 420),
                 title="FOTKA 1 – TELO", color="#232", bgcolor="#353")
    face = g.add("LoadImage", {"image": "tvar.png"}, pos=(0, 480), size=(340, 420),
                 title="FOTKA 2 – TVÁR", color="#223", bgcolor="#335")
    react = g.add("ReActorFaceSwap", {
        "enabled": True, "swap_model": "inswapper_128.onnx", "facedetection": "retinaface_resnet50",
        "face_restore_model": "none", "face_restore_visibility": 0.7, "codeformer_weight": 0.5,
        "detect_gender_input": "no", "detect_gender_source": "no", "input_faces_index": "0",
        "source_faces_index": "0", "console_log_level": 1,
    }, pos=(420, 0), size=(360, 380), title="ReActor face swap", input_image=(body, 0), source_image=(face, 0))
    g.group("1 · Výmena tváre (ReActor)", [body, face, react], GREEN)

    unet = g.add("UNETLoader", {"unet_name": "flux1-dev-fp8.safetensors", "weight_dtype": "default"},
                 pos=(860, 0), size=(360, 90), title="Flux.1 dev (tvoj model)")
    clip = g.add("DualCLIPLoader", {"clip_name1": "clip_l.safetensors",
                                    "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux",
                                    "device": "default"},
                 pos=(860, 140), size=(360, 130))
    vae = g.add("VAELoader", {"vae_name": "ae.safetensors"}, pos=(860, 320), size=(360, 60))
    g.group("2 · Flux.1 modely", [unet, clip, vae], BLUE)

    text = g.add("CLIPTextEncode", {"text": "photo of a woman, natural skin texture with pores, sharp focus, "
                                            "realistic lighting, high detail"},
                 pos=(1300, 0), size=(380, 160), title="Prompt doladenia", clip=(clip, 0))
    guid = g.add("FluxGuidance", {"guidance": 2.5}, pos=(1300, 210), size=(300, 60), conditioning=(text, 0))
    neg = g.add("ConditioningZeroOut", {}, pos=(1300, 320), size=(300, 30), conditioning=(text, 0))
    enc = g.add("VAEEncode", {}, pos=(1300, 400), size=(300, 50), pixels=(react, 0), vae=(vae, 0))
    ks = g.add("KSampler", {"seed": 20261103, "*control*": "randomize", "steps": 20, "cfg": 1.0,
                            "sampler_name": "euler", "scheduler": "beta", "denoise": 0.2},
               pos=(1740, 0), size=(320, 270), title="Doladenie (denoise 0.15–0.30)",
               model=(unet, 0), positive=(guid, 0), negative=(neg, 0), latent_image=(enc, 0))
    dec = g.add("VAEDecode", {}, pos=(1740, 330), size=(320, 50), samples=(ks, 0), vae=(vae, 0))
    g.group("3 · Doladenie kože (Flux.1 dev img2img)", [text, guid, neg, enc, ks, dec], ORANGE)

    save = g.add("SaveImage", {"filename_prefix": "modelka/reactor"}, pos=(2140, 0), size=(460, 620),
                 title="VÝSLEDOK", images=(dec, 0))
    raw = g.add("PreviewImage", {}, pos=(2140, 680), size=(460, 360), title="Surový swap (bez doladenia)",
                images=(react, 0))
    g.group("4 · Výstup", [save, raw], GREY)

    g.add("MarkdownNote", {"text": NOTE_02}, pos=(0, 1020), size=(1220, 620), title="NÁVOD",
          color="#432", bgcolor="#653")
    return g


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, builder in [
        ("01_modelka_telo+tvar_flux2klein_BFS_16GB",
         lambda: build_01("flux-2-klein-4b.safetensors", 0.75, NOTE_16GB + NOTE_01)),
        ("01_modelka_telo+tvar_flux2klein_BFS", build_01),
        ("03_modelka_telo+tvar_flux2klein_BASE_realisticka_16GB",
         lambda: build_01("flux-2-klein-base-4b.safetensors", 0.75, NOTE_BASE + NOTE_01, base=True)),
        ("02_modelka_telo+tvar_reactor_zaloha", build_02),
    ]:
        g = builder()
        with open(os.path.join(OUT_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(g.ui_json(), f, ensure_ascii=False, indent=1)
        os.makedirs(os.path.join(OUT_DIR, "api"), exist_ok=True)
        with open(os.path.join(OUT_DIR, "api", name + ".api.json"), "w", encoding="utf-8") as f:
            json.dump(g.api_json(), f, ensure_ascii=False, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    main()
