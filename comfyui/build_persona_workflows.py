"""Generate the cloud (Seedream 5.0 + Kling 4.0) persona workflows 04-07.

Run:  python3 comfyui/build_persona_workflows.py

The partner-API nodes (ByteDanceSeedreamNodeV3, Kling4*, SaveVideo) use dynamic
inputs, so their JSON is copied 1:1 from the structure of the official ComfyUI
workflow templates (Comfy-Org/workflow_templates) instead of being derived.
These workflows need a Comfy account with credits (Settings -> User -> Login).
"""

import json
import os

from build_workflows import BLUE, GREEN, GREY, ORANGE, PURPLE, OUT_DIR, Graph


def add_raw(g, ntype, inputs, outputs, wv, wvn, pos, size, title=None, mode=0, color=None, bgcolor=None):
    g._nid += 1
    node = {
        "id": g._nid,
        "type": ntype,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": g._nid - 1,
        "mode": mode,
        "inputs": [dict(i, link=None) for i in inputs],
        "outputs": [{"name": n, "type": t, "links": [], "slot_index": k} for k, (n, t) in enumerate(outputs)],
        "properties": {"Node name for S&R": ntype} if ntype != "SaveVideo" else {},
        "widgets_values": wv,
        "widgets_values_named": wvn,
    }
    if title:
        node["title"] = title
    if color:
        node["color"] = color
    if bgcolor:
        node["bgcolor"] = bgcolor
    g.nodes.append(node)
    return node


def seedream(g, prompt, n_refs, preset, w, h, max_images, pos, title, mode=0, seed=42):
    """ByteDanceSeedreamNodeV3 with model 'seedream 5.0 lite' (same layout as the official
    api_bytedance_seedream_5_0_lite_image_edit template). n_refs connected slots + 1 empty."""
    inputs = [{"label": f"image_{i}", "name": f"model.images.image_{i}", "shape": 7, "type": "IMAGE"}
              for i in range(1, n_refs + 2)]
    wv = [prompt, "seedream 5.0 lite", preset, w, h, max_images, False, seed, "randomize", False, True]
    wvn = {"prompt": prompt, "model": "seedream 5.0 lite", "model.size_preset": preset, "model.width": w,
           "model.height": h, "model.max_images": max_images, "model.fail_on_partial": False,
           "model.seed": seed, "control_after_generate": "randomize", "model.watermark": False,
           "model.thinking": True}
    return add_raw(g, "ByteDanceSeedreamNodeV3", inputs, [("IMAGE", "IMAGE")], wv, wvn, pos, (440, 620),
                   title=title, mode=mode, color="#432", bgcolor="#653")


def kling_i2v(g, prompt, pos, title, resolution="720p", duration=5, mode=0):
    inputs = [{"name": "model.first_frame", "shape": 7, "type": "IMAGE"},
              {"name": "model.middle_frames.middle_frame_1", "shape": 7, "type": "IMAGE"},
              {"name": "model.last_frame", "shape": 7, "type": "IMAGE"}]
    wv = ["kling-4.0", prompt, resolution, duration, 42, "randomize", "fixed"]
    wvn = {"model": "kling-4.0", "model.prompt": prompt, "model.resolution": resolution,
           "model.duration": duration, "model.seed": 42, "control_after_generate#1": "randomize",
           "control_after_generate": "fixed"}
    return add_raw(g, "Kling4ImageToVideoNode", inputs, [("VIDEO", "VIDEO")], wv, wvn, pos, (440, 560),
                   title=title, mode=mode, color="#432", bgcolor="#653")


def kling_r2v(g, prompt, pos, title, resolution="720p", aspect="9:16", duration=5, mode=0):
    inputs = [{"name": f"model.reference_images.image_{i}", "shape": 7, "type": "IMAGE"} for i in (1, 2, 3)]
    inputs += [{"name": f"model.reference_videos.video_{i}", "shape": 7, "type": "VIDEO"} for i in (1, 2)]
    inputs += [{"name": "model.first_frame", "shape": 7, "type": "IMAGE"},
               {"name": "model.last_frame", "shape": 7, "type": "IMAGE"}]
    wv = ["kling-4.0", prompt, resolution, aspect, duration, 42, "randomize", "randomize"]
    wvn = {"model": "kling-4.0", "model.prompt": prompt, "model.resolution": resolution,
           "model.aspect_ratio": aspect, "model.duration": duration, "model.seed": 42,
           "control_after_generate#1": "randomize", "control_after_generate": "randomize"}
    return add_raw(g, "Kling4ReferenceToVideoNode", inputs, [("VIDEO", "VIDEO")], wv, wvn, pos, (440, 640),
                   title=title, mode=mode, color="#432", bgcolor="#653")


def save_video(g, prefix, pos, mode=0):
    wv = [prefix, "mp4", "auto", "auto"]
    wvn = {"filename_prefix": prefix, "format": "mp4", "format.codec": "auto", "codec": "auto"}
    return add_raw(g, "SaveVideo", [{"name": "video", "type": "VIDEO"}], [("video", "VIDEO")], wv, wvn,
                   pos, (460, 560), title="VIDEO VÝSLEDOK", mode=mode)


def note(g, text, pos, size):
    return g.add("MarkdownNote", {"text": text}, pos=pos, size=size, title="NÁVOD (čítaj prvé)",
                 color="#432", bgcolor="#653")


def load(g, fname, title, pos, green=True):
    return g.add("LoadImage", {"image": fname}, pos=pos, size=(340, 420), title=title,
                 color="#232" if green else "#223", bgcolor="#353" if green else "#335")


# ---------------------------------------------------------------- prompts

FUSION = (
    "Image 1 is the base photo, image 2 is the face reference.\n"
    "Keep everything from image 1 exactly as it is: body, body proportions, pose, clothing, background, "
    "framing, camera angle and lighting.\n"
    "Replace only the head and face with the person from image 2: identical face shape, eyes and eye "
    "color, eyebrows, nose, lips, skin tone and hairstyle. Match the head size and angle to the body and "
    "blend the neck and skin tone seamlessly, so it looks like one real photograph.\n"
    "The result must look like a real, unretouched photo: natural skin texture with visible pores, "
    "realistic hair strands, natural light, no beauty filter."
)

REFSET = (
    "Create 4 separate photos of exactly the same woman as in image 1: identical face, eye color, "
    "nose, lips, skin tone, hair color, hairstyle and body type. These are reference photos of one person:\n"
    "1. Close-up portrait, facing the camera, neutral relaxed expression.\n"
    "2. Three-quarter view portrait, head turned slightly to her left, soft smile.\n"
    "3. Side profile facing right.\n"
    "4. Full-body photo, standing in a relaxed pose, wearing a simple fitted black top and blue jeans.\n"
    "All 4 photos: plain light grey studio background, soft even daylight, real unretouched photographs "
    "with natural skin texture and visible pores, sharp focus."
)

NEWPHOTO = (
    "PERSONA: The woman in image 1, image 2 and image 3 is one and the same person. Keep her face, facial "
    "proportions, eye color, eyebrows, nose, lips, skin tone, hair color and hairstyle exactly identical "
    "to the reference images.\n\n"
    "SCENE: She is sitting at a small table in a sunny café, holding a cup of coffee, wearing a cream knit "
    "sweater and small gold hoop earrings, laughing naturally while looking slightly away from the camera.\n\n"
    "STYLE: candid smartphone photo like a real Instagram post, natural daylight, real unretouched skin "
    "with visible pores, no beauty filter, slight phone-camera grain."
)

VIDEO_I2V = (
    "@Image1 is the first frame. She slowly turns her head toward the camera, smiles softly and tucks a "
    "strand of hair behind her ear, then takes a sip of coffee. Gentle handheld smartphone camera, natural "
    "daylight, realistic motion. Keep her face, hair, outfit and the background exactly as in @Image1, "
    "no morphing, no change of her face."
)

VIDEO_R2V = (
    "The woman from @Image1 and @Image2 is one person: keep her face, hair and body exactly as in the "
    "references. She walks along a sunny beach promenade at golden hour in a white linen dress, laughs and "
    "looks back over her shoulder at the camera. Smartphone vlog style, natural handheld movement, "
    "realistic lighting and skin."
)

# ---------------------------------------------------------------- notes

COMMON = """
## Cloud = platíš kreditmi
Tieto uzly bežia na serveroch ByteDance/Kling cez Comfy účet (vpravo hore **Sign in** + kredit).
Cenu ukazuje štítok na uzle. Každé **Run** = nová platba.

## Pravidlá
Tvár persony musí byť **vymyslená AI tvár** (alebo tvoja / človeka so súhlasom).
Žiadne známe osoby. Na sociálnych sieťach zapni označenie AI (Instagram *AI-generated profile*, TikTok *AIGC* label).
"""

NOTE_04 = """# Persona – KROK 1: Fúzia (Seedream 5.0 lite)
**Telo z fotky 1 + tvár z fotky 2 → prvá fotka tvojej AI modelky.**

1. **FOTKA 1 – TELO** a **FOTKA 2 – TVÁR** (poradie nemeň, prompt hovorí *image 1 = telo, image 2 = tvár*).
2. **Run.** Výsledok sa uloží do `output/persona/01_fuzia_…`.
3. Spúšťaj, kým nemáš fúziu, ktorá sa ti páči. **Toto bude tvár persony navždy.**
4. Pokračuj workflowom **05 – Referenčná sada**.

- Rozmer: `(2K) 1728x2304 (3:4)`. Iný pomer strán zmeníš v `size_preset`.
- Tvár málo podobná fotke 2 → orež fotku 2 na hlavu a vlasy, použi ostrejšiu fotku.
- ByteDance môže odmietnuť fotky, ktoré vyzerajú ako konkrétna reálna osoba.
""" + COMMON

NOTE_05 = """# Persona – KROK 2: Referenčná sada („pamäť“ persony)
Z jednej fotky persony vytvorí **4 fotky tej istej osoby** z rôznych uhlov (zblízka, 3/4, profil, celá postava).
Tieto fotky sú jej **pamäť**: v každom ďalšom generovaní ich dávaš ako referencie, a tak tvár zostane rovnaká.

1. **PERSONA – vybraná fúzia**: nahraj najlepší výsledok z workflow 04 (`output/persona/01_fuzia_…`).
2. **Run** → `output/persona/02_referencie_…` (1–4 obrázky, model rozhodne).
3. Vyber **3 najlepšie** (ideálne: tvár zblízka, polcelok/3/4, celá postava).
   Ulož si ich do priečinka `persona_MENO/` a už ich **nemeň**. To je tvoja „identity pack“.

Ak niektorá fotka tvár zmení, nepoužívaj ju. Radšej spusti znova.
""" + COMMON

NOTE_06 = """# Persona – KROK 3: Nové fotky (vždy tá istá tvár)
1. **REF 1–3**: nahraj 3 fotky z identity packu (z workflow 05). Pri každej fotke vždy tie isté.
2. V prompte meníš **len časť `SCENE:`** (kde je, čo robí, čo má oblečené, výraz).
   `PERSONA:` a `STYLE:` nechaj rovnaké. Zapíš si tam pevné detaily (napr. znamienko, tetovanie).
3. **Run** → `output/persona/03_foto_…`.

Tipy:
- Na Instagram feed `(2K) 1728x2304 (3:4)`. Ak z fotky budeš robiť video na Reels/TikTok, prepni na `(2K) 1600x2848 (9:16)`.
- `max_images` 2–4 = viac variant naraz (každá sa platí).
- Ak sa tvár začne „rozchádzať“, vymeň REF za lepšie fotky, nie prompt.
- Oblečenie/miesto môžeš dať aj ako ďalšiu referenciu (image 4) a napísať do SCENE: *wearing the outfit from image 4*.
""" + COMMON

NOTE_07 = """# Persona – KROK 4: Video (Kling 4.0)
**Seedance (video od ByteDance) realistické tváre ako referenciu odmieta**, preto video robí **Kling 4.0**.

## A) Fotka → video (zapnuté, najvernejšia tvár)
1. **PRVÝ SNÍMOK**: fotka persony z workflow 06 (pre Reels/TikTok vygeneruj fotku v 9:16).
2. V prompte opíš **pohyb** (čo robí, ako sa hýbe kamera). `@Image1` = prvý snímok.
3. **Run** → `output/video/persona_…mp4`. Kling pridá aj zvuk.

## B) Referencie → video (vypnuté, voľnejšia scéna)
Postava z identity packu v **novej scéne** bez prvého snímku. Tvár drží menej presne ako A.
Zapni: vyber oba uzly skupiny B a stlač **Ctrl+B** (bypass off). Potom A vypni, inak zaplatíš obe.

- Cena: 720p ≈ 0.22 $/s (5 s ≈ 1.12 $), 1080p ≈ 0.34 $/s. Dĺžka 3–30 s.
""" + COMMON


# ---------------------------------------------------------------- builders

def build_04():
    g = Graph()
    body = load(g, "telo.png", "FOTKA 1 – TELO (image 1)", (0, 0))
    face = load(g, "tvar.png", "FOTKA 2 – TVÁR (image 2)", (0, 480), green=False)
    g.group("1 · Vstupné fotky", [body, face], GREEN)
    sd = seedream(g, FUSION, 2, "(2K) 1728x2304 (3:4)", 1728, 2304, 1, (420, 0),
                  "Seedream 5.0 – FÚZIA (telo + tvár)")
    g.connect(body, 0, sd, "model.images.image_1")
    g.connect(face, 0, sd, "model.images.image_2")
    g.group("2 · Fúzia (Seedream 5.0 lite)", [sd], PURPLE)
    save = g.add("SaveImage", {"filename_prefix": "persona/01_fuzia"}, pos=(940, 0), size=(460, 680),
                 title="PERSONA – výsledok fúzie", images=(sd, 0))
    g.group("3 · Výstup", [save], GREY)
    note(g, NOTE_04, (0, 1000), (1400, 700))
    return g


def build_05():
    g = Graph()
    src = load(g, "persona_fuzia.png", "PERSONA – vybraná fúzia (z workflow 04)", (0, 0))
    g.group("1 · Persona", [src], GREEN)
    sd = seedream(g, REFSET, 1, "(2K) 1728x2304 (3:4)", 1728, 2304, 4, (420, 0),
                  "Seedream 5.0 – REFERENČNÁ SADA (4 uhly)")
    g.connect(src, 0, sd, "model.images.image_1")
    g.group("2 · Referenčná sada", [sd], PURPLE)
    save = g.add("SaveImage", {"filename_prefix": "persona/02_referencie"}, pos=(940, 0), size=(700, 680),
                 title="IDENTITY PACK – vyber 3 najlepšie", images=(sd, 0))
    g.group("3 · Výstup", [save], GREY)
    note(g, NOTE_05, (0, 1000), (1640, 640))
    return g


def build_06():
    g = Graph()
    r1 = load(g, "persona_ref1.png", "REF 1 – tvár zblízka", (0, 0))
    r2 = load(g, "persona_ref2.png", "REF 2 – 3/4 alebo polcelok", (0, 480))
    r3 = load(g, "persona_ref3.png", "REF 3 – celá postava", (0, 960))
    g.group("1 · Identity pack (vždy tie isté 3 fotky)", [r1, r2, r3], GREEN)
    sd = seedream(g, NEWPHOTO, 3, "(2K) 1728x2304 (3:4)", 1728, 2304, 1, (420, 0),
                  "Seedream 5.0 – NOVÁ FOTKA (meň len SCENE:)")
    for i, r in enumerate((r1, r2, r3), 1):
        g.connect(r, 0, sd, f"model.images.image_{i}")
    g.group("2 · Nová fotka persony", [sd], PURPLE)
    save = g.add("SaveImage", {"filename_prefix": "persona/03_foto"}, pos=(940, 0), size=(460, 680),
                 title="NOVÁ FOTKA", images=(sd, 0))
    g.group("3 · Výstup", [save], GREY)
    note(g, NOTE_06, (0, 1480), (1400, 720))
    return g


def build_07():
    g = Graph()
    first = load(g, "persona_foto.png", "A · PRVÝ SNÍMOK (fotka z workflow 06)", (0, 0))
    vi = kling_i2v(g, VIDEO_I2V, (420, 0), "A · Kling 4.0 – fotka → video")
    g.connect(first, 0, vi, "model.first_frame")
    sv = save_video(g, "video/persona_A", (920, 0))
    g.connect(vi, 0, sv, "video")
    g.group("A · Fotka → video (zapnuté)", [first, vi, sv], ORANGE)

    r1 = load(g, "persona_ref1.png", "B · REF 1 – tvár zblízka", (0, 760))
    r2 = load(g, "persona_ref3.png", "B · REF 2 – celá postava", (0, 1240))
    vr = kling_r2v(g, VIDEO_R2V, (420, 760), "B · Kling 4.0 – referencie → video (vypnuté)", mode=4)
    g.connect(r1, 0, vr, "model.reference_images.image_1")
    g.connect(r2, 0, vr, "model.reference_images.image_2")
    sv2 = save_video(g, "video/persona_B", (920, 760), mode=4)
    g.connect(vr, 0, sv2, "video")
    g.group("B · Referencie → video (zapni Ctrl+B, A vypni)", [r1, r2, vr, sv2], BLUE)
    note(g, NOTE_07, (0, 1760), (1380, 760))
    return g


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, builder in [
        ("04_persona_1_fuzia_seedream", build_04),
        ("05_persona_2_referencna_sada_seedream", build_05),
        ("06_persona_3_nove_fotky_seedream", build_06),
        ("07_persona_4_video_kling", build_07),
    ]:
        g = builder()
        with open(os.path.join(OUT_DIR, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(g.ui_json(), f, ensure_ascii=False, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    main()
