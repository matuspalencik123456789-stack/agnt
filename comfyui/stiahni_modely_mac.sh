#!/usr/bin/env bash
# Stiahne modely + workflow pre "telo z fotky 1 + tvár z fotky 2" (FLUX.2 Klein 4B + BFS)
# do existujúcej inštalácie ComfyUI. Verzia pre Mac (bf16 model – fp8 na Macu s LoRA padá).
#
# Použitie:  bash stiahni_modely_mac.sh /cesta/k/ComfyUI [base]
#   base = stiahni aj nedestilovaný Klein 4B base (+7.8 GB) pre realistický workflow 03.
# Bez cesty skúsi ~/ComfyUI. Prerušené sťahovanie sa pri ďalšom spustení dokončí.
set -euo pipefail

BRANCH_RAW="https://raw.githubusercontent.com/matuspalencik123456789-stack/agnt/claude/lucid-goldberg-86v1xi/comfyui/workflows"

# dl <priečinok v models/> <súbor> <očakávaná veľkosť v bajtoch> <url>
dl() {
  local dir="$COMFY/models/$1" file="$2" size="$3" url="$4"
  mkdir -p "$dir"
  if [[ -f "$dir/$file" && "$(wc -c < "$dir/$file" | tr -d ' ')" == "$size" ]]; then
    echo "✓ $file už je stiahnutý"
    return
  fi
  echo "→ sťahujem $file do models/$1/"
  curl -L --fail --retry 5 --retry-delay 3 -C - -o "$dir/$file" "$url"
  local got
  got="$(wc -c < "$dir/$file" | tr -d ' ')"
  if [[ "$got" != "$size" ]]; then
    echo "✗ $file má $got bajtov, čakal som $size. Spusti skript znova." >&2
    exit 1
  fi
}

main() {
  COMFY="${1:-$HOME/ComfyUI}"
  COMFY="${COMFY%/}"
  # Manuálna inštalácia má main.py; ComfyUI Desktop app má v base path len models/, user/, .venv ...
  if [[ ! -f "$COMFY/main.py" && ! -d "$COMFY/models" ]]; then
    echo "Nenašiel som ComfyUI v: $COMFY"
    echo "Spusti: bash $0 /cesta/k/ComfyUI   (priečinok, v ktorom je models/)"
    exit 1
  fi
  echo "ComfyUI: $COMFY"

  dl diffusion_models flux-2-klein-4b.safetensors 7751105712 \
    "https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors"
  dl text_encoders qwen_3_4b.safetensors 8044982048 \
    "https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
  dl vae flux2-vae.safetensors 336211292 \
    "https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/vae/flux2-vae.safetensors"
  dl loras bfs_head_v1_flux-klein_4b.safetensors 369644384 \
    "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_4b.safetensors"
  if [[ "${2:-}" == "base" ]]; then
    dl diffusion_models flux-2-klein-base-4b.safetensors 7751105712 \
      "https://huggingface.co/Comfy-Org/Flux-2-Klein-4B/resolve/main/split_files/diffusion_models/flux-2-klein-base-4b.safetensors"
  fi

  # Workflowy -> ComfyUI bočný panel "Workflows"
  local wf="$COMFY/user/default/workflows"
  mkdir -p "$wf"
  curl -L --fail -s -o "$wf/modelka_telo+tvar_16GB.json" \
    "$BRANCH_RAW/01_modelka_telo%2Btvar_flux2klein_BFS_16GB.json"
  curl -L --fail -s -o "$wf/modelka_telo+tvar_reactor_zaloha.json" \
    "$BRANCH_RAW/02_modelka_telo%2Btvar_reactor_zaloha.json"
  curl -L --fail -s -o "$wf/modelka_telo+tvar_BASE_realisticka.json" \
    "$BRANCH_RAW/03_modelka_telo%2Btvar_flux2klein_BASE_realisticka_16GB.json"
  echo "✓ workflowy uložené do $wf"

  if [[ -f "$COMFY/models/diffusion_models/flux-2-klein-4b-fp8.safetensors" ]]; then
    echo
    echo "Tip: fp8 verzia na Macu nefunguje. Keď ti workflow pobeží, môžeš ju zmazať (uvoľní 4 GB):"
    echo "  rm \"$COMFY/models/diffusion_models/flux-2-klein-4b-fp8.safetensors\""
  fi
  echo
  start_hint
}

start_hint() {
  if [[ -f "$COMFY/main.py" ]]; then
    echo "Hotovo. Spusti ComfyUI:"
    echo "  cd \"$COMFY\" && source .venv/bin/activate && python main.py"
    echo "a otvor http://127.0.0.1:8188  → panel Workflows (klávesa W) → modelka_telo+tvar_16GB"
  else
    echo "Hotovo. Toto je priečinok ComfyUI Desktop aplikácie:"
    echo "  zatvor a znova otvor aplikáciu ComfyUI → panel Workflows (klávesa W) → modelka_telo+tvar_16GB"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
