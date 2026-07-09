#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Rebuild Moondream / semantic navigation Python environment
# For:
#   - Moondream 2B 4-bit local image description
#   - Ollama qwen2.5:7b planner
#   - graph-based semantic navigation scripts
#
# Default project folder:
#   /home/vladkobli/rocon-demos/moondream
#
# Usage:
#   ./rebuild_moondream_env.sh
#   ./rebuild_moondream_env.sh --force
#   ./rebuild_moondream_env.sh --pull-ollama
#   ./rebuild_moondream_env.sh --force --pull-ollama
#   ./rebuild_moondream_env.sh --torch-cuda cu121
#   ./rebuild_moondream_env.sh --torch-cuda cpu
# ============================================================

PROJECT_DIR="/home/vladkobli/rocon-demos/moondream"
VENV_NAME="venv_moondream"
VENV_DIR=""
FORCE=0
PULL_OLLAMA=0
SKIP_MODEL_TEST=0

# Good default for RTX 4070 on modern NVIDIA drivers.
# You can change to cu124/cu126/cu128 if your installed PyTorch supports it.
TORCH_CUDA="cu121"

MOONDREAM_MODEL="moondream/moondream-2b-2025-04-14-4bit"
OLLAMA_MODEL="qwen2.5:7b"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --venv-name)
            VENV_NAME="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --pull-ollama)
            PULL_OLLAMA=1
            shift
            ;;
        --skip-model-test)
            SKIP_MODEL_TEST=1
            shift
            ;;
        --torch-cuda)
            TORCH_CUDA="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--force] [--pull-ollama] [--skip-model-test] [--torch-cuda cu121|cu124|cu126|cpu] [--project-dir PATH]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

VENV_DIR="${PROJECT_DIR}/${VENV_NAME}"

echo "============================================================"
echo "Moondream environment rebuild"
echo "Project dir:      ${PROJECT_DIR}"
echo "Venv dir:         ${VENV_DIR}"
echo "Torch CUDA mode:  ${TORCH_CUDA}"
echo "Moondream model:  ${MOONDREAM_MODEL}"
echo "Ollama model:     ${OLLAMA_MODEL}"
echo "Force reinstall:  ${FORCE}"
echo "Pull Ollama:      ${PULL_OLLAMA}"
echo "============================================================"

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Creating project directory: ${PROJECT_DIR}"
    mkdir -p "${PROJECT_DIR}"
fi

cd "${PROJECT_DIR}"

echo
echo "[1/9] Checking system dependencies..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found."
    echo "Install it with:"
    echo "  sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "ERROR: python3-venv is missing."
    echo "Install it with:"
    echo "  sudo apt update && sudo apt install -y python3-venv python3-pip"
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "WARNING: git not found. Installing git is recommended:"
    echo "  sudo apt install -y git"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "WARNING: nvidia-smi not found. GPU may not be available inside this environment."
fi

echo
echo "[2/9] Removing old venv if requested..."

if [[ -d "${VENV_DIR}" && "${FORCE}" -eq 1 ]]; then
    echo "Removing old venv: ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
elif [[ -d "${VENV_DIR}" ]]; then
    echo "Existing venv found. Reusing it."
    echo "Use --force to recreate it from zero."
fi

echo
echo "[3/9] Creating venv..."

if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "Python used:"
which python
python --version

echo
echo "[4/9] Upgrading pip/setuptools/wheel..."

python -m pip install --upgrade pip setuptools wheel

echo
echo "[5/9] Installing PyTorch..."

if [[ "${TORCH_CUDA}" == "cpu" ]]; then
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [[ "${TORCH_CUDA}" == "cu121" ]]; then
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "${TORCH_CUDA}" == "cu124" ]]; then
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
elif [[ "${TORCH_CUDA}" == "cu126" ]]; then
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
    echo "Unknown --torch-cuda value: ${TORCH_CUDA}"
    echo "Use one of: cpu, cu121, cu124, cu126"
    exit 1
fi

echo
echo "[6/9] Installing Moondream / navigation Python dependencies..."

cat > requirements_moondream.txt <<'EOF'
transformers==4.49.0
accelerate
bitsandbytes
torchao==0.7.0
huggingface_hub
safetensors
tokenizers
einops
Pillow
numpy
opencv-python
matplotlib
tqdm
ollama
pydantic
requests
EOF

python -m pip install --upgrade -r requirements_moondream.txt

echo
echo "[7/9] Creating helper scripts..."

cat > activate_moondream.sh <<EOF
#!/usr/bin/env bash
cd "${PROJECT_DIR}"
source "${VENV_DIR}/bin/activate"
echo "Activated ${VENV_DIR}"

# Optional Hugging Face token.
# Put HF_TOKEN=hf_... in /home/vladkobli/rocon-demos/moondream/.env
# or export it before running this script.
if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    source "${PROJECT_DIR}/.env"
    set +a
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
    echo "HF_TOKEN is set."
else
    echo "WARNING: HF_TOKEN is not set. Hugging Face downloads may be rate-limited."
fi

python --version
EOF

chmod +x activate_moondream.sh

cat > verify_torch.py <<'EOF'
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device="cuda")
    y = x @ x
    print("CUDA tensor test:", float(y[0, 0]))
else:
    print("WARNING: CUDA is not available to PyTorch.")
EOF

cat > verify_moondream.py <<EOF
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM
import torch

MODEL_ID = "${MOONDREAM_MODEL}"

print("Torch CUDA available:", torch.cuda.is_available())
print("Loading model:", MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map={"": "cuda"} if torch.cuda.is_available() else {"": "cpu"},
)

model.generation_config.do_sample = False
model.generation_config.max_new_tokens = 40

img = Image.new("RGB", (512, 384), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((120, 120, 390, 260), outline="black", width=5)
draw.text((150, 170), "test image", fill="black")

answer = model.query(img, "Describe the image in one short sentence.")["answer"]

print("Moondream answer:")
print(answer)
print("Moondream test finished.")
EOF

cat > verify_ollama.py <<EOF
from ollama import chat

MODEL = "${OLLAMA_MODEL}"

print("Testing Ollama model:", MODEL)

response = chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Return only the word OK."}
    ],
    options={
        "temperature": 0,
        "num_predict": 10,
    },
)

if isinstance(response, dict):
    print(response["message"]["content"])
else:
    print(response.message.content)
EOF

echo
echo "[8/9] Testing PyTorch..."

python verify_torch.py

echo
echo "[9/9] Optional model checks..."

if [[ "${PULL_OLLAMA}" -eq 1 ]]; then
    if command -v ollama >/dev/null 2>&1; then
        echo "Pulling Ollama model: ${OLLAMA_MODEL}"
        ollama pull "${OLLAMA_MODEL}"
    else
        echo "WARNING: Ollama command not found."
        echo "Install Ollama first, then run:"
        echo "  ollama pull ${OLLAMA_MODEL}"
    fi
else
    echo "Skipping Ollama pull. Use --pull-ollama if needed."
fi

if [[ "${SKIP_MODEL_TEST}" -eq 0 ]]; then
    echo
    echo "Testing Moondream model load."
    echo "This may take a while the first time because the model is downloaded."
    python verify_moondream.py
else
    echo "Skipping Moondream model test."
fi

echo
echo "============================================================"
echo "DONE"
echo
echo "Activate with:"
echo "  cd ${PROJECT_DIR}"
echo "  source ${VENV_NAME}/bin/activate"
echo
echo "Or:"
echo "  source ${PROJECT_DIR}/activate_moondream.sh"
echo
echo "Run checks:"
echo "  python verify_torch.py"
echo "  python verify_moondream.py"
echo
echo "If using Ollama planner:"
echo "  ollama serve"
echo "  ollama pull ${OLLAMA_MODEL}"
echo "  python verify_ollama.py"
echo "============================================================"