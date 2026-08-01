#!/usr/bin/env bash
set -e

ENV_NAME="drl_mac_env"

echo "========================================================="
echo " macOS Apple Silicon GPU (MPS) Environment Setup Script "
echo "========================================================="

# # 1. Ensure Xcode Command Line Tools are installed (Required for Box2D & Metal)
# if ! xcode-select -p &>/dev/null; then
#     echo "Installing Xcode Command Line Tools..."
#     xcode-select --install
# else
#     echo "✔ Xcode Command Line Tools already installed."
# fi

# 2. Create Python Virtual Environment
echo "Creating virtual environment: $ENV_NAME..."
python3 -m venv $ENV_NAME

# 3. Activate Virtual Environment
source $ENV_NAME/bin/activate

# 4. Upgrade core build tools
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# 5. Install PyTorch with MPS (Metal) GPU Support and project requirements
echo "Installing project requirements and PyTorch for Apple Silicon..."
pip install -r requirements.txt

# 6. Register Kernel with IPykernel for Jupyter / VS Code / PyCharm
echo "Registering Jupyter Kernel..."
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python (DRL Mac GPU)"

# 7. Verification Step
echo "========================================================="
echo " Verifying Apple Silicon GPU (MPS) Acceleration..."
echo "========================================================="

python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
if torch.backends.mps.is_available():
    print(' SUCCESS: Apple Silicon GPU (MPS) is available!')
    device = torch.device('mps')
    x = torch.ones(1, device=device)
    print(f'   Test Tensor on GPU: {x}')
else:
    print(' WARNING: MPS is not available. Check your macOS version (requires macOS 12.3+).')
"

echo "========================================================="
echo "Setup Complete!"
echo "To activate manually in terminal:"
echo "   source $ENV_NAME/bin/activate"
echo ""
echo "In Jupyter Notebook / VS Code:"
echo "   Select 'Python (DRL Mac GPU)' kernel."
echo "========================================================="