#!/bin/bash
echo "=== Enemy Agent CUDA Training ==="
echo

echo "Step 1: Verifying CUDA extensions..."
python verify_cuda_extensions.py
if [ $? -ne 0 ]; then
    echo
    echo "Error: CUDA extensions verification failed."
    exit 1
fi

echo
echo "Step 2: Training enemy agent with custom CUDA..."
python train_enemy_cuda.py "$@"
if [ $? -ne 0 ]; then
    echo
    echo "Error: Training failed."
    exit 1
fi

echo
echo "=== Training completed successfully! ==="
echo "The enemy agent was trained using custom C++/CUDA modules on your NVIDIA GPU."
echo