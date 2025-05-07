import torch
from FovConvNeXt.models import make_model

# Model parameters
n_fixations = 1
radius = 0.4
block_sigma = 0.8
block_max_ord = 4
patch_sigma = 1.0
patch_max_ord = 4
ds_sigma = 0.6
ds_max_ord = 0

model = make_model(
        n_fixations=n_fixations,
        n_classes=100,  # Use full 100 classes
        radius=radius,
        block_sigma=block_sigma,
        block_max_ord=block_max_ord,
        patch_sigma=patch_sigma,
        patch_max_ord=patch_max_ord,
        ds_sigma=ds_sigma,
        ds_max_ord=ds_max_ord
    )

model.fc = torch.nn.Linear(320, 2)

model.load_state_dict(torch.load('best_model_foveated.pth'))
model = model.cuda()

# Dummy input matching your model's expected input shape
dummy_input = torch.randn(1, 3, 224, 224).cuda()  # Or .cpu() if no CUDA


# Export the model
torch.onnx.export(
    model,                           # Your PyTorch model
    dummy_input,                     # Dummy input(s)
    "model.onnx",                    # Output filename
    export_params=True,              # Store trained weights
    opset_version=16,                # Use 11 or higher
    do_constant_folding=True,        # Fold constants
    input_names=["input"],           # Optional: name inputs
    output_names=["output"],         # Optional: name outputs
    dynamic_axes={                   # Optional: for dynamic shapes
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)