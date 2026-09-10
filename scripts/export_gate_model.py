"""One-time local export of a pretrained ImageNet classifier to ONNX, used as an
'is this even a strawberry' gate before the ripe/unripe model runs. torch/torchvision
are only needed to run this script - the deployed webapp only ships the resulting
.onnx file plus the lightweight onnxruntime package.
"""
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights)
model.eval()

# Confirmed via weights.meta["categories"][949] == "strawberry" (standard ImageNet-1k ordering)
dummy_input = torch.randn(1, 3, 224, 224)

output_path = "webapp/api/model/gate_model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

print(f"Exported gate model to {output_path}")
