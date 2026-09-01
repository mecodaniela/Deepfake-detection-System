"""
model.py — EfficientNet-B0 pretrained (transfer learning) për klasifikim
binar real/fake në nivel frame.

Strategji: freeze shumica e backbone-it, fine-tune N blloqe të fundit + 
classifier head i ri. N (num_unfrozen_blocks) është i rregullueshëm —
fillo me 2-3, rrit nëse saktësia mbi val s'është e mjaftueshme.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(num_unfrozen_blocks: int = 3, num_classes: int = 2) -> nn.Module:
    """
    Ndërton EfficientNet-B0 pretrained me freeze selektiv.

    num_unfrozen_blocks: sa nga blloqet e fundit të 'features' (ka gjithsej
                          9, indeksuar 0-8) do të mbeten trainueshëm.
                          P.sh. 3 -> blloqet 6, 7, 8 janë trainueshëm,
                          blloqet 0-5 janë frozen.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # --- Freeze gjithçka fillimisht ---
    for param in model.parameters():
        param.requires_grad = False

    # --- Zbllokon N blloqet e fundit të 'features' (backbone-i) ---
    total_blocks = len(model.features)  # 9 për EfficientNet-B0
    unfreeze_from = max(0, total_blocks - num_unfrozen_blocks)

    for i in range(unfreeze_from, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # --- Zëvendëso classifier head-in (1000 klasa ImageNet -> 2 klasa yni) ---
    classifier_last_layer = model.classifier[1]
    assert isinstance(classifier_last_layer, nn.Linear), \
    "Pritej që classifier[1] të jetë nn.Linear — struktura e EfficientNet-it ndryshoi?"
    in_features = classifier_last_layer.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    # classifier i ri është gjithmonë trainueshëm (requires_grad=True default)

    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Kthen (parametra_trainueshëm, parametra_gjithsej) — për verifikim shpejt."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    model = build_model(num_unfrozen_blocks=3)
    trainable, total = count_trainable_params(model)
    print(f"Parametra trainueshëm: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

    # Test i shpejtë me input fiktiv (batch=2, RGB, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # duhet: torch.Size([2, 2])