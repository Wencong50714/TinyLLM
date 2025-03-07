import triton
import triton.language as tl

def triton_cross_entropy(
    logits,
    targets,
    n_classes,
    n_classes_per_token,
):
    pass