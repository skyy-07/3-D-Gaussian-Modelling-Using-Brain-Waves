"""Stage 1 -> Stage 2 fusion."""

from .attention import (
    AttentionBlob,
    AttentionField,
    assign_object_salience,
    bbox_mask,
    blobs_from_objects,
    build_attention_field,
    colorize,
    default_masks,
    hemifield_blob,
    latent_bias,
    object_summary,
    overlay,
    resize_bilinear,
)

__all__ = [
    "AttentionBlob",
    "AttentionField",
    "assign_object_salience",
    "bbox_mask",
    "blobs_from_objects",
    "build_attention_field",
    "colorize",
    "default_masks",
    "hemifield_blob",
    "latent_bias",
    "object_summary",
    "overlay",
    "resize_bilinear",
]
