"""Временный отсоединённый K/V-контекст без изменения штатного forward модели."""

from .contracts import ModelsError


def forward_with_context_reference(torch, wrapper, image, reference, *, return_two_class_logits, supervise_reference=False):
    if (not isinstance(image, torch.Tensor) or not isinstance(reference, torch.Tensor)
            or image.ndim != 4 or reference.ndim != 4
            or image.shape[:2] != reference.shape[:2] or image.device != reference.device
            or reference.shape[-2] != reference.shape[-1]
            or min(reference.shape[-2:]) < 32
            or any(side % 32 for side in (*image.shape[-2:], *reference.shape[-2:]))
            or any(crop > full for crop, full in zip(reference.shape[-2:], image.shape[-2:]))):
        raise ModelsError("K/V-контекст требует совместимую квадратную вырезку того же batch, каналов и устройства, со сторонами кратными32")
    if any(module.training for module in wrapper.modules()):
        raise ModelsError("Подмена K/V-контекста допускается только в eval-режиме")
    encoder = getattr(wrapper.model, "encoder", None)
    projections = []
    for stage in range(1, 5):
        blocks = getattr(encoder, f"block{stage}", None)
        if not isinstance(blocks, (torch.nn.Sequential, torch.nn.ModuleList)) or not len(blocks):
            raise ModelsError("Неожиданная структура стадий SMP SegFormer для K/V-контекста")
        for block in blocks:
            attention = getattr(block, "attn", None)
            query, key_value = getattr(attention, "q", None), getattr(attention, "kv", None)
            if (not isinstance(query, torch.nn.Linear) or not isinstance(key_value, torch.nn.Linear)
                    or key_value.out_features != 2 * query.out_features):
                raise ModelsError("Неожиданная структура attention SMP SegFormer для K/V-контекста")
            projections.append(key_value)

    values, handles = {}, []

    def capture(index):
        def hook(module, inputs, output):
            if index in values or output.ndim != 3:
                raise ModelsError("K/V-проекция должна выполняться однократно и возвращать трёхмерный тензор")
            values[index] = output.detach().clone()
        return hook

    def replace(index):
        def hook(module, inputs, output):
            value = values[index]
            if (value.shape[0] != output.shape[0] or value.shape[2:] != output.shape[2:]
                    or value.dtype != output.dtype or value.device != output.device):
                raise ModelsError("Несовместимые K/V-проекции полного входа и вырезки")
            return value
        return hook

    try:
        for index, projection in enumerate(projections):
            handles.append(projection.register_forward_hook(capture(index)))
        with torch.set_grad_enabled(torch.is_grad_enabled() and supervise_reference):
            reference_logits = wrapper(reference, return_two_class_logits=True)
        if not supervise_reference:
            reference_logits = None
    finally:
        for handle in handles:
            handle.remove()
    if len(values) != len(projections):
        raise ModelsError("Не все attention-блоки предоставили K/V-контекст вырезки")
    handles = []
    try:
        for index, projection in enumerate(projections):
            handles.append(projection.register_forward_hook(replace(index)))
        query_logits = wrapper(image, return_two_class_logits=return_two_class_logits)
        return (query_logits, reference_logits) if supervise_reference else query_logits
    finally:
        for handle in handles:
            handle.remove()
