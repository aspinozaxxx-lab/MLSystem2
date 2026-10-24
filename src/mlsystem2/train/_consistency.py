"""Согласованность общих допустимых пикселей полного train-окна и его вырезки."""

from __future__ import annotations

from .contracts import TrainError


def context_consistency_loss(torch, model, images, masks, valid_pixels, class_negative, *, seed, epoch, batch_index):
    """Вернуть дополнительный KL-граф; состояния модулей и исходный вход не изменять."""
    stats = {"attempted_batches": 1, "selected_batches": 0, "supervised_pixels": 0,
             "kl_sum": 0.0, "weight": 0.25, "probability": 0.5}
    height, width = images.shape[-2:]
    if min(height, width) < 64:
        return None, stats
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch * 1_000_003 + batch_index * 97)
    if float(torch.rand((), generator=generator)) >= 0.5:
        return None, stats
    fraction = (0.5, 0.75)[int(torch.randint(2, (), generator=generator))]
    side = max(32, int(min(height, width) * fraction) // 32 * 32)
    top = int(torch.randint((height - side) // 32 + 1, (), generator=generator)) * 32
    left = int(torch.randint((width - side) // 32 + 1, (), generator=generator)) * 32
    region = (..., slice(top, top + side), slice(left, left + side))
    target = masks[:, 0].long() if masks.ndim == 4 else masks.long()
    target = target[region]
    valid = torch.ones_like(target, dtype=torch.bool)
    if valid_pixels is not None:
        usable = valid_pixels[:, 0] if valid_pixels.ndim == 4 else valid_pixels
        valid &= usable[region]
    if class_negative is not None:
        valid &= ~class_negative.any(dim=1)[region]
    if not valid.any():
        return None, stats
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            teacher = model(images, return_two_class_logits=True).softmax(dim=1)[region]
            if not torch.isfinite(teacher).all():
                raise TrainError("Неконечный прогноз учителя согласованности")
            confidence, labels = teacher.max(dim=1)
            selected = valid & (labels == target) & (confidence > 0.9)
        if not selected.any():
            return None, stats
        logits = model(images[region].contiguous(), return_two_class_logits=True)
        log_probability = logits.log_softmax(dim=1)
        # log_softmax сохраняет градиент даже у уверенного, но неверного ученика.
        kl = torch.nn.functional.kl_div(log_probability, teacher, reduction="none").sum(dim=1)
        means = [kl[selected & (target == index)].mean()
                 for index in range(teacher.shape[1]) if (selected & (target == index)).any()]
        divergence = torch.stack(means).mean().clamp_min(0)
        stats.update(selected_batches=1, supervised_pixels=int(selected.sum().item()),
                     kl_sum=float(divergence.detach().item()))
        return 0.25 * divergence, stats
    finally:
        for module, training in states:
            module.training = training
