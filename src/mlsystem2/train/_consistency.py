"""Согласованность общих допустимых пикселей полного train-окна и его вырезки."""

from __future__ import annotations

from .contracts import TrainError


def full_query_context_loss(torch, model, images, masks, valid_pixels, class_negative, *, seed, epoch, batch_index, supervise_reference=False):
    """Обучить полный query-вход по train-target при отсоединённом контексте вырезки."""
    stats = {"attempted_batches": 1, "selected_batches": 0, "supervised_pixels": 0,
             "half_side_selected_batches": 0,
             "three_quarter_side_selected_batches": 0, "context_loss_sum": 0.0}
    stats.update({"query_supervised_pixels": 0, "reference_supervised_pixels": 0,
                  "query_contribution_sum": 0.0, "reference_contribution_sum": 0.0}
                 if supervise_reference else {"reference_valid_pixels": 0})
    height, width = images.shape[-2:]
    if min(height, width) < 64:
        return None, stats
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch * 1_000_003 + batch_index * 97)
    if float(torch.rand((), generator=generator)) >= 0.5:
        return None, stats
    fraction_index = int(torch.randint(2, (), generator=generator))
    fraction = (0.5, 0.75)[fraction_index]
    side = max(32, int(min(height, width) * fraction) // 32 * 32)
    top = int(torch.randint((height - side) // 32 + 1, (), generator=generator)) * 32
    left = int(torch.randint((width - side) // 32 + 1, (), generator=generator)) * 32
    region = (..., slice(top, top + side), slice(left, left + side))
    target = masks[:, 0].long() if masks.ndim == 4 else masks.long()
    valid = torch.ones_like(target, dtype=torch.bool)
    if valid_pixels is not None:
        valid &= valid_pixels[:, 0] if valid_pixels.ndim == 4 else valid_pixels
    if class_negative is not None:
        valid &= ~class_negative.any(dim=1)
    if not valid.any() or not valid[region].any():
        return None, stats
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        result = model(images, return_two_class_logits=True, context_reference=images[region].contiguous())
        if supervise_reference:
            if not isinstance(result, tuple) or len(result) != 2:
                raise TrainError("Профиль v6 требует пару прогнозов query и источника K/V")
            logits, reference_logits = result
        else:
            logits = result
        query_loss = _balanced_ground_truth_ce(torch, logits, target, valid)
        query_pixels, reference_pixels = int(valid.sum().item()), int(valid[region].sum().item())
        if supervise_reference:
            reference_loss = _balanced_ground_truth_ce(torch, reference_logits, target[region], valid[region])
            loss = torch.stack((query_loss, reference_loss)).mean()
            stats.update(query_supervised_pixels=query_pixels, reference_supervised_pixels=reference_pixels,
                         query_contribution_sum=0.5 * float(query_loss.detach().item()),
                         reference_contribution_sum=0.5 * float(reference_loss.detach().item()))
        else:
            loss = query_loss
            stats["reference_valid_pixels"] = reference_pixels
        stats.update(selected_batches=1,
                     supervised_pixels=query_pixels + reference_pixels if supervise_reference else query_pixels,
                     context_loss_sum=float(loss.detach().item()))
        stats[("half_side_selected_batches", "three_quarter_side_selected_batches")[fraction_index]] = 1
        return 0.25 * loss, stats
    finally:
        for module, training in states:
            module.training = training


def _balanced_ground_truth_ce(torch, logits, target, valid):
    if ((target[valid] < 0) | (target[valid] >= logits.shape[1])).any():
        raise TrainError("Недопустимый класс допустимого query-пикселя согласованности")
    labels = target.masked_fill(~valid, -100)
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
    classes = [valid & (target == index) for index in range(logits.shape[1]) if (valid & (target == index)).any()]
    return torch.stack([ce[mask].mean() for mask in classes]).mean()


def context_consistency_loss(torch, model, images, masks, valid_pixels, class_negative, *, seed, epoch, batch_index, bidirectional=False, ground_truth_fallback=False):
    """Вернуть дополнительный KL-граф; состояния модулей и исходный вход не изменять."""
    stats = {"attempted_batches": 1, "selected_batches": 0, "supervised_pixels": 0,
             "kl_sum": 0.0, "weight": 0.25, "probability": 0.5}
    if bidirectional:
        stats.update(forward_selected_batches=0, reverse_selected_batches=0,
                     forward_supervised_pixels=0, reverse_supervised_pixels=0,
                     forward_kl_sum=0.0, reverse_kl_sum=0.0)
    if ground_truth_fallback:
        if not bidirectional:
            raise TrainError("Надзор неохваченных пикселей требует двух направлений")
        stats.update(fallback_selected_batches=0, fallback_supervised_pixels=0,
                     context_loss_sum=0.0, teacher_contribution_sum=0.0,
                     fallback_ce_contribution_sum=0.0)
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
        if bidirectional:
            full_logits = model(images, return_two_class_logits=True)[region]
            crop_logits = model(images[region].contiguous(), return_two_class_logits=True)
            probabilities = [logits.detach().softmax(dim=1) for logits in (full_logits, crop_logits)]
            if not all(torch.isfinite(value).all() for value in probabilities):
                raise TrainError("Неконечный прогноз учителя согласованности")
            if ground_truth_fallback:
                return _ground_truth_fallback_loss(
                    torch, full_logits, crop_logits, probabilities, target, valid, stats,
                )
            terms = []
            for direction, teacher, student in (
                ("forward", probabilities[0], crop_logits),
                ("reverse", probabilities[1], full_logits),
            ):
                confidence, labels = teacher.max(dim=1)
                selected = valid & (labels == target) & (confidence > 0.9)
                if not selected.any():
                    continue
                kl = torch.nn.functional.kl_div(student.log_softmax(dim=1), teacher, reduction="none").sum(dim=1)
                means = [kl[selected & (target == index)].mean()
                         for index in range(teacher.shape[1]) if (selected & (target == index)).any()]
                divergence = torch.stack(means).mean().clamp_min(0)
                terms.append(divergence)
                pixels = int(selected.sum().item())
                stats[direction + "_selected_batches"] = 1
                stats[direction + "_supervised_pixels"] = pixels
                stats[direction + "_kl_sum"] = float(divergence.detach().item())
                stats["supervised_pixels"] += pixels
            if not terms:
                return None, stats
            divergence = torch.stack(terms).mean()
            stats.update(selected_batches=1, kl_sum=float(divergence.detach().item()))
            return 0.25 * divergence, stats
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


def _ground_truth_fallback_loss(torch, full_logits, crop_logits, probabilities, target, valid, stats):
    """На неохваченных допустимых пикселях обучать оба контекста по train-target."""
    if ((target[valid] < 0) | (target[valid] >= full_logits.shape[1])).any():
        raise TrainError("Недопустимый класс допустимого пикселя согласованности")
    qualified = []
    for teacher in probabilities:
        confidence, labels = teacher.max(dim=1)
        qualified.append(valid & (labels == target) & (confidence > 0.9))
    uncovered = valid & ~qualified[0] & ~qualified[1]
    safe_target = target.masked_fill(~valid, 0)
    terms, teacher_terms, teacher_contributions, fallback_contributions = [], [], [], []
    for direction, teacher, student, selected in (
        ("forward", probabilities[0], crop_logits, qualified[0]),
        ("reverse", probabilities[1], full_logits, qualified[1]),
    ):
        active = selected | uncovered
        if not active.any():
            continue
        log_probability = student.log_softmax(dim=1)
        kl = torch.nn.functional.kl_div(log_probability, teacher, reduction="none").sum(dim=1)
        ce = -log_probability.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        if selected.any():
            means = [kl[selected & (target == index)].mean()
                     for index in range(teacher.shape[1]) if (selected & (target == index)).any()]
            teacher_divergence = torch.stack(means).mean().clamp_min(0)
            teacher_terms.append(teacher_divergence)
            stats[direction + "_selected_batches"] = 1
            stats[direction + "_supervised_pixels"] = int(selected.sum().item())
            stats[direction + "_kl_sum"] = float(teacher_divergence.detach().item())
        mixed = torch.where(uncovered, ce, kl)
        classes = [active & (target == index) for index in range(teacher.shape[1])
                   if (active & (target == index)).any()]
        term = torch.stack([mixed[mask].mean() for mask in classes]).mean().clamp_min(0)
        teacher_part = torch.stack([(kl * selected)[mask].mean() for mask in classes]).mean()
        fallback_part = torch.stack([(ce * uncovered)[mask].mean() for mask in classes]).mean()
        terms.append(term)
        teacher_contributions.append(teacher_part)
        fallback_contributions.append(fallback_part)
        stats["supervised_pixels"] += int(active.sum().item())
    if not terms:
        return None, stats
    divergence = torch.stack(terms).mean()
    stats.update(selected_batches=1,
        kl_sum=float(torch.stack(teacher_terms).mean().detach().item()) if teacher_terms else 0.0,
        fallback_selected_batches=int(uncovered.any().item()),
        fallback_supervised_pixels=2 * int(uncovered.sum().item()),
        context_loss_sum=float(divergence.detach().item()),
        teacher_contribution_sum=float(torch.stack(teacher_contributions).mean().detach().item()),
        fallback_ce_contribution_sum=float(torch.stack(fallback_contributions).mean().detach().item()))
    return 0.25 * divergence, stats
