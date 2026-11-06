"""Оценка подгоняемого пакета и сравнение с начальными весами."""

from .contracts import EpochMetrics, TrainError


def evaluate_package(request, epoch, pause):
    values = request.package_evaluator(request.model, epoch, pause)
    if values.get("package_evaluation", {}).get("scope") != "training_package":
        raise TrainError("Оценщик подгонки должен проверять весь обучаемый пакет.")
    return values


def package_epoch_metrics(request, values, epoch, train_loss, learning_rate, elapsed):
    return EpochMetrics(
        epoch=epoch, validation_performed=True, train_loss=train_loss, val_loss=None,
        quality_metric=request.config.quality_metric, learning_rate=learning_rate,
        epoch_time_sec=elapsed, package_evaluation=values["package_evaluation"],
        val_quality_f1=values["quality_f1"], val_quality_precision=values["quality_precision"],
        val_quality_recall=values["quality_recall"], val_best_threshold=values["threshold"],
        val_best_pixel_threshold=values["threshold"], val_best_threshold_pixel_f1=values["pixel_f1"],
        val_best_threshold_pixel_precision=values["pixel_precision"],
        val_best_threshold_pixel_recall=values["pixel_recall"],
        val_best_threshold_precision=values["quality_precision"],
        val_best_threshold_recall=values["quality_recall"],
        val_best_threshold_object_f1=values["object_f1"],
        val_best_threshold_object_precision=values["object_precision"],
        val_best_threshold_object_recall=values["object_recall"],
        val_per_scene_metrics=values.get("per_scene_metrics", []),
        val_per_class_metrics=values.get("per_class_metrics", []),
        val_macro_pixel_f1=values.get("macro_pixel_f1"),
        val_micro_pixel_f1=values.get("micro_pixel_f1"),
        val_object_separation=values.get("object_separation", {}),
    )


def package_rank(metrics):
    counts = metrics.package_evaluation.get("quality_counts", {})
    return (float(metrics.val_quality_f1), float(metrics.val_quality_precision),
            -int(counts.get("fp", 0)), -int(counts.get("fn", 0)))


def set_package_epoch(loader, epoch):
    raw = getattr(loader, "loader", loader)
    sampler = getattr(raw, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def package_selection(best, baseline):
    return {
        "metric": "fit/object_f1" if best.quality_metric == "objects" else "fit/pixel_f1",
        "mode": "max", "epoch": best.epoch, "val_loss": None,
        "quality_f1": best.val_quality_f1, "pixel_f1": best.val_best_threshold_pixel_f1,
        "object_f1": best.val_best_threshold_object_f1,
        "macro_pixel_f1": best.val_macro_pixel_f1,
        "baseline_f1": baseline.val_quality_f1, "baseline_selected": best.epoch == 0,
        "evaluation_scope": "training_package", "independent": False,
    }
