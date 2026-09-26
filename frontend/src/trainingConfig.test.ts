import { describe, expect, it } from "vitest";

import {
  configWithField,
  trainingConfigFieldVisible,
  trainingConfigSchema,
} from "./utils/trainingConfig";
import type { ConfigSchema } from "./api/types";

const schema: ConfigSchema = {
  fields: [
    {
      key: "train.loss",
      label: "Loss",
      value_type: "select",
      tooltip: "loss",
      required: true,
      options: [
        "bce_dice",
        "focal_dice",
        "focal_tversky",
        "cross_entropy",
        "cross_entropy_dice",
      ],
    },
  ],
};

describe("trainingConfigSchema", () => {
  it("shows only multiclass losses for a multiclass dataset", () => {
    expect(trainingConfigSchema(schema, "multiclass")?.fields[0].options).toEqual([
      "cross_entropy",
      "cross_entropy_dice",
    ]);
  });

  it("keeps binary losses for a binary dataset", () => {
    expect(trainingConfigSchema(schema, "binary")?.fields[0].options).toEqual([
      "bce_dice",
      "focal_dice",
      "focal_tversky",
    ]);
  });

  it("возвращает параметры legacy при переключении с next-gen2", () => {
    const defaults = { legacy: { "train.loss": "focal_tversky", "train.pretrained": false, "train.max_val_batches_per_epoch": 1000 } };
    const config = configWithField({ "train.pipeline_variant": "next_gen2", "train.loss": "cross_entropy_tversky" }, "train.pipeline_variant", "legacy", defaults);
    expect(config).toEqual({ ...defaults.legacy, "train.pipeline_variant": "legacy" });
  });

  it("не показывает настройки снятого с запуска next-gen", () => {
    expect(trainingConfigFieldVisible("next_gen.normalization", "legacy", "segformer_b0")).toBe(false);
    expect(trainingConfigFieldVisible("train.pretrained", "next_gen2", "smp_segformer_b3")).toBe(false);
  });

  it.each([16, 8, 4])("учитывает серверный batch архитектуры %i при изменении тайла", (baseBatch) => {
    const defaults = { next_gen2: { "train.batch_size": baseBatch, "tile_preparation.tile_size": 512 } };
    const value = configWithField({}, "train.pipeline_variant", "next_gen2", defaults);
    for (const [size, divisor] of [[512, 1], [768, 2], [1024, 4], [1536, 8]]) {
      const changed = configWithField(value, "tile_preparation.tile_size", size, defaults);
      expect(changed["train.batch_size"]).toBe(Math.max(1, baseBatch / divisor));
    }
  });

  it("применяет серверные параметры next-gen2 и убирает несовместимые настройки", () => {
    const defaults = {
      next_gen2: {
        "tile_preparation.tile_size": 512,
        "tile_preparation.stride": 256,
        "tile_preparation.context": 0,
        "train.batch_size": 16,
        "train.loss": "cross_entropy_tversky",
        "train.max_val_batches_per_epoch": null,
      },
    };
    const value = configWithField(
      { "tile_preparation.context": 128, "train.loss": "bce_dice" },
      "train.pipeline_variant", "next_gen2", defaults,
    );
    expect(value).toEqual({ ...defaults.next_gen2, "train.pipeline_variant": "next_gen2" });
    expect(trainingConfigSchema(schema, "binary", "next_gen2")?.fields[0].options).toEqual(["cross_entropy_tversky"]);
    expect(trainingConfigFieldVisible("tile_preparation.context", "next_gen2", "segformer_b0")).toBe(false);
    expect(trainingConfigFieldVisible("train.pos_weight", "next_gen2", "segformer_b0")).toBe(false);
    expect(trainingConfigFieldVisible("tile_preparation.stride", "next_gen2", "segformer_b0")).toBe(false);
    expect(trainingConfigFieldVisible("dataset.val_fraction", "next_gen2", "segformer_b0")).toBe(false);
    for (const size of [512, 768, 1024, 1536]) {
      const changed = configWithField(value, "tile_preparation.tile_size", size);
      expect(changed["tile_preparation.tile_size"]).toBe(size);
      expect(changed["tile_preparation.stride"]).toBe(size / 2);
      expect(changed["tile_preparation.context"]).toBe(0);
      expect(changed["train.batch_size"]).toBe(({512: 16, 768: 8, 1024: 4, 1536: 2} as Record<number, number>)[size]);
    }
    expect(trainingConfigFieldVisible("tile_preparation.tile_size", "next_gen2", "segformer_b0")).toBe(true);
    expect(trainingConfigFieldVisible("tile_preparation.stride", "legacy", "smp_segformer_b0")).toBe(true);
    expect(configWithField({ "train.pipeline_variant": "legacy", "tile_preparation.stride": 256 }, "tile_preparation.tile_size", 768)["tile_preparation.stride"]).toBe(256);
    expect(configWithField(value, "train.pipeline_variant", "legacy")["train.loss"]).toBe("bce_dice");
  });
});
