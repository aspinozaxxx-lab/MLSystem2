import { describe, expect, it } from "vitest";

import {
  configWithField,
  trainingConfigFieldVisible,
  trainingConfigSchema,
} from "./App";
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

  it("clears the validation limit when next-gen is selected", () => {
    expect(
      configWithField(
        { "train.max_val_batches_per_epoch": 1000 },
        "train.pipeline_variant",
        "next_gen",
      ),
    ).toEqual({
      "train.pipeline_variant": "next_gen",
      "train.max_val_batches_per_epoch": null,
    });
  });

  it("shows pretrained only for HF B0 next-gen", () => {
    expect(trainingConfigFieldVisible("train.pretrained", "next_gen", "segformer_b0")).toBe(true);
    expect(trainingConfigFieldVisible("train.pretrained", "legacy", "segformer_b0")).toBe(false);
    expect(trainingConfigFieldVisible("train.pretrained", "next_gen", "smp_segformer_b0")).toBe(false);
  });

  it("применяет серверные параметры next-gen2 и убирает несовместимые настройки", () => {
    const defaults = {
      next_gen2: {
        "tile_preparation.tile_size": 512,
        "tile_preparation.stride": 256,
        "tile_preparation.context": 0,
        "train.loss": "cross_entropy",
        "train.max_val_batches_per_epoch": null,
      },
    };
    const value = configWithField(
      { "tile_preparation.context": 128, "train.loss": "bce_dice" },
      "train.pipeline_variant", "next_gen2", defaults,
    );
    expect(value).toEqual({ ...defaults.next_gen2, "train.pipeline_variant": "next_gen2" });
    expect(trainingConfigSchema(schema, "binary", "next_gen2")?.fields[0].options).toEqual(["cross_entropy"]);
    expect(trainingConfigFieldVisible("tile_preparation.context", "next_gen2", "segformer_b0")).toBe(false);
    expect(trainingConfigFieldVisible("train.pos_weight", "next_gen2", "segformer_b0")).toBe(false);
    expect(trainingConfigFieldVisible("tile_preparation.stride", "next_gen2", "segformer_b0")).toBe(true);
    expect(configWithField(value, "train.pipeline_variant", "next_gen")["train.loss"]).toBe("bce_dice");
  });
});
