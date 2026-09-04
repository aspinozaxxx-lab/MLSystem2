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
});
