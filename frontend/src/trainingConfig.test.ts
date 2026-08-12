import { describe, expect, it } from "vitest";

import { trainingConfigSchema } from "./App";
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
});
