import React from "react";
import { Trans } from "@lingui/macro";
import { convertArea } from "@turf/helpers";

const unitTranslates = {
  meters: "{a} m<0>2</0>",
  kilometers: "{a} km<0>2</0>",
};

const units = ["meters", "kilometers"];
const k = 1000000;
export function formatArea(area, fixed = 2) {
  if (area === 0) return [0, "meters"];
  const fx = fixed > 0 ? fixed : 0;
  const i = Math.floor(Math.log(area) / Math.log(k));
  const to = units[i];
  const result = convertArea(area, "meters", to).toFixed(fx);
  return [result, to];
}

function cutTailZeroes(z) {
  const [head, tail] = z.split(".");
  if (parseInt(tail) === 0) return head;
  return z;
}

function FormatArea({ area, demicals = 2, cutZeros = false }) {
  const [value, u] = formatArea(area, demicals);
  const id = unitTranslates[u];
  const a = cutZeros ? cutTailZeroes(value.toString()) : value;
  return <Trans id={id} values={{ a }} components={[<sup />]} />;
}

export default React.memo(FormatArea);
