export function getPersistLayers({ prefix, styleObject }) {
  const sources = Object.entries(styleObject.sources).reduce(
    (acc, [id, source]) => {
      if (id.startsWith(prefix)) return [...acc, [id, source]];
      return acc;
    },
    [],
  );
  const layers = styleObject.layers.filter(({ id }) => id.startsWith(prefix));

  return { sources, layers };
}
