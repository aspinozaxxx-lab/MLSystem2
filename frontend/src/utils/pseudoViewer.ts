export type PseudoProperties = Record<string, unknown>;

export function pseudoClass(value?: PseudoProperties | null) {
  const properties = value ?? {};
  const key = String(properties.object_type_slug || properties.object_type_id || "objects");
  const color = typeof properties.object_type_color === "string" && /^#[0-9a-f]{6}$/i.test(properties.object_type_color)
    ? properties.object_type_color.toUpperCase() : "#22D3EE";
  return { key, color, name: String(properties.object_type_name || properties.object_type_slug || "Объекты") };
}

export function pseudoClasses(features: { properties?: PseudoProperties | null }[]) {
  const groups = new Map<string, ReturnType<typeof pseudoClass> & { count: number }>();
  for (const feature of features) {
    const item = pseudoClass(feature.properties);
    const group = groups.get(item.key);
    if (group) group.count += 1; else groups.set(item.key, { ...item, count: 1 });
  }
  return [...groups.values()].sort((left, right) => left.name.localeCompare(right.name, "ru"));
}
