import language from "./language";
import theme from "./theme";

const mergeEntity = (a, b, name) => (a[name] ? [...b[name], a[name]] : b[name]);
export const getResolvers = (...items) =>
  items.reduce(
    (acc, i) => {
      const resolvers = mergeEntity(i, acc, "resolvers");
      const typeDefs = mergeEntity(i, acc, "typeDefs");
      return { resolvers, typeDefs };
    },
    { resolvers: [], typeDefs: [] },
  );

const { resolvers, typeDefs } = getResolvers(language, theme);
export { resolvers, typeDefs };
