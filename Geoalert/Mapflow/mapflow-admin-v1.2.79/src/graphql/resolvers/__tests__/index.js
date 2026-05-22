import { getResolvers } from "../index";

describe("getResolvers", () => {
  it("should return empty without resolvers", () => {
    const expected = { resolvers: [], typeDefs: [] };
    expect(getResolvers()).toEqual(expected);
    expect(getResolvers({})).toEqual(expected);
    expect(getResolvers({}, {})).toEqual(expected);
  });

  it("should reduce resolvers", () => {
    const fooResolvers = {
      resolvers: { Query: { getFoo: () => "foo" } },
      typeDefs: "fooDefs",
    };
    const barResolvers = {
      resolvers: { Mutation: { addBar: () => "bar" } },
      typeDefs: "barDefs",
    };
    const expected = {
      resolvers: [fooResolvers.resolvers, barResolvers.resolvers],
      typeDefs: [fooResolvers.typeDefs, barResolvers.typeDefs],
    };
    const result = getResolvers(fooResolvers, barResolvers);
    expect(result).toEqual(expected);
  });
});
