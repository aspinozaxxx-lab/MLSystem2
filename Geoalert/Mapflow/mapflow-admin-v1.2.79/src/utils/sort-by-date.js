export const sortByDate = (a, b, desc = true) =>
  new Date(a.updated) < new Date(b.updated) ? desc : !desc;
