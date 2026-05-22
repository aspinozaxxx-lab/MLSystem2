export const setTestId = ([id]) => (ref) => {
  if (ref) ref.setAttribute("data-testid", id);
};
