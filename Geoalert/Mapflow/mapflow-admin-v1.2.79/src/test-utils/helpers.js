import { Classes } from "@blueprintjs/core";

export const getSpinnerClass = () => `.${Classes.SPINNER}`;
export const getSpinner = () => document.querySelector(getSpinnerClass());
export const mockGetBoundingClientRect = ({ width = 520, height = 220 } = {}) =>
  (Element.prototype.getBoundingClientRect = jest.fn(() => {
    return { width, height };
  }));

export function mockOffsetSize(width, height) {
  Object.defineProperty(HTMLElement.prototype, "offsetHeight", {
    configurable: true,
    value: height,
  });
  Object.defineProperty(HTMLElement.prototype, "offsetWidth", {
    configurable: true,
    value: width,
  });
}
