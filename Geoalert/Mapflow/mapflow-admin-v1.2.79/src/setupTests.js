// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom

import "@testing-library/jest-dom/extend-expect";
// required for react-hook-form
import "mutationobserver-shim";

import "jest-extended";
import { setConsole } from "@tanstack/react-query";

import mapboxGl from "mapbox-gl";
global.mapboxgl = mapboxGl;
jest.mock("mapbox-gl");
jest.mock("map-api-context");

// use LanguageProvider mocks
jest.mock("providers/LanguageProvider.js");

// suspress react-query errors
setConsole({ error: () => {} });

// required for react-hook-form
global.MutationObserver = window.MutationObserver;

// required for mock popper functions
jest.mock("popper.js", () => {
  const PopperJS = jest.requireActual("popper.js");

  return class {
    static placements = PopperJS.placements;

    constructor() {
      return {
        destroy: () => {},
        scheduleUpdate: () => {},
      };
    }
  };
});
