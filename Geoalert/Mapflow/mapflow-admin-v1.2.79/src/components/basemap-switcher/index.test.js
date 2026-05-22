import React from "react";

import { render, cleanup, wait, fireEvent, waitForElement } from "test-utils";
import BasemapSwitcher from ".";
import { StyleNames, Styles, STYLE_BASE_URL } from "constants/mapbox";
import { MapAPIContext } from "pages/processing/map-api-context";

describe("BasemapSwitcher", () => {
  let component, setStyleMock, getStyleMock, onceMock;
  beforeEach(() => {
    setStyleMock = jest.fn();
    onceMock = jest.fn();
    getStyleMock = jest.fn(function () {
      return { layers: [], sources: {} };
    });
    const mapAPI = {
      setStyle: setStyleMock,
      getStyle: getStyleMock,
      once: onceMock,
    };
    component = (
      <MapAPIContext.Provider value={mapAPI}>
        <BasemapSwitcher />
      </MapAPIContext.Provider>
    );
  });
  afterEach(cleanup);

  it("renders without error", async () => {
    render(component);
    await wait();
  });

  it("matches snapshot", async () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
    await wait();
  });

  it("should call setStyle", async () => {
    const { getByText, getByTestId } = render(component);

    const streersStyle = `${STYLE_BASE_URL}${Styles[StyleNames.STREETS]}`;

    fireEvent.click(getByTestId("layers-menu"));
    const streetsButton = await waitForElement(() => getByText("Streets"));
    fireEvent.click(streetsButton);
    await wait();
    // expect(setStyleMock).toHaveBeenNthCalledWith(
    //   1,
    //   `${STYLE_BASE_URL}${Styles[StyleNames.DARK]}`,
    // );
    expect(setStyleMock).toHaveBeenNthCalledWith(1, streersStyle);
    expect(getStyleMock).toHaveBeenCalledTimes(1);
    expect(onceMock).toHaveBeenCalledTimes(1);
    await wait();
  });
});
