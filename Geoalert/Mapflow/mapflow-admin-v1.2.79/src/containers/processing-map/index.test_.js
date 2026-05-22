import React from "react";
import { InMemoryCache } from "apollo-cache-inmemory";
import { QueryCache } from "@tanstack/react-query";

import geoViewport from "@mapbox/geo-viewport";

import {
  render,
  cleanup,
  wait,
  waitForElement,
  waitForElementToBeRemoved,
} from "test-utils";
import { mockGetBoundingClientRect, getSpinner } from "test-utils/helpers";
import processing from "fixtures/processing";

import { GET_PROCESSING } from "containers/processing-sidebar";
import { defaultMapOptions } from "hooks/use-mapbox-gl";
import { MAPBOX_TOKEN } from "constants/envs";
import { PROCESSING } from "constants/routes";
import ProcessinngMap from ".";

import mapboxGl from "mapbox-gl";

describe("ProcessinngMap", () => {
  let component, bbox, size;
  let mapAPIMock, setMapRefMock;
  beforeEach(() => {
    mapboxGl.Map.mockClear();
    ({ bbox } = processing);
    size = [256, 256];
    setMapRefMock = jest.fn((m) => {
      mapAPIMock = m;
    });
    component = (
      <ProcessinngMap ref={setMapRefMock} mapAPI={mapAPIMock} size={size} />
    );
    mockGetBoundingClientRect({ width: 256, height: 256 });
  });
  afterEach(cleanup);

  it("renders without error", async () => {
    const { center, zoom } = geoViewport.viewport(JSON.parse(bbox), size);
    const accessToken = 

    const processingId = processing["id"];
    const projectId = "1";
    const path = PROCESSING;
    const processingRoute = `/projects/${projectId}/processings/${processingId}`;

    QueryCache.setQueryData(["processing", processingId], { ...processing });

    const options = {
      withRouter: { route: processingRoute, path },
    };
    const { getByText } = render(component, options);

    expect(getSpinner()).toBeInTheDocument();
    await waitForElement(() => getByText("Initialize the map"));
    await waitForElement(() => document.querySelector(`#mapbox-map`));

    expect(mapboxGl.Map).toHaveBeenCalledWith(
      expect.objectContaining({
        ...defaultMapOptions,
        center,
        zoom: zoom - 1 > 0 ? zoom - 1 : 0,
        accessToken,
        container: expect.anything(),
      }),
    );
    // wait for mapbox-gl mock once event resolved
    await new Promise((resolve) => setTimeout(resolve, 101));
    expect(setMapRefMock).toHaveBeenCalledWith(mapAPIMock);
  });
});
