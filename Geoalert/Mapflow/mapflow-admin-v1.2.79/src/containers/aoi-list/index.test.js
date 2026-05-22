import React from "react";

import {
  render,
  cleanup,
  wait,
  waitForElement,
  waitForElementToBeRemoved,
  fireEvent,
} from "test-utils";
import { mockOffsetSize, mockGetBoundingClientRect } from "test-utils/helpers";

import AoiList, { GET_AOIS, BATCH_SIZE } from ".";
import { AppToaster } from "toaster";

import aoisMock from "aoi-list-mock";
import { InMemoryCache } from "apollo-cache-inmemory";
import { cacheRedirects } from "graphql/cache-redirects";
jest.mock("aoi-list-mock");
jest.mock("./use-aoi-list-poller");

describe("AoiList", () => {
  const projectId = "1";
  const processingId = "1";

  function getRouterOptions() {
    const path = `/projects/:projectId/processings/:processingId`;
    const processingRoute = `/projects/${projectId}/processings/${processingId}`;
    return { route: processingRoute, path };
  }

  function getMocks(aois = aoisMock) {
    return [
      {
        request: {
          query: GET_AOIS,
          variables: {
            processingId: processingId,
            offset: 0,
            limit: BATCH_SIZE,
            sort: [],
            statuses: null,
            geometry: null,
          },
        },
        result: {
          data: {
            aois: {
              __typename: "AoiList",
              hasMore: false,
              aois: aois,
            },
          },
        },
      },
    ];
  }

  beforeEach(() => {
    jest.resetAllMocks();
    mockGetBoundingClientRect();
    mockOffsetSize(400, 1000);
    AppToaster.clear();
  });
  afterEach(cleanup);

  it("renders without error", async () => {
    render(<AoiList />, {
      withApollo: { mocks: getMocks([]) },
      withRouter: getRouterOptions(),
    });
    await wait();
  });

  it("matches snapshot", async () => {
    const { asFragment } = render(<AoiList />, {
      withApollo: { mocks: getMocks([]) },
      withRouter: getRouterOptions(),
    });
    expect(asFragment()).toMatchSnapshot();
    await wait();
  });

  it("should show empty message", async () => {
    const onUploadMock = jest.fn();
    const { getByText, getByTestId } = render(
      <AoiList onUpload={onUploadMock} />,
      {
        withApollo: { mocks: getMocks([]) },
        withRouter: getRouterOptions(),
      },
    );

    await waitForElementToBeRemoved(() => getByText(/Fetching aoi's/i));
    await waitForElement(() => getByText(/You haven’t created any AOI yet/i));
    const uploadAoiButton = await waitForElement(() =>
      getByTestId("empty-message-upload-button"),
    );
    fireEvent.click(uploadAoiButton);
    expect(onUploadMock).toHaveBeenCalledTimes(1);
  });

  it("should show aois", async () => {
    const cache = new InMemoryCache({ cacheRedirects, addTypename: false });
    const onUploadMock = jest.fn();
    const { getByText, getByTestId, asFragment } = render(
      <AoiList onUpload={onUploadMock} />,
      {
        withApollo: { mocks: getMocks(aoisMock), cache },
        withRouter: getRouterOptions(),
      },
    );

    await waitForElementToBeRemoved(() => getByText(/Fetching aoi's/i));
    expect(asFragment()).toMatchSnapshot();
  });

  // it("should run processing", async () => {
  //   const [aoi] = aois;
  //   const { id, area } = aoi;

  //   const statusUnprocessed = T[Statuses.UNPROCESSED]["id"];
  //   const mocks = getMocks(
  //     {
  //       runProcessing: 1,
  //       aois: aois,
  //       selectedIds: [id],
  //     },
  //     [
  //       {
  //         request: {
  //           query: AOI_LIST_SORT,
  //         },
  //         result: getAoiListSortCacheObject(),
  //       },
  //       {
  //         request: {
  //           query: AOI_LIST_FILTER,
  //         },
  //         result: getAoiListFilterCacheObject(),
  //       },
  //     ],
  //   );
  //   const {
  //     getByTestId,
  //     getByText,
  //     asFragment,
  //     getAllByText,
  //   } = renderComponent(<AoiList />, { mocks });

  //   await waitForElementToBeRemoved(() => getByText("Fetching aoi's"));

  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));
  //   expect(getAllByText(statusUnprocessed)).toHaveLength(aois.length);
  //   expect(asFragment()).toMatchSnapshot();

  //   const firstRow = getByText(`${area} m`);
  //   const firstRowCheckbox = within(firstRow.parentNode).getByTestId(
  //     "aoi-checkbox",
  //   );
  //   expect(firstRowCheckbox.checked).toBeFalse();
  //   fireEvent.click(firstRowCheckbox);
  //   expect(firstRowCheckbox.checked).toBeTrue();
  //   expect(getByTestId("select-all").indeterminate).toBeTrue();
  //   await waitForElement(() => getByText("1 row selected"));

  //   fireEvent.click(getByTestId("run-processing"));
  //   await waitForElementToBeRemoved(() => getByText("1 row selected"));
  //   await waitForElement(() => getByText("Aoi List"));
  //   await waitForElement(() => getByText("Successfully started 1 area!"));
  //   expect(getByTestId("select-all").indeterminate).toBeFalse();
  //   expect(getByTestId("select-all").checked).toBeFalse();
  //   expect(firstRowCheckbox.checked).toBeFalse();
  // }, 10000);

  // it("should sort", async () => {
  //   const cache = new InMemoryCache({ addTypename: false });
  //   cache.writeQuery({
  //     query: AOI_LIST_SORT,
  //     data: getAoiListSortCacheObject(),
  //   });
  //   const processingId = "1";
  //   const statusUnprocessed = T[Statuses.UNPROCESSED]["id"];
  //   const mocks = [
  //     {
  //       request: {
  //         query: AOI_LIST_SORT,
  //       },
  //       result: getAoiListSortCacheObject(),
  //     },
  //     // {
  //     //   request: {
  //     //     query: AOI_LIST_FILTER,
  //     //   },
  //     //   result: getAoiListFilterCacheObject(),
  //     // },
  //     {
  //       request: {
  //         query: GET_AOIS,
  //         variables: {
  //           processingId: processingId,
  //           offset: 0,
  //           limit: 80,
  //           statusFilter: null,
  //           geometryFilter: null,
  //           sort: [],
  //         },
  //       },
  //       result: { data: { aois: { aois } } },
  //     },
  //     {
  //       request: {
  //         query: GET_AOIS,
  //         variables: {
  //           processingId: processingId,
  //           offset: 0,
  //           limit: 80,
  //           statusFilter: null,
  //           geometryFilter: null,
  //           sort: [{ field: "area", desc: false }],
  //         },
  //       },
  //       result: { data: { aois: { aois } } },
  //     },
  //     {
  //       request: {
  //         query: GET_AOIS,
  //         variables: {
  //           processingId: processingId,
  //           offset: 0,
  //           limit: 80,
  //           statusFilter: null,
  //           geometryFilter: null,
  //           sort: [{ field: "area", desc: true }],
  //         },
  //       },
  //       result: { data: { aois: { aois } } },
  //     },
  //     {
  //       request: {
  //         query: GET_AOIS,
  //         variables: {
  //           processingId: processingId,
  //           offset: 0,
  //           limit: 80,
  //           statusFilter: null,
  //           geometryFilter: null,
  //           sort: [{ field: "area", desc: false }],
  //         },
  //       },
  //       result: { data: { aois: { aois } } },
  //     },
  //     {
  //       request: {
  //         query: GET_AOIS,
  //         variables: {
  //           processingId: processingId,
  //           offset: 0,
  //           limit: 80,
  //           statusFilter: null,
  //           geometryFilter: null,
  //           sort: [{ field: "status", desc: false }],
  //         },
  //       },
  //       result: { data: { aois: { aois } } },
  //     },
  //     {
  //       request: {
  //         query: GET_AOIS,
  //         variables: {
  //           processingId: processingId,
  //           offset: 0,
  //           limit: 80,
  //           statusFilter: null,
  //           geometryFilter: null,
  //           sort: [{ field: "status", desc: true }],
  //         },
  //       },
  //       result: { data: { aois: { aois } } },
  //     },
  //     {
  //       request: {
  //         query: GET_AOIS,
  //         variables: {
  //           processingId: processingId,
  //           offset: 0,
  //           limit: 80,
  //           statusFilter: null,
  //           geometryFilter: null,
  //           sort: [{ field: "status", desc: false }],
  //         },
  //       },
  //       result: { data: { aois: { aois } } },
  //     },
  //   ];
  //   const { getByText, getAllByText, unmount } = renderComponent(<AoiList />, {
  //     mocks,
  //     cache,
  //   });

  //   const getSortStateFromCache = () => {
  //     const data = cache.readQuery({
  //       query: AOI_LIST_SORT,
  //     });
  //     return extractSortState(data);
  //   };

  //   await waitForElementToBeRemoved(() => getByText("Fetching aoi's"));

  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));
  //   expect(getAllByText(statusUnprocessed)).toHaveLength(aois.length);

  //   const areaHeader = getByText("Area");
  //   fireEvent.click(areaHeader);
  //   await waitForElement(() =>
  //     within(areaHeader.parentNode).getByText(IconNames.CARET_UP),
  //   );
  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));

  //   expect(getSortStateFromCache()).toEqual([{ field: "area", desc: false }]);

  //   fireEvent.click(areaHeader);
  //   await waitForElement(() =>
  //     within(areaHeader.parentNode).getByText(IconNames.CARET_DOWN),
  //   );
  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));

  //   expect(getSortStateFromCache()).toEqual([{ field: "area", desc: true }]);

  //   fireEvent.click(areaHeader);
  //   await waitForElement(() =>
  //     within(areaHeader.parentNode).getByText(IconNames.CARET_UP),
  //   );
  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));

  //   expect(getSortStateFromCache()).toEqual([{ field: "area", desc: false }]);

  //   const statusHeader = getByText("Status");
  //   fireEvent.click(statusHeader);
  //   await waitForElement(() =>
  //     within(statusHeader.parentNode).getByText(IconNames.CARET_UP),
  //   );
  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));

  //   expect(getSortStateFromCache()).toEqual([{ field: "status", desc: false }]);

  //   fireEvent.click(statusHeader);
  //   await waitForElement(() =>
  //     within(statusHeader.parentNode).getByText(IconNames.CARET_DOWN),
  //   );
  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));

  //   expect(getSortStateFromCache()).toEqual([{ field: "status", desc: true }]);

  //   fireEvent.click(statusHeader);
  //   await waitForElement(() =>
  //     within(statusHeader.parentNode).getByText(IconNames.CARET_UP),
  //   );
  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));

  //   expect(getSortStateFromCache()).toEqual([{ field: "status", desc: false }]);

  //   unmount();
  //   await wait();
  // }, 10000);

  // it("should locate aoi", async () => {
  //   const [aoi1, aoi2] = aois;

  //   const processingId = "1";
  //   const statusUnprocessed = T[Statuses.UNPROCESSED]["id"];
  //   const mocks = [
  //     {
  //       request: {
  //         query: AOI_LIST_SORT,
  //       },
  //       result: getAoiListSortCacheObject(),
  //     },
  //     {
  //       request: {
  //         query: AOI_LIST_FILTER,
  //       },
  //       result: getAoiListFilterCacheObject(),
  //     },
  //     {
  //       request: {
  //         query: GET_AOIS,
  //         variables: {
  //           processingId: processingId,
  //           offset: 0,
  //           limit: 80,
  //           sort: [],
  //           statusFilter: null,
  //           geometryFilter: null,
  //         },
  //       },
  //       result: { data: { aois: { aois } } },
  //     },
  //   ];

  //   const fitBoundsMock = jest.fn();
  //   const component = (
  //     <MapAPIContext.Provider value={{ fitBounds: fitBoundsMock }}>
  //       <AoiList />
  //     </MapAPIContext.Provider>
  //   );
  //   const { getByText, getAllByText, getAllByRole } = renderComponent(
  //     component,
  //     {
  //       mocks,
  //     },
  //   );

  //   await waitForElementToBeRemoved(() => getByText("Fetching aoi's"));
  //   for (const { area } of aois)
  //     await waitForElement(() => getByText(`${area} m`));
  //   expect(getAllByText(statusUnprocessed)).toHaveLength(aois.length);

  //   // skip header row
  //   const [, firstRow, secondRow] = getAllByRole("row");

  //   fireEvent.doubleClick(firstRow);
  //   await wait();

  //   let bounds = bbox(JSON.parse(aoi1.geometry));
  //   let expectedArgs = [bounds, { padding: 50 }];
  //   expect(fitBoundsMock).toHaveBeenNthCalledWith(1, ...expectedArgs);

  //   bounds = bbox(JSON.parse(aoi2.geometry));
  //   expectedArgs = [bounds, { padding: 50 }];
  //   fireEvent.contextMenu(secondRow);
  //   const locateMunuItem = await waitForElement(() => getByText("Locate aoi"));
  //   fireEvent.click(locateMunuItem);
  //   expect(fitBoundsMock).toHaveBeenNthCalledWith(2, ...expectedArgs);
  // });

  // TODO add tests - delete aoi, sorting table, select/unselect all aois, last rows loading states
});
