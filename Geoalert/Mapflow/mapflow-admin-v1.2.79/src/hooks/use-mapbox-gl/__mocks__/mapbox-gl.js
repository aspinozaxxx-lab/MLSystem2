export const mockfitBounds = jest.fn();
export const addControl = jest.fn();
export const remove = jest.fn();
export const mockOn = jest.fn((_, callback) => {
  setTimeout(callback, 100);
});
export const resizeMock = jest.fn();
export const setStyleMock = jest.fn();

export const getCenterMock = jest.fn(function () {
  return this.center;
});
export const getZoomMock = jest.fn(function () {
  return this.zoom;
});

export const getCanvasMock = jest.fn(() => ({
  clientWidth: 500,
  clientHeight: 500,
}));
export const getBoundsMock = jest.fn(() => ({
  toArray() {
    return [];
  },
}));
export const getSourceMock = jest.fn(function (sourceId) {
  return this.sources[sourceId];
});
export const setSourceDataMock = jest.fn();
export const addSourceMock = jest.fn(function (sourceId, source) {
  this.sources[sourceId] = { ...source, setData: setSourceDataMock };
  return this;
});
export const removeSourceMock = jest.fn(function (sourceId) {
  const { [sourceId]: _, ...rest } = this.sources;
  this.sources = rest;
  return this;
});
export const addLayerMock = jest.fn(function (layer) {
  this.layers.push(layer);
  return this;
});
export const removeLayerMock = jest.fn(function (layerId) {
  this.layers.filter(({ id }) => id !== layerId);
  return this;
});

export const getStyleMock = jest.fn(function () {
  return { layers: [], sources: {} };
});

export const mock = jest.fn().mockImplementation((options) => {
  const mapObject = {
    sources: {},
    layers: [],
    zoom: options.zoom,
    getStyle: getStyleMock,
    center: options.center,
    setStyle: setStyleMock,
    removeSource: removeSourceMock,
    removeLayer: removeLayerMock,
    addSource: addSourceMock,
    getSource: getSourceMock,
    getCanvas: getCanvasMock,
    fitBounds: mockfitBounds,
    getBounds: getBoundsMock,
    getCenter: getCenterMock,
    addControl: addControl,
    addLayer: addLayerMock,
    getZoom: getZoomMock,
    resize: resizeMock,
    remove: remove,
    once: mockOn,
    on: mockOn,
  };
  getStyleMock.bind(mapObject);
  removeSourceMock.bind(mapObject);
  removeLayerMock.bind(mapObject);
  addSourceMock.bind(mapObject);
  getSourceMock.bind(mapObject);
  getCenterMock.bind(mapObject);
  addLayerMock.bind(mapObject);
  setStyleMock.bind(mapObject);
  getZoomMock.bind(mapObject);
  return mapObject;
});

export default { Map: mock };
