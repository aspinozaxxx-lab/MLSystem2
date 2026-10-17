import Collection from "ol/Collection";
import Feature from "ol/Feature";
import type Geometry from "ol/geom/Geometry";
import Polygon from "ol/geom/Polygon";
import type OLMap from "ol/Map";
import MapBrowserEvent from "ol/MapBrowserEvent";
import Observable from "ol/Observable";
import Modify from "ol/interaction/Modify";
import Select from "ol/interaction/Select";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import View from "ol/View";
import { describe, expect, it, vi } from "vitest";

import { createEditorGestureGuard, updateEditorVertexSelection } from "./datasetEditor";

function dragFixture(startDrag = true) {
  const feature = new Feature<Geometry>(new Polygon([[
    [100, 100], [400, 100], [400, 400], [100, 400], [100, 100],
  ]]));
  const features = new Collection([feature]);
  const view = new View({ projection: "EPSG:3857", resolution: 1, center: [0, 0] });
  const map = Object.assign(new Observable(), {
    getView: () => view,
    isRendered: () => true,
    getPixelFromCoordinate: (coordinate: number[]) => coordinate,
    getCoordinateFromPixel: (pixel: number[]) => pixel,
    forEachFeatureAtPixel: () => undefined,
    render: () => undefined,
  }) as unknown as OLMap;
  const modify = new Modify({ features, deleteCondition: () => false });
  modify.setMap(map);
  const layer = new VectorLayer({ source: new VectorSource({ features }) });
  layer.setMap(map);
  const events: string[] = [];
  modify.on("modifystart", () => { events.push("начало"); });
  modify.on("modifyend", () => { events.push("завершение"); });
  const gesture = createEditorGestureGuard();
  const makeEvent = (type: string, coordinate: number[]) => {
    const pointer = Object.assign(new Event(type), {
      pointerId: 1, pointerType: "mouse", isPrimary: true,
      button: 0, buttons: type === "pointerup" ? 0 : 1,
      clientX: coordinate[0], clientY: coordinate[1],
    }) as PointerEvent;
    const event = new MapBrowserEvent(type, map, pointer, type === "pointerdrag", undefined,
      type === "pointerup" ? [] : [pointer]);
    event.coordinate = [...coordinate];
    event.pixel = [...coordinate];
    return event;
  };
  const send = (type: string, coordinate: number[]) => {
    const event = makeEvent(type, coordinate);
    if (type === "pointerdown") gesture.pointerDown(event.originalEvent);
    if (type === "pointerup") gesture.pointerUp(event.originalEvent);
    modify.handleEvent(event);
    return event.originalEvent;
  };
  const point = () => (feature.getGeometry() as Polygon).getCoordinates()[0][0];
  const dispose = () => {
    layer.setMap(null);
    layer.dispose();
    features.clear();
    modify.getOverlay().setMap(null);
    modify.dispose();
  };
  if (startDrag) {
    send("pointermove", [100, 100]);
    send("pointerdown", [100, 100]);
    send("pointerdrag", [150, 160]);
  }
  return { feature, features, modify, layer, events, send, makeEvent, point, dispose, gesture };
}

describe("удержание вершины при перетаскивании", () => {
  it("воспроизводит снятие выбора запоздалым singleclick посреди следующего жеста", () => {
    const fixture = dragFixture();
    const select = new Select({ features: fixture.features, style: null });
    try {
      select.handleEvent(fixture.makeEvent("singleclick", [100, 100]));
      expect(fixture.features.getLength()).toBe(0);
      expect(fixture.send("pointerdrag", [230, 250]).buttons).toBe(1);
      expect(fixture.point()).toEqual([150, 160]);
      expect(fixture.events).toEqual(["начало"]);
    } finally {
      select.dispose();
      fixture.dispose();
    }
  });

  it("не даёт старому singleclick прервать удержание и снять выбор после быстрого следующего жеста", () => {
    const fixture = dragFixture(false);
    const select = new Select({ features: fixture.features, style: null,
      condition: fixture.gesture.allowsSelection });
    try {
      fixture.send("pointermove", [100, 100]);
      fixture.send("pointerdown", [100, 100]);
      fixture.send("pointerup", [100, 100]);
      const lateClick = fixture.makeEvent("singleclick", [100, 100]);
      fixture.gesture.registerClick(lateClick);
      fixture.send("pointerdown", [100, 100]);
      fixture.send("pointerdrag", [150, 160]);
      select.handleEvent(lateClick);
      expect(fixture.features.getLength()).toBe(1);
      expect(fixture.send("pointerdrag", [230, 250]).buttons).toBe(1);
      expect(fixture.point()).toEqual([230, 250]);
      expect(fixture.events).toEqual(["начало"]);
      fixture.send("pointerup", [230, 250]);
      select.handleEvent(lateClick);
      expect(fixture.features.getLength()).toBe(1);
      const currentClick = fixture.makeEvent("singleclick", [100, 100]);
      fixture.gesture.registerClick(currentClick);
      select.handleEvent(currentClick);
      expect(fixture.features.getLength()).toBe(0);
      expect(fixture.events).toEqual(["начало", "завершение"]);
    } finally {
      select.dispose();
      fixture.dispose();
    }
  });

  it("не принимает отпускание другого указателя за завершение жеста и учитывает все указатели", () => {
    const gesture = createEditorGestureGuard();
    const click = { type: "singleclick", originalEvent: {} };
    gesture.pointerDown({ pointerId: 1 });
    gesture.registerClick(click);
    gesture.pointerDown({ pointerId: 2 });
    gesture.pointerUp({ pointerId: 99 });
    gesture.pointerUp({ pointerId: 1 });
    expect(gesture.allowsSelection(click)).toBe(false);
    gesture.pointerUp({ pointerId: 2 });
    expect(gesture.allowsSelection(click)).toBe(true);
    gesture.pointerDown({ pointerId: 3 });
    gesture.pointerUp({ pointerId: 3 });
    expect(gesture.allowsSelection(click)).toBe(false);
    const next = { type: "singleclick", originalEvent: {} };
    gesture.registerClick(next);
    expect(gesture.allowsSelection(next)).toBe(true);
  });

  it("воспроизводит потерю захвата от визуального feature.changed без отпускания мыши и без Snap", () => {
    const fixture = dragFixture();
    try {
      expect(fixture.point()).toEqual([150, 160]);
      fixture.feature.changed();
      const pointer = fixture.send("pointerdrag", [230, 250]);
      expect(pointer.buttons).toBe(1);
      expect(fixture.events).toEqual(["начало"]);
      expect(fixture.point()).toEqual([150, 160]);
      fixture.send("pointerup", [230, 250]);
      expect(fixture.events).toEqual(["начало", "завершение"]);
    } finally {
      fixture.dispose();
    }
  });

  it("обновляет подсветку во время удержания кнопки без сброса текущей вершины", () => {
    const fixture = dragFixture();
    const selectionRef = { current: [fixture.feature] };
    try {
      const revision = fixture.feature.getRevision();
      const layerRevision = fixture.layer.getRevision();
      updateEditorVertexSelection(selectionRef, [...selectionRef.current], fixture.layer);
      expect(fixture.feature.getRevision()).toBe(revision);
      expect(fixture.layer.getRevision()).toBeGreaterThan(layerRevision);
      expect(fixture.send("pointerdrag", [230, 250]).buttons).toBe(1);
      expect(fixture.point()).toEqual([230, 250]);
      updateEditorVertexSelection(selectionRef, [], fixture.layer);
      expect(selectionRef.current).toEqual([]);
      expect(fixture.send("pointerdrag", [260, 270]).buttons).toBe(1);
      expect(fixture.point()).toEqual([260, 270]);
      expect(fixture.events).toEqual(["начало"]);
      fixture.send("pointerup", [260, 270]);
      expect(fixture.events).toEqual(["начало", "завершение"]);
    } finally {
      fixture.dispose();
    }
  });

  it("сохраняет непрерывный жест при запоздалых обновлениях после автосохранения и singleclick", () => {
    vi.useFakeTimers();
    const fixture = dragFixture();
    const selectionRef = { current: [fixture.feature] };
    const delayedRefresh = () => updateEditorVertexSelection(selectionRef,
      selectionRef.current.filter(() => true), fixture.layer);
    try {
      setTimeout(delayedRefresh, 250);
      setTimeout(delayedRefresh, 1000);
      setTimeout(delayedRefresh, 2200);
      for (let step = 1; step <= 40; step += 1) {
        vi.advanceTimersByTime(100);
        const point = [150 + step * 3, 160 + step * 2];
        expect(fixture.send("pointerdrag", point).buttons).toBe(1);
        expect(fixture.point()).toEqual(point);
        expect(fixture.events).toEqual(["начало"]);
      }
      fixture.send("pointerup", [270, 240]);
      expect(fixture.events).toEqual(["начало", "завершение"]);
    } finally {
      fixture.dispose();
      vi.useRealTimers();
    }
  });
});
