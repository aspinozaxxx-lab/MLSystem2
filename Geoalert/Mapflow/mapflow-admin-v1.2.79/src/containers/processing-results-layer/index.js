import React, { useEffect } from "react";

import { SourceIds, LayerStyles, LayerIds } from "./constants";
import { TileCacheRefresher } from "./refresh-tiles";

const tilesResresher = new TileCacheRefresher(SourceIds.PROCESSED);

function ProcessingResultsLayer({ mapAPI, vectorLayer, completedArea }) {
  const tileJsonUrl = vectorLayer?.tileJsonUrl;

  useEffect(() => {
    if (!tileJsonUrl) {
      return;
    }
    if (completedArea === 0) {
      return;
    }

    if (mapAPI.getSource(SourceIds.PROCESSED)) {
      return;
    }
    addVectorLayer(mapAPI, tileJsonUrl);
  }, [tileJsonUrl, mapAPI, completedArea]);

  useEffect(() => {
    if (completedArea === 0) return;
    tilesResresher.hardRefresh(mapAPI);
  }, [mapAPI, completedArea]);

  return null;
}

function addVectorLayer(mapAPI, url) {
  const sourceId = SourceIds.PROCESSED;
  const source = { type: "vector", url };
  const fillId = LayerIds.PROCESSED;
  const fill = LayerStyles.getProcessed(fillId, sourceId);
  const outlineId = LayerIds.PROCESSED_OUTLINE;
  const outline = LayerStyles.getProcessedOutline(outlineId, sourceId);
  const symbolId = LayerIds.PROCESSED_POINTS;
  const symbol = LayerStyles.getProcessedSymbols(symbolId, source);
  const line = LayerStyles.getProcessedLines(outlineId, sourceId);

  mapAPI
    .addSource(sourceId, source)
    .addLayer(fill)
    .addLayer(outline)
    .addLayer(line)
    .addLayer(symbol);
}

export default React.memo(ProcessingResultsLayer);
