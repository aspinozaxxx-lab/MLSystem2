export function refreshTiles(params) {}

export class TileCacheRefresher {
  constructor(sourceId) {
    this.sourceId = sourceId;
  }

  experimentalSoftRefresh(api) {
    if (!api.getSource(this.sourceId)) return;
    const sourceCache = api.style.sourceCaches[this.sourceId];
    console.log({ sourceCache });
    for (const id in sourceCache._tiles) {
      sourceCache._tiles[id].expirationTime = Date.now() - 1;
      sourceCache._reloadTile(id, "reloading");
    }
    sourceCache._cache.reset();
    api.triggerRepaint();
  }

  async hardRefresh(api) {
    if (!api.getSource(this.sourceId)) return;
    if (!api.getSource(this.sourceId).tiles) return;

    const [tilesUrl] = api.getSource(this.sourceId).tiles;

    api.getSource(this.sourceId).tiles = [`${tilesUrl}&dt=${Date.now()}`];

    // Clear the tile cache for a particular source
    api.style.sourceCaches[this.sourceId].clearTiles();

    // Load the new tiles for the current viewport (api.transform -> viewport)
    api.style.sourceCaches[this.sourceId].update(api.transform);

    // Force a repaint, so that the api will be repainted without movements
    api.triggerRepaint();
  }
}
