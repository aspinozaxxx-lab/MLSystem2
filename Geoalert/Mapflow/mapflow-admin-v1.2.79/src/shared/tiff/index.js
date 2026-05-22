
import projLib from "proj4";
export class Tiff {
  // Private contructor
  _image;
   _file;

  // Cached values
   _geoData
   _aoi

  isLoading;
  constructor(file) {
    this.isLoading = true;
    // @ts-expect-error
    return (async () => {
      this._file = file;

      const { fromBlob } = await import("geotiff");
      const tif = await fromBlob(this.file);
      this._image = await tif.getImage();
      this.isLoading = false;

      return this;
    })();
  }

  get image() {
    if (!this._image) throw Error();
    return this._image;
  }

  get file() {
    if (!this._file) throw Error();
    return this._file;
  }

  get geoData() {
    if (this._geoData) return this._geoData;

    const geoKeys = this.image.getGeoKeys();
    const gtmodel = geoKeys.GTModelTypeGeoKey;
    let projection = "";

    if (gtmodel === 1) {
      // Most probably, this value will be EPSG code. Needs testing
      projection = this.makeEPSG(geoKeys);
    } else if (gtmodel === 2) {
      projection =
        dictWGS[geoKeys.GeogCitationGeoKey] || geoKeys.GeogCitationGeoKey;
    } else {
      throw Error("unknown coordinate model");
    }

    const geoData = {
      resolution: this.image.getResolution(),
      bbox: this.image.getBoundingBox(),
      origin: this.image.getOrigin(),
      geoKeys,
      gtmodel,
      projection,
    };

    this._geoData = geoData;
    return this._geoData;
  }

  get aoi() {
    if (this._aoi) return this._aoi;

    // bbox is [x_min, y_min, x_max, y_max]
    const bboxTL = [this.geoData.bbox[0], this.geoData.bbox[3]];
    const bboxTR = [this.geoData.bbox[2], this.geoData.bbox[3]];
    const bboxBR = [this.geoData.bbox[2], this.geoData.bbox[1]];
    const bboxBL = [this.geoData.bbox[0], this.geoData.bbox[1]];

    // making geometry from boundary points
    const aoi = [
      this.toReprojectionPoint(bboxTL),
      this.toReprojectionPoint(bboxBL),
      this.toReprojectionPoint(bboxBR),
      this.toReprojectionPoint(bboxTR),
      this.toReprojectionPoint(bboxTL),
    ];

    this._aoi = aoi;
    return this._aoi;
  }

  get info() {
    return {
      width: this.image.getWidth(),
      height: this.image.getHeight(),
      tileWidth: this.image.getTileWidth(),
      tileHeight: this.image.getTileHeight(),
      samplesPerPixel: this.image.getSamplesPerPixel(),
    };
  }

  toReprojectionPoint(coords) {
    // this works if the projection is understandable by Proj4js. If not, we will reject the file
    const src_crs = projLib.Proj(this.geoData.projection);
    // we always want the destination projection to be WGS84 as it is the geojson standard
    const dst_crs = projLib.Proj("WGS84");
    const point = projLib.toPoint(coords);
    // reprojection points to lat-lon
    const latlonPoint = projLib.transform(src_crs, dst_crs, point);
    return [latlonPoint.x, latlonPoint.y];
  }

  makeEPSG(keys) {
    const projKey = String(keys.ProjectedCSTypeGeoKey);
    // UTM
    if (projKey.length === 5) {
      const prefix = projKey.slice(0, 3);
      const zone = projKey.slice(3);
      if (zone >= "01" && zone <= "60") {
        // South Zones
        if (prefix === "327")
          return `+proj=utm +zone=${zone} +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs`;

        // North Zones
        if (prefix === "326")
          return `+proj=utm +zone=${zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs `;
      }
    }
    return "EPSG:" + projKey;
  }

  async getBase64URL() {
    // Prepeare
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    if (!context) throw Error("Canvas(2d) not exist in browser");

    const { Pool } = await import("geotiff");
    const pool = new Pool();
    const input = (await this.image.readRGB({ pool })) ;

    const { width, height } = this.info;
    // WRITING
    const imageData = new ImageData(width, height);
    const pixels = imageData.data;

    let j = 0;
    for (let i = 0; i < input.length; i += 3) {
      pixels[j] = input[i]; // R
      pixels[j + 1] = input[i + 1]; // G
      pixels[j + 2] = input[i + 2]; // B
      pixels[j + 3] = 255; // A
      j += 4;
    }

    // GETTING BASE64URL
    context.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  }

  get geometry() {
    return { type: "Polygon", coordinates: [this.aoi] };
  }

  get featureColleacton() {
    return {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: {},
          geometry: this.geometry,
        },
      ],
    };
  }
}

const dictWGS = { "WGS 84": "WGS84" };
