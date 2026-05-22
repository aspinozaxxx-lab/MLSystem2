// craco.config.js
const inliner = require("@vgrid/sass-inline-svg");

module.exports = {
  style: {
    sass: {
      loaderOptions: {
        // issue https://github.com/palantir/blueprint/issues/5334
        // functions from blueprint : https://github.com/palantir/blueprint/blob/develop/packages/core/scripts/sass-custom-functions.js
        sassOptions: {
          functions: {
            "svg-icon($path, $selectors: null)": inliner("resources/icons", {
              // run through SVGO first
              optimize: true,
              // minimal "uri" encoding is smaller than base64
              encodingFormat: "uri",
            }),
          },
        },
      },
    },
  },
};
