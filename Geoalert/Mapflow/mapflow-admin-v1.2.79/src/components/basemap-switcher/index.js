import React, { useEffect, useState } from "react";
import {
  Popover,
  PopoverInteractionKind,
  Position,
  Button,
  Menu,
  MenuItem,
  MenuDivider,
  Intent,
} from "@blueprintjs/core";
import { t, Trans } from "@lingui/macro";
import { MapAPIContext } from "pages/processing/map-api-context";
import { setTestId } from "test-utils/set-testid";

import { addRasterLayer, removeRasterLayer } from "shared/geo/layers";
import { NSPDRASTERMAPURL } from "containers/aoi-geo-json-layer/constants";
import { BASEMAP_URL } from "constants/envs";

const overlaysData = [
  {
    text: t`NSPD basemap`,
    url: BASEMAP_URL,
  },
];

function BasemapSwitcher({ dataProvider }) {
  const [overlays, setOverlays] = useState(overlaysData);
  const [name, setName] = React.useState("NSPD basemap");

  const mapAPI = React.useContext(MapAPIContext);
  const updateStyle = (text, url) => {
    if (name === text) return;

    removeRasterLayer(mapAPI, NSPDRASTERMAPURL);
    setName(text);
    addRasterLayer(mapAPI, {
      url,
      layerId: NSPDRASTERMAPURL,
      beforeId: "country-label",
    });
  };

  useEffect(() => {
    if (
      dataProvider &&
      dataProvider.previewUrl &&
      !overlays.some(
        (ov) =>
          (ov.text === dataProvider.displayName ||
            ov.text === dataProvider.name) &&
          ov.url === dataProvider.previewUrl,
      )
    ) {
      setOverlays([
        ...overlays,
        {
          text: dataProvider.displayName || dataProvider.name,
          url: dataProvider.previewUrl,
        },
      ]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <Popover
        interactionKind={PopoverInteractionKind.CLICK}
        position={Position.BOTTOM_LEFT}
      >
        <Button elementRef={setTestId`layers-menu`} icon="layers" large />
        <Menu>
          <MenuDivider title={<Trans>Base overlay</Trans>} />
          {overlays.map(({ url, text }, index) => (
            <MenuItem
              key={index}
              text={<Trans id={text} />}
              intent={text === name ? Intent.PRIMARY : Intent.NONE}
              onClick={() => updateStyle(text, url)}
            />
          ))}
        </Menu>
      </Popover>
    </div>
  );
}

export default React.memo(BasemapSwitcher);
