import { Button, ButtonGroup } from "@blueprintjs/core";
import React, { useEffect, useState } from "react";
import { t } from "@lingui/macro";
import { useMutation } from "@tanstack/react-query";
import { gql, useApolloClient } from "@apollo/client";

import { useDrawAPI } from "components/mapbox-draw-control";
import { useParams } from "react-router-dom";
import { useAppToaster } from "hooks/use-app-toaster";
import { ToastTemplates } from "constants/toasts";

export const CREATE_AOIS_FROM_GEOMETRY = gql`
  mutation createAoisFromGeometry($data: CreateAoisFromGeometryInput!) {
    createAoisFromGeometry(data: $data) {
      count
    }
  }
`;

export const createAoisFromGeometry = (client) => async (variables) => {
  const result = await client.mutate({
    mutation: CREATE_AOIS_FROM_GEOMETRY,
    fetchPolicy: "no-cache",
    variables: { data: variables },
  });

  return result?.data?.createAoisFromGeometry;
};

function getDiffRight(from, to) {
  return to.filter(({ id }) => !from.some((item) => item.id === id));
}

function DrawRectangleButton({ mapAPI, isGeotiffDataProvider }) {
  const showToast = useAppToaster();

  const { processingId } = useParams();

  const [features, setFeatures] = useState(null);

  const { drawAPI, drawRectangle } = useDrawAPI();

  const [drawing, setDrawing] = useState(false);

  useEffect(() => {
    if (!mapAPI || !drawAPI) return;

    const handleDrawCreated = () => {
      if (!drawAPI) return;

      setDrawing(false);
      setFeatures(drawAPI.getAll().features);
    };

    const handleDrawUpdate = () => {
      if (!drawAPI) return;
      setFeatures(drawAPI.getAll().features);
    };
    mapAPI.on("draw.create", handleDrawCreated);
    mapAPI.on("draw.update", handleDrawUpdate);
  }, [mapAPI, drawAPI]);

  const onCancel = () => {
    setDrawing(false);
  };

  const creteDrawHandler = (drawFn, options) => () => {
    setDrawing(true);

    drawAPI.deleteAll();

    const updateOptions = options || {};
    const drawOptions = {
      onCancel,
      areaLimit: 101_000_000,
      ...updateOptions,
    };

    drawFn(drawOptions);
  };

  const handleCancelDrawing = () => {
    if (!mapAPI || !drawAPI) {
      return;
    }

    try {
      const toDelete = getDiffRight(features ?? [], drawAPI.getAll().features);
      toDelete.forEach(({ id }) => drawAPI.delete(id));
    } catch (error) {
      console.log(error);
    } finally {
      setDrawing(false);
      drawAPI.changeMode("simple_select");
      mapAPI.getCanvas().style.cursor = "default";
    }
  };

  const handleClearAll = () => {
    if (!drawAPI) return;

    drawAPI.deleteAll();
    setFeatures(null);
  };

  const client = useApolloClient();
  const mutation = useMutation(createAoisFromGeometry(client), {
    refetchQueries: ["getAois"],
    onSuccess: ({ data }) => {
      showToast(getSuccessToast(createAoisFromGeometry.count));
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast());
    },
  });

  const handleSaveFeatures = () => {
    mutation.mutate({
      processingId: processingId,
      geometry: JSON.stringify(features[0].geometry),
    });

    drawAPI.deleteAll();
    setFeatures(null);
  };

  return (
    <div className="draw-rectangle-button">
      <ButtonGroup large vertical>
        <Button
          disabled={isGeotiffDataProvider}
          icon={drawing ? "cross" : "polygon-filter"}
          onClick={
            drawing ? handleCancelDrawing : creteDrawHandler(drawRectangle)
          }
        />
        <Button
          icon="floppy-disk"
          disabled={!features}
          loading={mutation.isLoading}
          onClick={handleSaveFeatures}
        />
        <Button icon="trash" disabled={!features} onClick={handleClearAll} />
      </ButtonGroup>
    </div>
  );
}

export default DrawRectangleButton;

const getSuccessToast = (count) => ({
  message: t`Successfully created features`,
  template: ToastTemplates.SUCCESS,
});

const getErrorToast = () => ({
  message: t`Error while created aoi's`,
  template: ToastTemplates.ERROR,
});
