import React, { memo, useMemo, useState } from "react";
import { areEqual } from "react-window";
import { Button, Popover, Icon } from "@blueprintjs/core";
import { useQuery } from "@tanstack/react-query";
import { IconNames } from "@blueprintjs/icons";
import pathOr from "ramda/src/pathOr";

import { FormatArea, StatusTag } from "components";
import { Statuses } from "constants/common";
import { useTheme } from "hooks/use-theme";
import { ActionsMenu } from "./actions-menu";
import { LoadingItem } from "./loading-item";
import AoiMessagesDialog from "./aoi-messages-dialog";

let AoiListItem = function AoiListItem({ id, isGeotiffDataProvider }) {
  const [errorModalOpen, setErrorModalOpen] = useState(false);
  const [modalMessages, setModalMessages] = useState([]);

  const handleClose = () => {
    setModalMessages([]);
    setErrorModalOpen(false);
  };

  const { data, isLoading } = useQuery({
    queryKey: ["aoi", id],
    queryFn: () => null,
  });
  //], { variables: { id } });
  const area = pathOr(0, ["area"])(data);
  const percentCompleted = pathOr(0, ["progress", "percentCompleted"])(data);
  const aoiStatus = pathOr(Statuses.UNPROCESSED, ["progress", "status"])(data);
  const messages = pathOr(null, ["messages"])(data);
  const { themeClassName } = useTheme();

  const hasData = data != null;

  return useMemo(
    () =>
      isLoading || !hasData ? (
        <LoadingItem />
      ) : (
        <>
          <div className="aoi-list-item">
            <div className="aoi-item-coll area">
              <FormatArea cutZeros area={area} />
            </div>
            <div className="aoi-item-coll percent-completed">{`${percentCompleted} %`}</div>
            <div className="aoi-item-coll status">
              <StatusTag statusCode={aoiStatus} />
                {messages && messages.length > 0 && (
                <Icon
                  icon="info-sign"
                  className="aoi-messages-icon"
                  onClick={(event) => {
                    event.stopPropagation();
                    setModalMessages(messages);
                    if (messages.length > 0) {
                      setErrorModalOpen(true);
                    }
                  }}
                />
              )}

              <AoiMessagesDialog messages={messages} />
            </div>
            <div className="aoi-item-coll actions">
              <Popover
                className={themeClassName}
                
                content={
                  <ActionsMenu
                    isGeotiffDataProvider={isGeotiffDataProvider}
                    aoiId={id}
                    aoiStatus={aoiStatus}
                  />
                }
              >
                <Button small minimal icon={IconNames.MORE} />
              </Popover>
            </div>
          </div>
          <AoiMessagesDialog
            handleClose={handleClose}
            isOpen={errorModalOpen && modalMessages.length > 0}
            messages={modalMessages}
          />

        </>
      ),
    [
      id,
      area,
      percentCompleted,
      aoiStatus,
      themeClassName,
      isLoading,
      hasData,
      messages,
      errorModalOpen,
      modalMessages,
    ],
  );
};
AoiListItem = memo(AoiListItem, areEqual);

export { AoiListItem };
