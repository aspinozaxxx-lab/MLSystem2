import React, { useState, useEffect } from "react";
import {
  Tab,
  Tabs,
  Classes,
  ResizeSensor,
  Button,
  Intent,
  Tooltip,
  Position,
  Divider,
} from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { Trans } from "@lingui/macro";
import classnames from "classnames";

import * as routes from "constants/routes";
import { useWindowWidth } from "hooks/use-window-width";
import { setTestId } from "test-utils/set-testid";
import { TabTitle } from "containers/main-navbar/tab-title";
import { Link, generatePath } from "react-router-dom";
import { useActiveTab, useGetRouteParam } from "shared/routing";
import { Tooltip2 } from "@blueprintjs/popover2";
import { useIsAdmin } from "providers/auth/use-user";

const tabWrapperSelector = `.${Classes.TAB_INDICATOR_WRAPPER}`;
function resizeActiveTabSelector() {
  const selectedTabElement = document.querySelector(tabWrapperSelector);
  if (selectedTabElement !== null)
    selectedTabElement.style.setProperty("width", "100%");
}

function MainNavbar({ minified = false }) {
  const [minify, toggleMinify] = useState(minified);

  const isAdmin = useIsAdmin();

  const activeTab = useActiveTab();
  const projectId = useGetRouteParam("projectId");

  const width = useWindowWidth();
  useEffect(() => {
    if (width < 450) toggleMinify(true);
  }, [width]);

  const className = classnames("main-navbar", {
    "main-navbar--minified": minify,
  });

  return (
    <div className={className}>
      <ResizeSensor onResize={resizeActiveTabSelector}>
        <Tabs
          large
          animate
          vertical
          className="tabs"
          id="MainNavTabs"
          key={"vertical"}
          selectedTabId={activeTab}
        >
          <Tab
            className="tab"
            id={routes.PROJECTS}
            title={
              <Tooltip2 content={<Trans id="Projects" />} disabled={!minify}>
                <Link to={routes.PROJECTS}>
                  <TabTitle
                    icon={IconNames.PROJECTS}
                    text={<Trans id="Projects" />}
                  />
                </Link>
              </Tooltip2>
            }
          />

          {projectId && (
            <Tab
              className="tab"
              id={routes.PROJECT_WORKFLOWS}
              title={
                <Tooltip2 content={<Trans id="Workflows" />} disabled={!minify}>
                  <Link
                    to={generatePath(routes.PROJECT_WORKFLOWS, { projectId })}
                  >
                    <TabTitle
                      icon={IconNames.CARET_RIGHT}
                      text={<Trans id="Workflows" />}
                      textIconLeft={IconNames.APPLICATIONS}
                    />
                  </Link>
                </Tooltip2>
              }
            />
          )}

          {projectId && (
            <Tab
              className="tab"
              id={routes.PROJECT_PROCESSINGS}
              title={
                <Tooltip2
                  content={<Trans id="Processings" />}
                  disabled={!minify}
                >
                  <Link
                    to={generatePath(routes.PROJECT_PROCESSINGS, { projectId })}
                  >
                    <TabTitle
                      icon={IconNames.CARET_RIGHT}
                      text={<Trans id="Processings" />}
                      textIconLeft={IconNames.SOCIAL_MEDIA}
                    />
                  </Link>
                </Tooltip2>
              }
            />
          )}

          {projectId && <Divider />}

          {isAdmin && (
            <Tab
              className="tab"
              id={routes.WORKFLOWS}
              title={
                <Tooltip2 content={<Trans id="Workflows" />} disabled={!minify}>
                  <Link to={routes.WORKFLOWS}>
                    <TabTitle
                      icon={IconNames.APPLICATIONS}
                      text={<Trans id="Workflows" />}
                    />
                  </Link>
                </Tooltip2>
              }
            />
          )}

          {isAdmin && (
            <Tab
              className="tab"
              id={routes.DATA_PROVIDERS}
              title={
                <Tooltip2
                  content={<Trans id="Data Providers" />}
                  disabled={!minify}
                >
                  <Link to={routes.DATA_PROVIDERS}>
                    <TabTitle
                      icon={IconNames.APPLICATIONS}
                      text={<Trans id="Data Providers" />}
                    />
                  </Link>
                </Tooltip2>
              }
            />
          )}

          {isAdmin && (
            <Tab
              className="tab"
              id={routes.PROCESSING_STATS}
              title={
                <Tooltip2
                  content={<Trans id="Stats of Processings" />}
                  disabled={!minify}
                >
                  <Link to={routes.PROCESSING_STATS}>
                    <TabTitle
                      icon={IconNames.APPLICATIONS}
                      text={<Trans>Stats of Processings</Trans>}
                    />
                  </Link>
                </Tooltip2>
              }
            />
          )}

          <Tabs.Expander />
          <Tooltip
            className="expand-tooltip"
            intent={Intent.PRIMARY}
            position={Position.RIGHT}
            disabled={!minify}
            content={<Trans id="Expand navbar" />}
          >
            <Button
              fill
              minimal
              intent={Intent.PRIMARY}
              elementRef={setTestId`minify-navbar`}
              icon={minify ? IconNames.ARROW_RIGHT : IconNames.ARROW_LEFT}
              text={!minify && <Trans id="Minify navbar" />}
              onClick={() => toggleMinify(!minify)}
            />
          </Tooltip>
        </Tabs>
      </ResizeSensor>
    </div>
  );
}

export default React.memo(MainNavbar);
