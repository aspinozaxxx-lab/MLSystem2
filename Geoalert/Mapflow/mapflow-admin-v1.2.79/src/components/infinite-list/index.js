import React, { memo, useMemo, useCallback } from "react";
import { FixedSizeList as List, areEqual } from "react-window";
import InfiniteLoader from "react-window-infinite-loader";
import AutoSizer from "react-virtualized-auto-sizer";
import classnames from "classnames";

const createClassName = (index) => (classes) =>
  classnames(classes, { odd: index % 2 !== 0 });

// Render an item or a loading indicator.
const renderListItem = ({
  isItemLoaded,
  ItemComponent,
  LoadingItemComponent,
}) =>
  memo(function InfiniteListItem({ index, style, data }) {
    const className = createClassName(index);
    return (
      <div className={className`infinite-list-item`} style={style}>
        {isItemLoaded(index) ? (
          <ItemComponent id={data[index]["id"]} />
        ) : (
          <LoadingItemComponent />
        )}
      </div>
    );
  }, areEqual);

function InfiniteList({
  items,
  hasNextPage,
  loadNextPage,
  itemComponent,
  loadingItemComponent,
  isNextPageLoadingRef,
  minimumBatchSize = 15,
  overscanCount = 15,
  itemSize = 40,
}) {
  // If there are more items to be loaded then add an extra row to hold a loading indicator.
  const length = useMemo(() => items.length, [items.length]);
  const itemCount = useMemo(() => (hasNextPage ? length + 1 : length), [
    hasNextPage,
    length,
  ]);
  // Every row is loaded except for our loading indicator row.
  const isItemLoaded = useCallback((index) => !hasNextPage || index < length, [
    hasNextPage,
    length,
  ]);

  // Only load 1 page of items at a time.
  // Pass an empty callback to InfiniteLoader in case it asks us to load more than once.
  const loadMoreItems = useCallback(
    (...args) => {
      if (!isNextPageLoadingRef.current) return loadNextPage(...args);
      console.info("skip next page call"); // wait until next page loaded
    },
    [isNextPageLoadingRef, loadNextPage],
  );

  return (
    <div className="infinite-list">
      <InfiniteLoader
        minimumBatchSize={minimumBatchSize}
        isItemLoaded={isItemLoaded}
        itemCount={itemCount}
        loadMoreItems={loadMoreItems}
      >
        {({ onItemsRendered, ref }) => (
          <AutoSizer>
            {({ width, height }) => (
              <List
                ref={ref}
                itemData={items}
                itemCount={itemCount}
                overscanCount={overscanCount}
                onItemsRendered={onItemsRendered}
                width={width}
                height={height}
                itemSize={itemSize}
              >
                {renderListItem({
                  isItemLoaded,
                  ItemComponent: itemComponent,
                  LoadingItemComponent: loadingItemComponent,
                })}
              </List>
            )}
          </AutoSizer>
        )}
      </InfiniteLoader>
    </div>
  );
}

export default React.memo(InfiniteList);
