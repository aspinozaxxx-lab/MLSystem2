import React from "react";
import { Button, Tooltip } from "@blueprintjs/core";
import paginationModel from "hooks/use-pagination";

const pageSizeVariants = [10, 20, 30];

const Pagination = ({
  onChangePage,
  onChangePageSize,
  isLoading,
  limit,
  offset,
  total,
}) => {
  const pageSize = limit;
  const currentPage = 1 + Math.floor(offset / limit);
  const totalPages = Math.round(total / limit);

  const options = {
    currentPage,
    pageSize,
    total,
    siblingCount: 2,
  };

  const { range, dotOffset } = paginationModel.usePagination(options);

  const hasRange = (range?.length || 0) > 1;

  return (
    <div className="pagination">
      <div className="stack-h-sm">
        {hasRange &&
          range &&
          range.map((page) => {
            if (page === currentPage) {
              return (
                <Button key={page} disabled>
                  {page}
                </Button>
              );
            }

            let nextPage = page;

            if (page === paginationModel.LEFT_DOT)
              nextPage = Math.max(1, currentPage - dotOffset);
            if (page === paginationModel.RIGHT_DOT)
              nextPage = Math.min(totalPages, currentPage + dotOffset);

            if (page < 0) {
              return (
                <Tooltip
                  key={page}
                  content={page < 0 ? nextPage : ""}
                  position="bottom"
                >
                  <Button onClick={() => onChangePage(nextPage)}>
                    {page < 0 ? "..." : page}
                  </Button>
                </Tooltip>
              );
            }

            return (
              <Button key={page} onClick={() => onChangePage(nextPage)}>
                {page < 0 ? "..." : page}
              </Button>
            );
          })}
      </div>
      <div className="stack-h-sm">
        <Button disabled minimal hidden={!isLoading} loading={isLoading} />
        {pageSizeVariants.map((sizeVariant) => {
          if (sizeVariant === pageSize) {
            return (
              <Button key={sizeVariant} disabled>
                {sizeVariant}
              </Button>
            );
          }

          return (
            <Button
              key={sizeVariant}
              variant="outline"
              onClick={() => onChangePageSize(sizeVariant)}
            >
              {sizeVariant}
            </Button>
          );
        })}
      </div>
    </div>
  );
};

export default Pagination;

