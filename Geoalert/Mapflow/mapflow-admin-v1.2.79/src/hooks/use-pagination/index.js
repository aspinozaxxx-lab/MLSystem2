import { range } from "ramda";
import { useMemo } from "react";

const LEFT_DOT = -1;
const RIGHT_DOT = -2;

function usePagination({ total, pageSize, siblingCount = 1, currentPage }) {
  const paginationRange = useMemo(() => {
    const totalPages = Math.ceil(total / pageSize);

    // Pages count is determined as siblingCount + firstPage + lastPage + currentPage + 2*DOTS
    const totalPageNumbers = siblingCount + 5;

    /*
      Case 1:
      If the number of pages is less than the page numbers we want to show in our
      paginationComponent, we return the range [1..totalPageCount]
    */
    if (totalPageNumbers >= totalPages) {
      return range(1, totalPages + 1);
    }

    /*
    	Calculate left and right sibling index and make sure they are within range 1 and totalPageCount
    */
    const leftSiblingIndex = Math.max(currentPage - siblingCount, 1);
    const rightSiblingIndex = Math.min(currentPage + siblingCount, totalPages);

    /*
      We do not show dots just when there is just one page number to be inserted between the extremes of sibling and the page limits i.e 1 and totalPageCount. Hence we are using leftSiblingIndex > 2 and rightSiblingIndex < totalPageCount - 2
    */
    const shouldShowLeftDots = leftSiblingIndex > 2;
    const shouldShowRightDots = rightSiblingIndex < totalPages - 2;

    const firstPageIndex = 1;
    const lastPageIndex = totalPages;

    /*
    	Case 2: No left dots to show, but rights dots to be shown
    */
    if (!shouldShowLeftDots && shouldShowRightDots) {
      const leftItemCount = 3 + 2 * siblingCount;
      const leftRange = range(1, leftItemCount + 1);

      return [...leftRange, RIGHT_DOT, totalPages];
    }

    /*
    	Case 3: No right dots to show, but left dots to be shown
    */
    if (shouldShowLeftDots && !shouldShowRightDots) {
      const rightItemCount = 3 + 2 * siblingCount;
      const rightRange = range(totalPages - rightItemCount + 1, totalPages + 1);
      return [firstPageIndex, LEFT_DOT, ...rightRange];
    }

    /*
    	Case 4: Both left and right dots to be shown
    */
    if (shouldShowLeftDots && shouldShowRightDots) {
      const middleRange = range(leftSiblingIndex, rightSiblingIndex + 1);
      return [
        firstPageIndex,
        LEFT_DOT,
        ...middleRange,
        RIGHT_DOT,
        lastPageIndex,
      ];
    }

    return range(1, totalPages + 1);
  }, [total, pageSize, siblingCount, currentPage]);

  return { range: paginationRange, dotOffset: siblingCount + 5 };
}

const paginationModel = {
  usePagination,
  LEFT_DOT,
  RIGHT_DOT,
};

export default paginationModel;
