package ru.skoltech.aeronetlab.urban.dto.api;

import org.apache.commons.lang3.builder.ToStringBuilder;

public class WorkflowsPage {
    private WorkflowFilter filter;
    private WorkflowSort sort;
    private int offset;
    private int limit;

    public WorkflowFilter getFilter() {
        return filter;
    }

    public void setFilter(WorkflowFilter filter) {
        this.filter = filter;
    }

    public WorkflowSort getSort() {
        return sort;
    }

    public void setSort(WorkflowSort sort) {
        this.sort = sort;
    }

    public int getOffset() {
        return offset;
    }

    public void setOffset(int offset) {
        this.offset = offset;
    }

    public int getLimit() {
        return limit;
    }

    public void setLimit(int limit) {
        this.limit = limit;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("filter", filter)
                .append("sort", sort)
                .append("offset", offset)
                .append("limit", limit)
                .toString();
    }
}
