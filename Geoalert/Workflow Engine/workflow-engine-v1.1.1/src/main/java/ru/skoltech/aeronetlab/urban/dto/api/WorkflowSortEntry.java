package ru.skoltech.aeronetlab.urban.dto.api;

import org.apache.commons.lang3.builder.ToStringBuilder;

public class WorkflowSortEntry {
    private WorkflowSortField sortField;
    private boolean desc;

    public WorkflowSortField getSortField() {
        return sortField;
    }

    public boolean isDesc() {
        return desc;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("sortField", sortField)
                .append("desc", desc)
                .toString();
    }
}
