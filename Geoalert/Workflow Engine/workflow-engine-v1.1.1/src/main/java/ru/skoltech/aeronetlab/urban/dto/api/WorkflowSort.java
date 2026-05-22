package ru.skoltech.aeronetlab.urban.dto.api;

import org.apache.commons.lang3.builder.ToStringBuilder;

import java.util.ArrayList;
import java.util.List;

public class WorkflowSort {
    private List<WorkflowSortEntry> entries = new ArrayList<>();

    public List<WorkflowSortEntry> getEntries() {
        return entries;
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this)
                .append("entries", entries)
                .toString();
    }
}
