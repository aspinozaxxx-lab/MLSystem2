package ru.skoltech.aeronetlab.urban.service.definition;

import java.util.*;
import java.util.function.Function;

public class TopologicalSorter<T> {

    private Map<T, Set<T>> graph = new HashMap<>();
    private Set<T> visited;
    private LinkedList<T> sorting;

    public TopologicalSorter(Set<T> nodes,
                             Function<T, Set<T>> getParents) {
        for (T node : nodes) {
            graph.put(node, new HashSet<>());
        }

        for (T node : nodes) {
            for (T parent : getParents.apply(node)) {
                graph.get(parent).add(node);
            }
        }
    }

    public List<T> sort() {
        return sortStartingWith(graph.keySet());
    }

    public List<T> sortStartingWith(Set<T> roots) {
        sorting = new LinkedList<>();
        visited = new HashSet<>();

        for (T node : roots) {
            visit(node);
        }

        return sorting;
    }

    private void visit(T node) {
        if (visited.contains(node)) return;

        visited.add(node);

        for (T child : graph.get(node)) {
            visit(child);
        }

        sorting.push(node);
    }
}
