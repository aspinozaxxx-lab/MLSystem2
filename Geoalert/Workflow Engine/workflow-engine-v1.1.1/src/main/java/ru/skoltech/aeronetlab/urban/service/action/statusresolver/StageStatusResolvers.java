package ru.skoltech.aeronetlab.urban.service.action.statusresolver;

import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.definition.action.WithStageStatusResolver;

import java.lang.reflect.Field;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

@Service
public class StageStatusResolvers {

    private final Map<Action, StageStatusResolver> stageStatusResolversMap = new HashMap<>();

    public StageStatusResolvers(Collection<StageStatusResolver> StageStatusResolvers) {
        Map<Action, Class<? extends StageStatusResolver>> classes = new HashMap<>();
        for (Action action : Action.values()) {
            try {
                Field field = Action.class.getField(action.name());
                if (field.isAnnotationPresent(WithStageStatusResolver.class)) {
                    classes.put(action, field.getAnnotation(WithStageStatusResolver.class).value());
                }
            } catch (NoSuchFieldException e) {
                e.printStackTrace();
            }
        }

        classes.forEach((a, c) -> stageStatusResolversMap.put(
                a,
                StageStatusResolvers.stream()
                        .filter(s -> s.getClass() == c)
                        .findAny()
                        .orElseThrow(() -> new RuntimeException("Error configuring service dependencies.")))
        );
    }

    public StageStatusResolver get(Action action) {
        return stageStatusResolversMap.getOrDefault(action, stageStatus -> {
            throw new UnsupportedOperationException(
                    "StageStatusResolver for " + stageStatus.getStage() + " is not defined."
            );
        });
    }
}
