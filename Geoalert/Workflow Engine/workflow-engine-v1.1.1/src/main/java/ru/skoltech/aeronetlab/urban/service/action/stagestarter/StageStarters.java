package ru.skoltech.aeronetlab.urban.service.action.stagestarter;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.definition.action.WithStageStarter;

import java.lang.reflect.Field;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

@Service
public class StageStarters {

    @Autowired
    private DefaultStageStarter defaultStageStarter;

    private final Map<Action, StageStarter> stageStartersMap = new HashMap<>();

    public StageStarters(Collection<StageStarter> stageStarters) {
        for (Action action : Action.values()) {
            try {
                Field field = Action.class.getField(action.name());
                if (field.isAnnotationPresent(WithStageStarter.class)) {
                    Class<? extends  StageStarter> clazz = field.getAnnotation(WithStageStarter.class).value();
                    //classes.put(action, );
                    StageStarter stageStarter = stageStarters.stream()
                            .filter(s -> s.getClass() == clazz)
                            .findFirst()
                            .orElseThrow(() -> new RuntimeException("Cannot find Stage Starter by annotation: " + clazz));
                    stageStartersMap.put(action, stageStarter);
                }
            } catch (NoSuchFieldException e) {
                throw new RuntimeException("Action misconfigured: " + action);
            }
        }
    }

    public StageStarter get(Action action) {
        return stageStartersMap.getOrDefault(action, defaultStageStarter);
    }
}
