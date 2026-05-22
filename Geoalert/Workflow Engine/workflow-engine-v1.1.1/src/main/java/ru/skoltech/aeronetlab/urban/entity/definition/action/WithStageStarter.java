package ru.skoltech.aeronetlab.urban.entity.definition.action;

import ru.skoltech.aeronetlab.urban.service.action.stagestarter.StageStarter;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface WithStageStarter {
    Class<? extends StageStarter> value();
}
