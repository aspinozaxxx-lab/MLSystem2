package ru.skoltech.aeronetlab.urban.entity.definition.action;

import ru.skoltech.aeronetlab.urban.service.action.statusresolver.StageStatusResolver;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface WithStageStatusResolver {
    Class<? extends StageStatusResolver> value();
}
