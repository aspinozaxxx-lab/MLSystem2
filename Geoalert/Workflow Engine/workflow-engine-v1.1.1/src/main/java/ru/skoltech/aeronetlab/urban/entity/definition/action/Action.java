package ru.skoltech.aeronetlab.urban.entity.definition.action;

import ru.skoltech.aeronetlab.urban.action.buildcog.BuildCogStageStarter;
import ru.skoltech.aeronetlab.urban.action.buildcog.BuildCogTaskCreator;
import ru.skoltech.aeronetlab.urban.action.dataloader.DataLoaderStageStarter;
import ru.skoltech.aeronetlab.urban.action.dataloader.DataLoaderTaskCreator;
import ru.skoltech.aeronetlab.urban.action.importvector.ImportVectorStageStarter;
import ru.skoltech.aeronetlab.urban.action.importvector.ImportVectorTaskCreator;
import ru.skoltech.aeronetlab.urban.action.inference.InferenceStageStarter;
import ru.skoltech.aeronetlab.urban.action.inference.InferenceTaskCreator;
import ru.skoltech.aeronetlab.urban.action.selectmodel.SelectModelStageStarter;
import ru.skoltech.aeronetlab.urban.action.selectmodel.SelectModelStageStatusResolver;
import ru.skoltech.aeronetlab.urban.action.selectmodel.SelectModelTaskCreator;
import ru.skoltech.aeronetlab.urban.action.selectsource.SelectSourceStageStarter;
import ru.skoltech.aeronetlab.urban.action.selectsource.SelectSourceStageStatusResolver;
import ru.skoltech.aeronetlab.urban.action.userinput.UserInputStageStarter;
import ru.skoltech.aeronetlab.urban.action.userinput.UserInputStageStatusResolver;
import ru.skoltech.aeronetlab.urban.action.validatesource.ValidateSourceStageStarter;
import ru.skoltech.aeronetlab.urban.action.validatesource.ValidateSourceTaskCreator;
import ru.skoltech.aeronetlab.urban.service.action.statusresolver.TaskStageStatusResolver;

import java.util.Optional;
import java.util.stream.Stream;

public enum Action {
    @WithStageStarter(DataLoaderStageStarter.class)
    @WithStageStatusResolver(TaskStageStatusResolver.class)
    @WithTaskCreator(DataLoaderTaskCreator.class)
    DATA_LOADER("dataloader", Optional.of("dataloader"), false),

    @WithStageStarter(SelectSourceStageStarter.class)
    @WithStageStatusResolver(SelectSourceStageStatusResolver.class)
    SELECT_SOURCE("select-source", Optional.empty(), false),

    @WithStageStarter(SelectModelStageStarter.class)
    @WithStageStatusResolver(SelectModelStageStatusResolver.class)
    @WithTaskCreator(SelectModelTaskCreator.class)
    SELECT_MODEL("select-model", Optional.of("model-selector"), false),

    @WithStageStarter(ValidateSourceStageStarter.class)
    @WithStageStatusResolver(TaskStageStatusResolver.class)
    @WithTaskCreator(ValidateSourceTaskCreator.class)
    VALIDATE_SOURCE("validate-source", Optional.of("source-validator"), false),

    @WithStageStarter(UserInputStageStarter.class)
    @WithStageStatusResolver(UserInputStageStatusResolver.class)
    USER_INPUT("user-input", Optional.empty(), true),

    @WithStageStarter(BuildCogStageStarter.class)
    @WithStageStatusResolver(TaskStageStatusResolver.class)
    @WithTaskCreator(BuildCogTaskCreator.class)
    BUILD_COG("build-cog", Optional.of("raster-processor"), false),

    @WithStageStarter(InferenceStageStarter.class)
    @WithStageStatusResolver(TaskStageStatusResolver.class)
    @WithTaskCreator(InferenceTaskCreator.class)
    INFERENCE("inference", Optional.of("inference"), false),

    @WithStageStarter(ImportVectorStageStarter.class)
    @WithStageStatusResolver(TaskStageStatusResolver.class)
    @WithTaskCreator(ImportVectorTaskCreator.class)
    IMPORT_VECTOR("import-vector", Optional.of("vector-processor"), false);

    private final String actionName;

    private final Optional<String> workerName;

    private final boolean userInput;

    public static Optional<Action> fromActionName(String actionName) {
        return Stream.of(Action.values()).filter(st -> st.actionName.equalsIgnoreCase(actionName)).findAny();
    }

    Action(String actionName, Optional<String> workerName, boolean userInput) {
        this.actionName = actionName;
        this.workerName = workerName;
        this.userInput = userInput;
    }

    public String getActionName() {
        return actionName;
    }

    public Optional<String> getWorkerName() {
        return workerName;
    }

    public boolean requiresUserInput() {
        return userInput;
    }
}
