package ru.skoltech.aeronetlab.urban.service.action.taskcreator;

import org.apache.commons.lang3.StringUtils;
import ru.skoltech.aeronetlab.urban.entity.business.AreaOfInterest;
import ru.skoltech.aeronetlab.urban.entity.workflow.Stage;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public interface TaskCreator {

    Set<Task> create(Stage stage);

    /**
     * Params can be either
     * 1. WD only, like "pipeline". Those parameters cannot be overriden by a Workflow
     * 2. Workflow, then WD, like "url". Workflow parameter will be used if any. WD parameter will be used as a default otherwise.
     * 3. AOI, then Workflow, then WD, like "model". AOI parameter will be used if any. Workflow or WD param will be used otherwise.
     *
     * Only explicitly listed parameters will end up int the results map
     *
     * @param stage WD stage
     * @param aoi Area of interest
     * @param wdOnlyParamNames set of params that should be defined in WD and cannot be overridden in Workflow parameters
     * @param workflowParamNames set of params that can be defined in Workflow parameters. If not, default valur from WD will be used
     * @param aoiParamNames set of parameters that can be returned during previous stages and stored in AOI
     * @param requiredParamNames set of required parameters. If any of required parameters is not defined, workflow will fail
     */
    default Map<String, String> prepareStageParameters(Stage stage,
                                                       AreaOfInterest aoi,
                                                       Set<String> wdOnlyParamNames,
                                                       Set<String> workflowParamNames,
                                                       Set<String> aoiParamNames,
                                                       Set<String> requiredParamNames) {

        Set<String> acceptedParamNames = new HashSet<>();
        acceptedParamNames.addAll(wdOnlyParamNames);
        acceptedParamNames.addAll(workflowParamNames);
        acceptedParamNames.addAll(aoiParamNames);

        assert acceptedParamNames.containsAll(requiredParamNames);

        Map<String, String> wdParams = stage.getStageDefinition().getParams();
        Map<String, String> workflowParams = stage.getWorkflow().getParams();
        Map<String, String> aoiParams = aoi.getParams();

        Map<String, String> result = new HashMap<>();
        for (String param : wdOnlyParamNames) {
            result.put(param, wdParams.get(param));
        }

        for (String param : workflowParamNames) {
            result.put(param, workflowParams.getOrDefault(param, wdParams.get(param)));
        }

        for (String param : aoiParamNames) {
            result.put(param, aoiParams.getOrDefault(param, workflowParams.getOrDefault(param, wdParams.get(param))));
        }

        Set<String> missingParams = new HashSet<>(requiredParamNames);
        missingParams.removeAll(result.keySet());

        if (!missingParams.isEmpty()) {
            throw new RuntimeException("Stage " + stage.getId() + " parameters doesn't contain required parameters " +
                    StringUtils.join(missingParams, ", "));
        }

        return result;
    }
}
