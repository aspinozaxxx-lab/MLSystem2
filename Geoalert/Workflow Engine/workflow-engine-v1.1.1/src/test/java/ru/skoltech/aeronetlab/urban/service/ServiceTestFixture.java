package ru.skoltech.aeronetlab.urban.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.entity.definition.StageDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.entity.workflow.*;
import ru.skoltech.aeronetlab.urban.repository.definition.StageDefinitionRepository;
import ru.skoltech.aeronetlab.urban.repository.definition.WorkflowDefinitionRepository;
import ru.skoltech.aeronetlab.urban.repository.definition.WorkflowDefinitionVerRepository;
import ru.skoltech.aeronetlab.urban.repository.workflow.*;

import java.time.LocalDateTime;

@Service
public class ServiceTestFixture {
    @Autowired
    private WorkflowRepository workflowRepository;

    @Autowired
    private WorkflowDefinitionRepository workflowDefinitionRepository;

    @Autowired
    private WorkflowDefinitionVerRepository workflowDefinitionVerRepository;

    @Autowired
    private WorkflowStatusRepository workflowStatusRepository;

    @Autowired
    private StageDefinitionRepository stageDefinitionRepository;

    @Autowired
    private StageRepository stageRepository;

    @Autowired
    private StageStatusRepository stageStatusRepository;

    @Autowired
    private TaskRepository taskRepository;

    @Autowired
    private TaskStatusRepository taskStatusRepository;


    @Transactional(propagation = Propagation.REQUIRED)
    public WorkflowDefinitionVer createWorkflowDefinitionVer() {
        WorkflowDefinition wd = new WorkflowDefinition();
        wd.setName("Mock workflow definition");
        workflowDefinitionRepository.save(wd);
        WorkflowDefinitionVer wdv = new WorkflowDefinitionVer();
        wdv.setWorkflowDefinition(wd);
        wdv.setVersion(0);
        workflowDefinitionVerRepository.save(wdv);

        StageDefinition sd = new StageDefinition();
        sd.setWorkflowDefinitionVer(wdv);
        sd.setAction(Action.INFERENCE);
        sd.setName("inference");
        stageDefinitionRepository.save(sd);

        return wdv;
    }

    @Transactional(propagation = Propagation.REQUIRED)
    public Workflow createWorkflow(WorkflowDefinitionVer wdv) {
        Workflow w = new Workflow(wdv);
        w.setCreateDate(LocalDateTime.parse("2022-05-13T00:12:35"));
        workflowRepository.save(w);

        WorkflowStatus ws = new WorkflowStatus(w);
        ws.setUpdateDate(LocalDateTime.parse("2022-05-13T00:13:00"));
        ws.setStatus(StatusType.IN_PROGRESS);
        w.setWorkflowStatus(ws);
        workflowStatusRepository.save(ws);

        stageDefinitionRepository.findAllByWorkflowDefinitionVer(wdv).forEach(sd -> {
            Stage s = new Stage();
            s.setStageDefinition(sd);
            s.setWorkflow(w);
            stageRepository.save(s);

            StageStatus ss = new StageStatus(s);
            ss.setStatus(StatusType.IN_PROGRESS);
            ss.setStartDate(LocalDateTime.parse("2022-05-13T00:12:40"));
            ss.setUpdateDate(LocalDateTime.parse("2022-05-13T00:13:00"));
            stageStatusRepository.save(ss);
            s.setStageStatus(ss);

            Task t = new Task(s, null);
            taskRepository.save(t);

            TaskStatus ts = new TaskStatus(t);
            ts.setStartDate(LocalDateTime.parse("2022-05-13T00:12:40"));
            ts.setUpdateDate(LocalDateTime.parse("2022-05-13T00:13:00"));
            ts.setStatus(StatusType.IN_PROGRESS);
            taskStatusRepository.save(ts);
            t.setTaskStatus(ts);
        });

        return w;
    }


}
