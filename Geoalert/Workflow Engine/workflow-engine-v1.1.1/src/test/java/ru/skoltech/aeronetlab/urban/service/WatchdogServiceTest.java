package ru.skoltech.aeronetlab.urban.service;


import io.minio.MinioClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.annotation.Rollback;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.entity.workflow.*;
import ru.skoltech.aeronetlab.urban.repository.workflow.*;
import ru.skoltech.aeronetlab.urban.service.queue.MessageListener;
import ru.skoltech.aeronetlab.urban.service.queue.MessageSender;
import ru.skoltech.aeronetlab.urban.service.queue.status.StatusUpdateListener;
import ru.skoltech.aeronetlab.urban.service.queue.status.StatusUpdateSender;
import ru.skoltech.aeronetlab.urban.service.watchdog.WatchdogScheduler;
import ru.skoltech.aeronetlab.urban.service.watchdog.WatchdogService;

import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import java.time.LocalDateTime;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest()
@ExtendWith(SpringExtension.class)
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
public class WatchdogServiceTest {
    @MockBean(name = "minioClient")
    private MinioClient minioClient;

    @MockBean(name = "minioClientExternally")
    private MinioClient minioClientExternally;

    @MockBean
    private StatusUpdateSender statusUpdateSender;

    @MockBean
    private StatusUpdateListener statusUpdateListener;

    @MockBean
    private MessageSender messageSender;

    @MockBean
    private WatchdogScheduler scheduler;

    @MockBean
    private MessageListener messageListener;

    @Autowired
    private WatchdogService watchdogService;

    @Autowired
    private ServiceTestFixture serviceTestFixture;

    @Autowired
    private WorkflowRepository workflowRepository;

    @Autowired
    private StageRepository stageRepository;

    @Autowired
    private TaskRepository taskRepository;

    @PersistenceContext
    private EntityManager entityManager;

    @Test
    @Transactional(propagation = Propagation.REQUIRED)
    @Rollback(false)
    @Timeout(3000)
    public void test() {
        WorkflowDefinitionVer wdv = serviceTestFixture.createWorkflowDefinitionVer();
        Workflow w = serviceTestFixture.createWorkflow(wdv);
        entityManager.flush();

        watchdogService.terminateStuckWorkflows(LocalDateTime.parse("2022-05-15T00:13:00"));
        entityManager.flush();

        Optional<Workflow> workflowOpt = workflowRepository.findById(w.getId());
        assertTrue(workflowOpt.isPresent());

        WorkflowStatus workflowStatus = workflowOpt.get().getWorkflowStatus();
        assertEquals(StatusType.FAILED, workflowStatus.getStatus());

        Set<Stage> stages = stageRepository.findAllByWorkflow(w);
        assertEquals(1, stages.size());
        Stage stage = stages.stream().findFirst().orElseThrow();
        assertEquals(StatusType.FAILED, stage.getStageStatus().getStatus());

        Set<Task> tasks = taskRepository.findAllByStage(stage);
        Task task = tasks.stream().findFirst().orElseThrow();
        assertEquals(StatusType.FAILED, task.getTaskStatus().getStatus());
    }
}
