package ru.skoltech.aeronetlab.urban.service.definition;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import org.apache.commons.io.IOUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.context.annotation.Import;
import org.springframework.core.io.InputStreamSource;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import ru.skoltech.aeronetlab.urban.entity.definition.StageDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinition;
import ru.skoltech.aeronetlab.urban.entity.definition.WorkflowDefinitionVer;
import ru.skoltech.aeronetlab.urban.entity.definition.action.Action;
import ru.skoltech.aeronetlab.urban.repository.definition.StageDefinitionRepository;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(SpringExtension.class)
@DataJpaTest
@Import({
        WorkflowDefinitionImporter.class,
        WorkflowDefinitionExporter.class,
        ObjectMapper.class
})
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
public class WorkflowDefinitionImporterIntegrationTest {

    @Autowired
    private WorkflowDefinitionImporter workflowDefinitionImporter;

    @Autowired
    private WorkflowDefinitionExporter workflowDefinitionExporter;

    @Autowired
    private StageDefinitionRepository stageDefinitionRepository;

    @Test
    public void testImportWorkflowDefinition() throws Exception {
        InputStream is = WorkflowDefinitionImporterIntegrationTest.class.getResourceAsStream("workflow_definition.yml");
        String yml = IOUtils.toString(is, StandardCharsets.UTF_8);
        int n = Lists.newArrayList(stageDefinitionRepository.findAll()).size();
        WorkflowDefinitionVer workflowDefVer = workflowDefinitionImporter.importWorkflowDefinition(yml);

        assertEquals(n + 6, Lists.newArrayList(stageDefinitionRepository.findAll()).size());

        Optional<StageDefinition> actualOpt = stageDefinitionRepository.findFirstStages(workflowDefVer).stream().findFirst();
        assertTrue(actualOpt.isPresent());
        StageDefinition actualFirstStage = actualOpt.get();

        assertEquals(5, actualFirstStage.getParams().values().size());
        assertEquals(1, stageDefinitionRepository.findNextStages(actualFirstStage).size());
    }

    @Test
    public void testImportWorkflowDefinitionWhenBadVersion() {
        assertThrows(BadWorkflowDefinitionException.class, () -> {
            InputStream is = WorkflowDefinitionImporterIntegrationTest.class.getResourceAsStream("workflow_definition_unsupported_version.yml");
            String yml = IOUtils.toString(is, StandardCharsets.UTF_8);
            workflowDefinitionImporter.importWorkflowDefinition(yml);
        });
    }

    @Test
    public void testImportWorkflowDefinitionWhenBadStageType() {
        assertThrows(BadWorkflowDefinitionException.class, () -> {
            InputStream is = WorkflowDefinitionImporterIntegrationTest.class.getResourceAsStream("workflow_definition_unsupported_action.yml");
            String yml = IOUtils.toString(is, StandardCharsets.UTF_8);
            workflowDefinitionImporter.importWorkflowDefinition(yml);
        });
    }


    @Test
    public void testExportWorkflowDefinition() throws Exception {
        WorkflowDefinition wd = new WorkflowDefinition();
        wd.setName("Test WD");
        wd.setId(42L);

        WorkflowDefinitionVer v0 = new WorkflowDefinitionVer();
        v0.setId(43L);
        v0.setVersion(0);
        v0.setWorkflowDefinition(wd);

        StageDefinition sd0 = new StageDefinition();
        sd0.setAction(Action.SELECT_SOURCE);
        sd0.setWorkflowDefinitionVer(v0);
        sd0.setDescription("Description");
        sd0.setName("select-source");
        sd0.setRetries(3);
        sd0.setRetryInterval(17);
        sd0.setParams(Collections.singletonMap("foo", "bar"));

        StageDefinition sd1 = new StageDefinition();
        sd1.setAction(Action.BUILD_COG);
        sd1.setWorkflowDefinitionVer(v0);
        sd1.setDescription("Description 2");
        sd1.setName("build-cog");
        sd1.setRetries(0);
        sd1.setRetryInterval(0);
        sd1.setPreviousStages(Collections.singleton(sd0));

        v0.setStageDefinitions(Lists.newArrayList(sd0, sd1));
        wd.setVersions(Collections.singletonList(v0));

        InputStreamSource iss = workflowDefinitionExporter.entityToYml(wd);
        String yml = new String(iss.getInputStream().readAllBytes(), Charsets.UTF_8);

        InputStream is = WorkflowDefinitionImporterIntegrationTest.class.getResourceAsStream("workflow_definition_export.yml");
        assert is != null;
        String expected = new String(is.readAllBytes(), Charsets.UTF_8);
        assertEquals(expected, yml);
    }
}
