package ru.skoltech.aeronetlab.markupstorage.queue;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vividsolutions.jts.geom.Geometry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.markupstorage.dto.ImportParams;
import ru.skoltech.aeronetlab.markupstorage.exception.ParseGeojsonException;
import ru.skoltech.aeronetlab.markupstorage.service.FeatureCollectionService;
import ru.skoltech.aeronetlab.markupstorage.service.export.ExportGeojsonService;
import ru.skoltech.aeronetlab.markupstorage.service.merge.MergeStrategy;
import ru.skoltech.aeronetlab.markupstorage.we.RunCheck;

import java.io.IOException;
import java.util.*;

@Service
public class MessageListener {

    private Logger log = LoggerFactory.getLogger(this.getClass());
    @Value("${FAIL_ON_RUNCHECK_ERROR:true}")
    private boolean failOnRuncheckError;

    @Autowired
    private FeatureCollectionService collectionService;

    @Autowired
    private ExportGeojsonService exportGeojsonService;

    @Autowired
    private MessageSender messageSender;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired 
    private RunCheck runCheck;

    @RabbitListener(queues = "#{taskQueue}")
    public void receiveMessage(Map<String, Object> message) {
        String taskId = message.get("task_id").toString();
        boolean shouldProcessTask = true;
        // if runcheck url is null or if we answer is 0 do the task
        try {
            String runcheckUrl = message.get("runcheck_url").toString();

            log.info("Send runcheck request to " + runcheckUrl);

            String runcheckResult = runCheck.checkTask(runcheckUrl);

            log.info("Runcheck result " + runcheckResult);

            if(Objects.equals(runcheckResult, "1")) {
              shouldProcessTask = false;
            }
        } catch (NullPointerException e) {
          log.error("Runcheck url is not provided", e.toString());
        } catch (Exception e) {
          shouldProcessTask = !failOnRuncheckError;
          log.error(e.toString());
        }

        try {
          if (shouldProcessTask) {
            log.info("Received message in task queue with task_id=" + taskId);

            String task = (String) ((Map<String, Object>) message.get("input")).get("task");
            if (task.equals("import_vector")) {
                importVector(message);
            } else if (task.equals("export_geojson")) {
                exportGeojson(message);
            } else {
                throw new UnsupportedOperationException("Unknown task: " + task);
            }

            messageSender.sendOk(taskId);
          } else {
            log.info("Skip the task: " + taskId);
          }
        } catch (Exception e) {
            log.error("Error processing task: task_id=" + taskId + "; error: " + e.toString(), e);
            messageSender.sendFail(taskId, e.toString());
        }
    }

    private void importVector(Map<String, Object> message) throws IOException, ParseGeojsonException {
        Map<String, Object> inputParams = (Map<String, Object>) message.get("input");
        Map<String, Object> outputParams = (Map<String, Object>) message.get("output");

        String bucket = inputParams.get("bucket").toString();
        String fileName = inputParams.get("filename").toString();

        Object areaObj = inputParams.get("mask");
        Optional<Geometry> mask = Optional.ofNullable(areaObj).map(a -> objectMapper.convertValue(a, Geometry.class));

        boolean cropToMask = Optional.ofNullable(inputParams.get("crop_to_mask"))
                .filter(s -> s.toString().equalsIgnoreCase("true"))
                .isPresent();

        MergeStrategy mergeStrategy = Optional.ofNullable(inputParams.get("merge_strategy"))
                .map(s -> MergeStrategy.valueOf(s.toString()))
                .orElse(MergeStrategy.INSTANCE_SEGMENTATION);

        Set<String> keyProperties = Optional.ofNullable(inputParams.get("key_properties"))
                .map(kp -> new HashSet<>((List<String>) kp))
                .orElse(new HashSet<>());

        Optional<UUID> layerId = Optional.of(UUID.fromString(outputParams.get("layer_id").toString()));
        Optional<String> layerName = Optional.ofNullable(outputParams.get("layer_name")).map(n -> n.toString());

        ImportParams params = new ImportParams()
                .setLayerName(layerName)
                .setLayerId(layerId)
                .setMergeStrategy(mergeStrategy)
                .setKeyProperties(keyProperties)
                .setMask(mask)
                .setCropToMask(cropToMask);

        collectionService.postFromS3(bucket, fileName, params);
    }

    private void exportGeojson(Map<String, Object> message) {
        Map<String, Object> inputParams = (Map<String, Object>) message.get("input");
        Map<String, Object> outputParams = (Map<String, Object>) message.get("output");

        UUID layerId = UUID.fromString(inputParams.get("layer_id").toString());

        Object areaObj = inputParams.get("mask");
        Optional<Geometry> mask = Optional.ofNullable(areaObj).map(a -> objectMapper.convertValue(a, Geometry.class));

        boolean points = Optional.ofNullable(inputParams.get("points"))
                .map(a -> Boolean.parseBoolean(a.toString()))
                .orElse(false);

        String bucket = outputParams.get("bucket").toString();
        String fileName = outputParams.get("filename").toString();

        exportGeojsonService.exportGeojsonToS3(layerId, mask, bucket, fileName, points);
    }
}
