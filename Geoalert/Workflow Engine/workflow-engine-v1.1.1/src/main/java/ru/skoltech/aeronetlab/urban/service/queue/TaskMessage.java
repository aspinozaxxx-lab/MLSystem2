package ru.skoltech.aeronetlab.urban.service.queue;

import java.util.HashMap;
import java.util.Map;

public class TaskMessage {

  private Long task_id;
  private String processing_id;
  private Map<String, Object> input = new HashMap<>();
  private Map<String, Object> output = new HashMap<>();
  private String runcheck_url;

  public Long getTask_id() {
    return task_id;
  }

  public void setTask_id(Long task_id) {
    this.task_id = task_id;
  }

  public String getProcessing_id() {
    return processing_id;
  }

  public void setProcessing_id(String processing_id) {
    this.processing_id = processing_id;
  }

  public Map<String, Object> getInput() {
    return input;
  }

  public void setInput(Map<String, Object> input) {
    this.input = input;
  }

  public Map<String, Object> getOutput() {
    return output;
  }

  public void setOutput(Map<String, Object> output) {
    this.output = output;
  }

  public String getRuncheck_url() {
    return runcheck_url;
  }

  public void setRuncheck_url(Long task_id, String baseUrl) {
    this.runcheck_url = String.format("%s/api/v0/tasks/%s/runcheck", baseUrl, task_id);
  }
}
