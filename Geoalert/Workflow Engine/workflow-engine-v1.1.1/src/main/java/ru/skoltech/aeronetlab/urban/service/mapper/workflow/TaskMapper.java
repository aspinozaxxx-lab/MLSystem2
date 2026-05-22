package ru.skoltech.aeronetlab.urban.service.mapper.workflow;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import ru.skoltech.aeronetlab.urban.api.exception.NotFoundException;
import ru.skoltech.aeronetlab.urban.dto.workflow.TaskDto;
import ru.skoltech.aeronetlab.urban.entity.workflow.Task;

import jakarta.persistence.EntityManager;
import jakarta.persistence.NoResultException;
import jakarta.persistence.TypedQuery;

@Service
public class TaskMapper {

    @Autowired
    private EntityManager entityManager;

    public TaskDto getTask(Long id) {
        TypedQuery<TaskDto> wQuery = entityManager.createQuery(
                "select new ru.skoltech.aeronetlab.urban.dto.workflow.TaskDto(" +
                        "t.id, t.request, ts.status, " +
                        "ts.message, ts.attempts, ts.updateDate) " +
                        "from TaskStatus ts " +
                        "join ts.task t " +
                        "where t.id = :id",
                TaskDto.class);

        try {
            return wQuery.setParameter("id", id).getSingleResult();
        } catch (NoResultException e) {
            throw new NotFoundException(Task.class, id);
        }
    }
}