package ru.skoltech.aeronetlab.markupstorage.controller;

import com.fasterxml.jackson.annotation.JsonView;
import com.vividsolutions.jts.geom.Geometry;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.transaction.annotation.Transactional;
import ru.skoltech.aeronetlab.markupstorage.controller.view.View;
import ru.skoltech.aeronetlab.markupstorage.dto.Feature;
import ru.skoltech.aeronetlab.markupstorage.service.FeatureService;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v0/collections/{collection_id}/features")
public class FeatureController {

    @Autowired
    private FeatureService featureService;

    @PostMapping("/search")
    @JsonView(View.Expanded.class)
    @ApiOperation("Searches for all the features located inside the given geometry (optionally).")
    @Transactional(readOnly = true)
    public ResponseEntity<List<Feature>> searchInArea(
            @PathVariable("collection_id") String collectionId,
            @ApiParam("If true, converts polygons to points")
            @RequestParam(required = false, defaultValue = "false") boolean points,
            @ApiParam(value = "Geojson geometry of the search area (optional)") @RequestBody Geometry area
    ) {
        System.out.println("Search request received." );
        LocalDateTime now = LocalDateTime.now();

        ResponseEntity<List<Feature>> result = ResponseEntity.ok(
                featureService.getFeatures(UUID.fromString(collectionId), Optional.of(area), points)
                        .collect(Collectors.toList())
        );
        System.out.println("Search request complete in " + now.until(LocalDateTime.now(), ChronoUnit.MILLIS) + "ms");
        return result;
    }
}
