package ru.skoltech.aeronetlab.markupstorage.controller;

import com.fasterxml.jackson.annotation.JsonView;
import com.vividsolutions.jts.geom.Geometry;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;
import ru.skoltech.aeronetlab.markupstorage.controller.view.View;
import ru.skoltech.aeronetlab.markupstorage.dao.FeatureCollectionEntity;
import ru.skoltech.aeronetlab.markupstorage.dto.ImportParams;
import ru.skoltech.aeronetlab.markupstorage.exception.ParseGeojsonException;
import ru.skoltech.aeronetlab.markupstorage.service.FeatureCollectionService;
import ru.skoltech.aeronetlab.markupstorage.service.export.StreamingExportGeojsonService;
import ru.skoltech.aeronetlab.markupstorage.service.merge.MergeStrategy;

import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

@RestController
@RequestMapping("/api/v0/collections")
public class FeatureCollectionController {

    @Autowired
    private FeatureCollectionService collectionService;

    @Autowired
    private StreamingExportGeojsonService streamingExportGeojsonService;

    @GetMapping()
    @JsonView(View.Basic.class)
    @ApiOperation("Returns all feature collections in markup storage.")
    public List<FeatureCollectionEntity> getAllFeatureCollections() {

        return collectionService.getAllFeatureCollections();
    }

    @GetMapping("/{collection_id}")
    @JsonView(View.Expanded.class)
    @ApiOperation("Returns the specified feature collection from markup storage.")
    public ResponseEntity<FeatureCollectionEntity> getFeatureCollection(@PathVariable("collection_id") String collectionId) {

        return ResponseEntity.ok(collectionService.getFeatureCollection(UUID.fromString(collectionId)));
    }

    @PostMapping("/url")
    @JsonView(View.Basic.class)
    @ApiOperation("Imports geojson or zip with geojsons from the specified url. Features are imported asynchronously.")
    public ResponseEntity<FeatureCollectionEntity> postFeatureCollectionFromUrl(
            @ApiParam("Url to geojson file or zip with geojsons") @RequestBody String url,
            @RequestParam(required = false) String name,
            @ApiParam("INSTANTCE_SEGMENTATION or SEMANTIC_SEGMENTATION")
            @RequestParam(required = true) String mergeStrategy,
            @ApiParam("Key property list (for SEMANTIC_SEGMENTATION only)")
            @RequestParam(required = false, defaultValue = "[]") Set<String> keyProperties,
            @ApiParam("Geometry to mask features (optional)")
            @RequestParam(required = false) Geometry mask,
            @ApiParam("Сrop features that partially intersect with the mask")
            @RequestParam(required = false, defaultValue = "true") boolean cropToMask
    ) {

        ImportParams params = new ImportParams()
                .setLayerId(Optional.empty())
                .setLayerName(Optional.ofNullable(name))
                .setMask(Optional.ofNullable(mask))
                .setCropToMask(cropToMask)
                .setMergeStrategy(MergeStrategy.valueOf(mergeStrategy))
                .setKeyProperties(keyProperties);

        try {
            FeatureCollectionEntity collection = collectionService.postFromUrl(url, params);
            return ResponseEntity.accepted().body(collection);
        } catch (IOException | ParseGeojsonException e) {
            throw new RuntimeException(e); //TODO exception handling
        }
    }

    @PostMapping("/file")
    @JsonView(View.Basic.class)
    @ApiOperation("Imports geojson file. Features are imported asynchronously.")
    public ResponseEntity<FeatureCollectionEntity> postFeatureCollectionFromFile(
            @ApiParam("Geojson file") @RequestParam MultipartFile file,
            @RequestParam(required = false) String name,
            @ApiParam("INSTANTCE_SEGMENTATION or SEMANTIC_SEGMENTATION")
            @RequestParam(required = true) String mergeStrategy,
            @ApiParam("Key property list (for SEMANTIC_SEGMENTATION only)")
            @RequestParam(required = false, defaultValue = "[]") Set<String> keyProperties,
            @ApiParam("Geometry to mask features (optional)")
            @RequestParam(required = false) Geometry mask,
            @ApiParam("Сrop features that partially intersect with the mask")
            @RequestParam(required = false, defaultValue = "true") boolean cropToMask
    ) {

        ImportParams params = new ImportParams()
                .setLayerId(Optional.empty())
                .setLayerName(Optional.ofNullable(name))
                .setMask(Optional.ofNullable(mask))
                .setCropToMask(cropToMask)
                .setMergeStrategy(MergeStrategy.valueOf(mergeStrategy))
                .setKeyProperties(keyProperties);

        try {
            FeatureCollectionEntity collection =
                    collectionService.postFromMultipartFile(file, params);
            return ResponseEntity.accepted().body(collection);
        } catch (IOException | ParseGeojsonException e) {
            throw new RuntimeException(e); //TODO exception handling
        }
    }

    @Transactional
    @DeleteMapping("/{collection_id}")
    @ApiOperation("Deletes the specified feature collection from markup storage.")
    public ResponseEntity<Void> deleteFeatureCollection(@PathVariable("collection_id") String collectionId) {

        collectionService.deleteFeatureCollection(UUID.fromString(collectionId));
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @PostMapping("/{collection_id}/export")
    @ApiOperation("Exports features to geojson, filtered by the area (optionally)")
    @Transactional(readOnly = true)
    public StreamingResponseBody export(
            @PathVariable("collection_id") String collectionId,
            @ApiParam("If true, converts polygons to points")
            @RequestParam(required = false, defaultValue = "false") boolean points,
            @ApiParam("If true, wrap features in a FeatureCollection, otherwise returns features separated by comma")
            @RequestParam(required = false, defaultValue = "true") boolean wrapInFeatureCollection,
            @ApiParam(value = "Geojson geometry to filter features (optional)")
            @RequestBody(required = false) Geometry area
    ) {
        return os -> streamingExportGeojsonService.pipeGeojson(
                UUID.fromString(collectionId), Optional.ofNullable(area), os, points, wrapInFeatureCollection
        );
    }
}
