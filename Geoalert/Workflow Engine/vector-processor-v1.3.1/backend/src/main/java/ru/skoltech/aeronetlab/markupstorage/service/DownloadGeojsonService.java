package ru.skoltech.aeronetlab.markupstorage.service;

import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class DownloadGeojsonService {

    private final String DATA_PATH = "data";

    @Autowired
    private ZipService zipService;

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public List<Path> download(String urlString, boolean useCache) throws IOException {

        Paths.get(DATA_PATH).toFile().mkdirs();

        URL url = new URL(urlString);

        String ext = FilenameUtils.getExtension(url.getPath());

        if (ext.equalsIgnoreCase("zip")) {
            return downloadAndUnzipGeojson(url, useCache);
        } else if (ext.equalsIgnoreCase("geojson")) {
            return downloadGeojson(url, useCache);
        } else {
            throw new IllegalArgumentException("Unsupported file type: " + ext); //TODO exception handling, should be 400
        }
    }

    private List<Path> downloadGeojson(URL url, boolean useCache) throws IOException {

        String fileName = FilenameUtils.getName(url.getPath());
        Path path = Paths.get(DATA_PATH, fileName);

        if (useCache && Files.exists(path)) {
            logger.info("Using cache: " + path);
            return Arrays.asList(path);
        } else {
            downloadFile(url, path);
            return Arrays.asList(path);
        }
    }

    private List<Path> downloadAndUnzipGeojson(URL url, boolean useCache) throws IOException {

        String zipFileName = FilenameUtils.getName(url.getPath());
        Path zipPath = Paths.get(DATA_PATH, zipFileName);

        if (useCache && Files.exists(zipPath)) {
            logger.info("Using cache: " + zipPath);
            return unzipGeojsons(zipPath);
        } else {
            downloadFile(url, zipPath);
            return unzipGeojsons(zipPath);
        }
    }

    private void downloadFile(URL url, Path path) throws IOException {

        logger.info("Downloading from " + url);

        InputStream is = url.openStream();

        Files.copy(is, path, StandardCopyOption.REPLACE_EXISTING);
    }

    private List<Path> unzipGeojsons(Path zipPath) throws IOException {

        String dirName = UUID.randomUUID().toString();

        Path targetPath = Paths.get(DATA_PATH, dirName);
        targetPath.toFile().mkdirs();

        return zipService.unzipFile(zipPath, targetPath)
                .stream()
                .filter((p) -> FilenameUtils.getExtension(p.toString()).equalsIgnoreCase("geojson"))
                .collect(Collectors.toList());
    }
}
