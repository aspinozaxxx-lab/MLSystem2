package ru.skoltech.aeronetlab.markupstorage.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

@Service
public class ZipService {

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    public List<Path> unzipFile(Path zipPath, Path targetPath) throws IOException {

        logger.info("Unzipping from " + zipPath + " to " + targetPath);

        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(zipPath))) {
            List<Path> paths = new ArrayList<>();
            ZipEntry zipEntry = zis.getNextEntry();

            while (zipEntry != null) {
                String fileName = zipEntry.getName();
                Files.copy(zis, targetPath.resolve(fileName), StandardCopyOption.REPLACE_EXISTING);
                paths.add(targetPath.resolve(fileName));
                zipEntry = zis.getNextEntry();
            }

            return paths;
        }
    }
}
