package ru.skoltech.aeronetlab.markupstorage.exception;

public class ParseGeojsonException extends Exception {

    public ParseGeojsonException(String msg) {
        super(msg);
    }

    public ParseGeojsonException(String msg, Throwable cause) {
        super(msg, cause);
    }
}
