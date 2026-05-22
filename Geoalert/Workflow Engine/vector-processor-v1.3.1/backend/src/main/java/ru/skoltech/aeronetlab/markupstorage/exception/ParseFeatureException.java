package ru.skoltech.aeronetlab.markupstorage.exception;

public class ParseFeatureException extends ParseGeojsonException {

    public ParseFeatureException(Long ordinal, Throwable cause) {

        super("Couldn't parse feature #" + ordinal + ", error message is: " + cause.getMessage(), cause);
    }
}
