FROM eclipse-temurin:18-jdk-alpine

ADD ./build.gradle /backend/build.gradle
ADD ./gradlew /backend/gradlew
ADD ./gradle /backend/gradle

WORKDIR /backend

RUN ./gradlew resolveDependencies

ADD . /backend

EXPOSE 8080

#RUN apk add --no-cache curl && \
#    ./gradlew assemble

CMD ["./gradlew", "bootRun"]