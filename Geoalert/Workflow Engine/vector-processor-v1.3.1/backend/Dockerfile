FROM openjdk:8-jdk-alpine

ADD ./build.gradle /markup-storage/build.gradle
ADD ./gradlew /markup-storage/gradlew
ADD ./gradle /markup-storage/gradle

WORKDIR /markup-storage

RUN ./gradlew resolveDependencies

ADD . /markup-storage

EXPOSE 8080

RUN ./gradlew assemble

CMD ["./gradlew", "bootRun"]