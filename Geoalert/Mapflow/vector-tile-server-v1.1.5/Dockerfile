FROM aeronetlab/scala-sbt:1.3.3-jre11-alpine AS builder
ADD . .
RUN ["sbt", "clean", "assembly"]

FROM adoptopenjdk/openjdk11:jre-11.0.5_10-alpine
COPY --from=builder  ./target/scala-2.12/app.jar /scala-app/app.jar
WORKDIR /scala-app
CMD ["java", "-jar", "app.jar"]