FROM sbtscala/scala-sbt:17.0.2_1.6.2_2.13.8 AS builder
ADD . .
RUN ["sbt", "clean", "assembly"]
ENV JAVA_OPTS="-Xmx4g -Xms4g"

FROM openjdk:17.0.2-jdk-slim
COPY --from=builder  /root/target/scala-2.12/app.jar /scala-app/app.jar
WORKDIR /scala-app
CMD ["java $JAVA_OPTS -jar app.jar"]
