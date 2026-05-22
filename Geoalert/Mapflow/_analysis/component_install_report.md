# Отчет по установке компонентов

Дата проверки: 2026-05-22.

## Установлено

- Java JDK 17 установлен через `winget install --id EclipseAdoptium.Temurin.17.JDK --exact --silent`.
- `sbt` установлен через `winget install --id sbt.sbt --exact --silent`.

Проверка:

- `java -version` возвращает `openjdk version "17.0.19"`.
- `sbt --version` в `Geoalert/Mapflow/mapflow-api-v1.3.2` возвращает:
  - project sbt: `1.6.2`;
  - runner sbt: `1.12.11`.

## Не установлено

- Docker Desktop не установлен.

Причина: `winget install --id Docker.DockerDesktop --exact --silent` скачал установщик, но Docker Desktop потребовал запуск от администратора. Установка завершилась ошибкой `0x800704c7: The operation was canceled by the user`. Команда `docker --version` после этого недоступна.

Дополнительно: WSL не установлен, `wsl --status` сообщает, что подсистема Windows для Linux не установлена.

## Проверка сборки mapflow-api

Команда:

```powershell
$env:SBT_OPTS='-Djdk.util.zip.disableZip64ExtraFieldValidation=true'
sbt Test/compile
```

Результат:

- без `SBT_OPTS` JDK 17 не открывает старый `scalactic_2.13-3.2.5.jar`;
- с `SBT_OPTS` компиляция проходит дальше и падает на исходниках выгрузки Geoalert.

Текущий блокер сборки уже не установка компонентов, а поврежденный или санитизированный Scala-код в `mapflow-api-v1.3.2`. Примеры ошибок:

- `DefaultConfig.scala`: `val dbPassword:  = ...`;
- `DataProvider.scala`: `credentialsPassword: ]`;
- `User.scala`: `password: ]`;
- `AvanpostClient.scala`: `def userInfo(token: , id: UUID)`;
- несколько строковых литералов обрываются после `s"..."`.

## Вывод

Java и `sbt` готовы к использованию. Для Docker нужен интерактивный запуск установщика с правами администратора и, вероятно, включение WSL2 или Hyper-V.
