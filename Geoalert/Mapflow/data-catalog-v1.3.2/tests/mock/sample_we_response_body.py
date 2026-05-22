page_response = """
    [
  {
    "id": 4410719,
    "workflowDefinitionId": 4403750,
    "stages": [
      {
        "id": 4410729,
        "name": "build-cog",
        "description": "Preparing imagery",
        "status": "OK",
        "taskIds": [
          4410740
        ],
        "statusUpdateDate": "2022-12-23T06:01:04.616492"
      },
      {
        "id": 4410727,
        "name": "import-vector",
        "description": "Saving results",
        "status": "OK",
        "taskIds": [
          4410743
        ],
        "statusUpdateDate": "2022-12-23T06:01:06.325728"
      },
      {
        "id": 4410725,
        "name": "inference",
        "description": "Forests semantic segmentation",
        "status": "OK",
        "taskIds": [
          4410737
        ],
        "statusUpdateDate": "2022-12-23T06:01:05.093252"
      },
      {
        "id": 4410723,
        "name": "load-data",
        "description": "Downloading data",
        "status": "OK",
        "taskIds": [
          4410734
        ],
        "statusUpdateDate": "2022-12-23T06:01:03.430144"
      },
      {
        "id": 4410721,
        "name": "select-source",
        "description": "Source selection",
        "status": "OK",
        "statusUpdateDate": "2022-12-23T06:01:02.381049"
      }
    ],
    "areasOfInterest": [
      {
        "id": 4410720,
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [
                46.53527498245239,
                61.15699011530739
              ],
              [
                46.53527498245239,
                61.1593708531447
              ],
              [
                46.541454792022705,
                61.1593708531447
              ],
              [
                46.541454792022705,
                61.15699011530739
              ],
              [
                46.53527498245239,
                61.15699011530739
              ]
            ]
          ]
        }
      }
    ],
    "rasterLayer": {
      "id": 4410739,
      "uri": "https://rasters-staging.mapflow.ai/api/v0/cogs/tiles.json?uri=s3://workflow-white-maps/workflow-4410719/bc0f7df6-dc31-4ebf-9d5e-5b888f6be0fb"
    },
    "vectorLayer": {
      "id": 4410742,
      "uri": "https://vector-staging.mapflow.ai/api/layers/140b9cbf-1b7e-40aa-8344-eaf21cc1cef6.json",
      "layerId": "140b9cbf-1b7e-40aa-8344-eaf21cc1cef6"
    },
    "artifacts": [
      {
        "areaOfInterestId": 4410720,
        "artifactType": "RAW_VECTOR",
        "uri": "s3://workflow-white-maps/workflow-4410719/86661a67-cff1-4e88-bb0d-00d8cb480f18/area-4410720.geojson"
      },
      {
        "areaOfInterestId": 4410720,
        "artifactType": "RAW_RASTER",
        "uri": "s3://workflow-white-maps/workflow-4410719/2789280b-31c1-4755-8f94-fbe24d92d898/area-4410720.tif"
      }
    ],
    "status": "OK",
    "statusUpdateDate": "2022-12-23T06:01:06.32636",
    "createDate": "2022-12-23T06:01:02.239364",
    "system": "monitoring",
    "params": {},
    "meta": {}
  },
  {
    "id": 4410901,
    "workflowDefinitionId": 4403750,
    "stages": [
      {
        "id": 4410911,
        "name": "build-cog",
        "description": "Preparing imagery",
        "status": "OK",
        "taskIds": [
          4410919
        ],
        "statusUpdateDate": "2022-12-23T13:01:04.866109"
      },
      {
        "id": 4410909,
        "name": "import-vector",
        "description": "Saving results",
        "status": "OK",
        "taskIds": [
          4410925
        ],
        "statusUpdateDate": "2022-12-23T13:01:06.513273"
      },
      {
        "id": 4410907,
        "name": "inference",
        "description": "Forests semantic segmentation",
        "status": "OK",
        "taskIds": [
          4410922
        ],
        "statusUpdateDate": "2022-12-23T13:01:05.331583"
      },
      {
        "id": 4410905,
        "name": "load-data",
        "description": "Downloading data",
        "status": "OK",
        "taskIds": [
          4410916
        ],
        "statusUpdateDate": "2022-12-23T13:01:03.490859"
      },
      {
        "id": 4410903,
        "name": "select-source",
        "description": "Source selection",
        "status": "OK",
        "statusUpdateDate": "2022-12-23T13:01:02.168466"
      }
    ],
    "areasOfInterest": [
      {
        "id": 4410902,
        "geometry": {
          "type": "Polygon",
          "coordinates": [
            [
              [
                46.53527498245239,
                61.15699011530739
              ],
              [
                46.53527498245239,
                61.1593708531447
              ],
              [
                46.541454792022705,
                61.1593708531447
              ],
              [
                46.541454792022705,
                61.15699011530739
              ],
              [
                46.53527498245239,
                61.15699011530739
              ]
            ]
          ]
        }
      }
    ],
    "rasterLayer": {
      "id": 4410918,
      "uri": "https://rasters-staging.mapflow.ai/api/v0/cogs/tiles.json?uri=s3://workflow-white-maps/workflow-4410901/22c646c6-8dab-4b77-9df3-c05727f697da"
    },
    "vectorLayer": {
      "id": 4410924,
      "uri": "https://vector-staging.mapflow.ai/api/layers/1ad6e62d-6a29-4444-ad5d-3b0347a5818f.json",
      "layerId": "1ad6e62d-6a29-4444-ad5d-3b0347a5818f"
    },
    "artifacts": [
      {
        "areaOfInterestId": 4410902,
        "artifactType": "RAW_VECTOR",
        "uri": "s3://workflow-white-maps/workflow-4410901/5bb0f190-c60c-4633-bc31-8a8f47534bfd/area-4410902.geojson"
      },
      {
        "areaOfInterestId": 4410902,
        "artifactType": "RAW_RASTER",
        "uri": "s3://workflow-white-maps/workflow-4410901/88a9822d-ce1a-43cf-b5ce-f7c5744fec07/area-4410902.tif"
      }
    ],
    "status": "OK",
    "statusUpdateDate": "2022-12-23T13:01:06.514477",
    "createDate": "2022-12-23T13:01:02.050661",
    "system": "monitoring",
    "params": {},
    "meta": {}
    }
    ]
    """

add_workflow_response = """{
    "id": 4412998,
    "workflowDefinitionId": 4409285,
    "stages": [
        {
            "id": 4413004,
            "name": "build-cog",
            "description": "Preparing imagery",
            "status": "PENDING",
            "taskIds": [
                4413009
            ],
            "statusUpdateDate": "2022-12-26T13:19:31.796238"
        },
        {
            "id": 4413002,
            "name": "load-data",
            "description": null,
            "status": "PENDING",
            "statusUpdateDate": "2022-12-26T13:19:31.794893"
        },
        {
            "id": 4413000,
            "name": "select-source",
            "description": "Source selection",
            "status": "PENDING",
            "statusUpdateDate": "2022-12-26T13:19:31.793535"
        }
    ],
    "areasOfInterest": [
        {
            "id": 4412999,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            47.96080936452077,
                            42.40995566963073
                        ],
                        [
                            47.96080936452077,
                            42.417112542624224
                        ],
                        [
                            47.976346120449676,
                            42.417112542624224
                        ],
                        [
                            47.976346120449676,
                            42.40995566963073
                        ],
                        [
                            47.96080936452077,
                            42.40995566963073
                        ]
                    ]
                ]
            }
        }
    ],
    "rasterLayer": {
        "id": 4409501,
        "uri": "https://rasters-staging.mapflow.ai/api/v0/cogs/tiles.json?uri=s3://test-bucket/3/0f0353ff978d4998bdc994704ceeace8.tif"
    },
    "vectorLayer": null,
    "artifacts": [
        {
            "areaOfInterestId": 4412999,
            "artifactType": "RAW_RASTER",
            "uri": "s3://users-data/a.trekin@geoalert.io_1a761087-20a6-4436-b202-15ac857d27ea/f184b344-dcce-4710-9cb1-0fda4dc5bb8f/0f0353ff978d4998bdc994704ceeace8.tif"
        }
    ],
    "status": "IN_PROGRESS",
    "statusUpdateDate": "2022-12-26T13:19:31.796813",
    "createDate": "2022-12-26T13:19:31.770361",
    "processingId": "MyAwesomeProcessing",
    "params": {
        "raster-layer-uri": "s3://test-bucket/3/0f0353ff978d4998bdc994704ceeace8.tif",
        "source_type": "local",
        "priority": "9",
        "url": "s3://users-data/a.trekin@geoalert.io_1a761087-20a6-4436-b202-15ac857d27ea/f184b344-dcce-4710-9cb1-0fda4dc5bb8f/0f0353ff978d4998bdc994704ceeace8.tif"
    },
    "meta": {}
}
"""