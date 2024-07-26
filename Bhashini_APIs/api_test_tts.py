import requests
import base64
import json

def tts(inp: str, source_lang: str):
    # s = load_dataset('mozilla-foundation/common_voice_17_0')
    api_key = '3595c593d0-f5f8-43ed-89d9-f3705eef4a1c'
    user_id = "b408e5164680495c9e6ff8f4431f6469"
    pipeline_id = '64392f96daac500b55c543cd'

    headerdict = {
        "Content-type": "application/json",
        "userID": user_id,
        "ulcaApiKey": api_key
    }
    dict1 = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {
                        "sourceLanguage": source_lang
                    }
                }
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId" : pipeline_id
        }
    }

    response = requests.post(url="https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline", json=dict1, headers=headerdict)
    headers2 = {
        "Content-type": "application/json",
        response.json()["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]: response.json()["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    }

    dict2 = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {
                        "sourceLanguage": source_lang,
                    },
                    "serviceId": response.json()["pipelineResponseConfig"][0]["config"][0]["serviceId"],
                    "gender":"female"
                }
            }
        ],
        "inputData": {
            "input": [
                {
                    "source": inp
                }
            ]

        }
    }

    compute_response = requests.post(response.json()["pipelineInferenceAPIEndPoint"]["callbackUrl"], json=dict2, headers=headers2)
    return compute_response.json()["pipelineResponse"][0]["audio"][0]["audioContent"]
