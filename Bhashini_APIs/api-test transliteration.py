import requests
import base64
import json

def transliteration(inp: str, source_lang: str, tar_lang: str):
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
                "taskType": "transliteration",
                "config": {
                    "language": {
                        "sourceLanguage": source_lang,
                        "targetLanguage": tar_lang
                    }
                }
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId" : pipeline_id
        }
    }

    response = requests.post(url="https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline", json=dict1, headers=headerdict)

    print(response.content)

    {"languages":[{"sourceLanguage":"hi","targetLanguageList":["hi"]}],"pipelineResponseConfig":[{"taskType":"asr","config":[{"serviceId":"ai4bharat/conformer-hi-gpu--t4","modelId":"648025f27cdd753e77f461a9","language":{"sourceLanguage":"hi","sourceScriptCode":"Deva"},"domain":["general"]}]}],"feedbackUrl":"https://dhruva-api.bhashini.gov.in/services/feedback/submit","pipelineInferenceAPIEndPoint":{"callbackUrl":"https://dhruva-api.bhashini.gov.in/services/inference/pipeline","inferenceApiKey":{"name":"Authorization","value":"sJRgS3zJfI7VEdtY21DexkvZZbhC4AkRMUOxuAo0j-kKFA9iyydDhHp3mwnuRBQF"},"isMultilingualEnabled":True,"isSyncApi":True},"pipelineInferenceSocketEndPoint":{"callbackUrl":"wss://dhruva-api.bhashini.gov.in","inferenceApiKey":{"name":"Authorization","value":"sJRgS3zJfI7VEdtY21DexkvZZbhC4AkRMUOxuAo0j-kKFA9iyydDhHp3mwnuRBQF"},"isMultilingualEnabled":True,"isSyncApi":True}}


    headers2 = {
        "Content-type": "application/json",
        response.json()["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]: response.json()["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    }

    dict2 = {
        "pipelineTasks": [
            {
                "taskType": "transliteration",
                "config": {
                    "language": {
                        "sourceLanguage": source_lang,
                        "targetLanguage": tar_lang
                    },
                    "serviceId": response.json()["pipelineResponseConfig"][0]["config"][0]["serviceId"],
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

    return compute_response.json()["pipelineResponse"][0]["output"][0]["source"]