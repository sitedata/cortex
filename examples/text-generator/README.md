# Deploy OpenAI's GPT-2 as an API on AWS

This example shows how to deploy OpenAI's GPT-2 model as a service on AWS. To train your own GPT-2 with any text you'd like, run [this notebook in Google Colab.](https://colab.research.google.com/drive/1OoutHRGSk95c5QG1Ji5EbmOJ4Kjw7J06#scrollTo=kc5cIgeEmv8o)

GPT-2 is OpenAI's state of the art unsupervised language model capable of generating coherent paragraphs of text—so coherent, in fact, that OpenAI didn't initially release their full trained model for fear of it being used nefariously. With over 1.5 billion paramters, GPT-2 is designed to predict the next word in a string of words, based on the preceding text. As an example of how powerful it is, read [this example](https://openai.com/blog/better-language-models/#sample2).

To deploy GPT-2 as an api, you first need to export a trained model, which you can do [here](https://colab.research.google.com/drive/1OoutHRGSk95c5QG1Ji5EbmOJ4Kjw7J06#scrollTo=kc5cIgeEmv8o). Once you have a model, you can deploy with Cortex by following the below steps:

## Define a deployment

A `deployment` specifies a set of resources that are deployed as a single unit. An `api` makes a model available as a web service that can serve real-time predictions. This configuration will download the GPT-2 model from your S3 bucket, preprocess the payload and postprocess the inference with functions defined in `handler.py`, and deploy each replica of the API on 1 GPU.

```yaml
- kind: deployment
  name: text

- kind: api
  name: generator
  model: s3://your-bucket/your-model
  request_handler: handler.py
  compute:
    cpu: 1
    gpu: 1
```

<!-- CORTEX_VERSION_MINOR -->
You can run the code to generate a pretrained GPT-2 model [here](https://colab.research.google.com/github/cortexlabs/cortex/blob/master/examples/text-generator/gpt-2.ipynb). You can also train your own GPT-2 model with any text you'd like with [this notebook here.](https://colab.research.google.com/drive/1OoutHRGSk95c5QG1Ji5EbmOJ4Kjw7J06#scrollTo=lOgT-s9pucCB)

## Add request handling

The model requires encoded data for inference, but the API should accept strings of natural language as input. It should also decode the model’s prediction before responding to the client. This can be implemented in a request handler file using the pre_inference and post_inference functions.

```python
from encoder import get_encoder
encoder = get_encoder()


def pre_inference(sample, metadata):
    context = encoder.encode(sample["text"])
    return {"context": [context]}


def post_inference(prediction, metadata):
    response = prediction["sample"]
    return encoder.decode(response)
```

## Deploy to AWS

`cortex deploy` takes the declarative configuration from `cortex.yaml` and creates it on the cluster.

```bash
$ cortex deploy

deployment started
```

Behind the scenes, Cortex containerizes the model, makes it servable using TensorFlow Serving, exposes the endpoint with a load balancer, and orchestrates the workload on Kubernetes.

You can track the status of a deployment using `cortex get`:

```bash
$ cortex get generator --watch

status   up-to-date   available   requested   last update   avg latency
live     1            1           1           8s            -
```

The output above indicates that one replica of the API was requested and one replica is available to serve predictions. Cortex will automatically launch more replicas if the load increases and spin down replicas if there is unused capacity.

## Serve real-time predictions

```bash
$ cortex get generator

url: http://***.amazonaws.com/text/generator

$ curl http://***.amazonaws.com/text/generator \
    -X POST -H "Content-Type: application/json" \
    -d '{"text": "machine learning"}'
```

Any questions? [chat with us](https://gitter.im/cortexlabs/cortex).
