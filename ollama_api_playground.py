import json
from dataclasses import dataclass
from typing import Optional
import requests


@dataclass
class OllamaConfiguration:
    """Model configuration class."""

    base_url: str = "http://localhost:11434"
    """Base url the model is hosted under."""

    model: str = "mistral"
    """The model name."""

    prompt: str = "Hi"
    """The user prompt."""

    format: Optional[str] = None
    """The format to return a response in. Currently the only accepted value is `json`."""

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback from the generated text."""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity of the output."""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the next token."""

    num_gqa: Optional[int] = None
    """The number of GQA groups in the transformer layer."""

    num_gpu: Optional[int] = None
    """The number of layers to send to the GPU(s)."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation."""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent repetition."""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions."""

    temperature: Optional[float] = None
    """The temperature of the model."""

    seed: Optional[int] = None
    """Sets the random number seed to use for generation."""

    stop: Optional[str] = None
    """Sets the stop sequences to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable tokens from the output."""

    num_predict: Optional[int] = None
    """Maximum number of tokens to predict when generating text."""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense."""

    top_p: Optional[float] = None
    """Works together with top-k."""

    system: Optional[str] = None
    """System prompt to (overrides what is defined in the `Modelfile`)."""

    template: Optional[str] = None
    """The full prompt or prompt template (overrides what is defined in the `Modelfile`)."""

    context: Optional[str] = None
    """The context parameter returned from a previous request to `/generate`, this can be used to keep a short conversational memory."""

    stream: Optional[bool] = None
    """If `false` the response will be returned as a single response object, rather than a stream of objects."""

    raw: Optional[bool] = None
    """If `true` no formatting will be applied to the prompt and no context will be returned."""

    @staticmethod
    def remove_none_values(d):
        if isinstance(d, dict):
            return {k: OllamaConfiguration.remove_none_values(v) for k, v in d.items() if v is not None}
        else:
            return d

    def to_json(self):

        data = {
            "model": self.model,
            "format": self.format,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gqa": self.num_gqa,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "seed": self.seed,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
                "num_predict": self.num_predict,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
            "system": self.system,
            "template": self.template,
            "context": self.context,
            "stream": self.stream,
            "raw": self.raw
        }

        return OllamaConfiguration.remove_none_values(data)

    def generate_response(self, prompt):
        response = requests.post(
            url=f"{self.base_url}/api/generate/",
            headers={"Content-Type": "application/json"},
            json={"prompt": prompt, **self.to_json()},
            stream=True,
        )

        return response


config = OllamaConfiguration(
    system="You are a general purpose thinking machine that can read text , reason it and then give out a strcutured response",
    model="openhermes2.5-mistral", temperature=0.4, stream=False, top_k=10, format="json")

prompt = "For the question 'A stock went down 30% over the night , how would you react to it' a user responded with 'Man, thats actually scary I cannot see that much of a swing\
    in values' . Classify this user's risk_tolerance as 'high' 'medium' or 'low' in a json format"

print(json.loads(config.generate_response(prompt=prompt).text)["response"])
