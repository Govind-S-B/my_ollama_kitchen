from typing import Optional
from pydantic import BaseModel


class ModelConfiguration(BaseModel):
    """Model configuration class."""

    base_url: str = "http://localhost:11434"
    """Base url the model is hosted under."""

    model: str = "mistral"
    """The model name."""

    prompt: str = "Hi"
    """The model name."""

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
