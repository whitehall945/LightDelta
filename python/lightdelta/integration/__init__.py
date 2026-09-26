"""vLLM general plugin: registration runs in both API and worker processes."""


def register() -> None:
    import os

    if "LIGHTDELTA_CONFIG" not in os.environ:
        return  # Native vLLM baseline deliberately does not install the layer replacement.
    from lightdelta.integration import vllm_layer  # noqa: F401
