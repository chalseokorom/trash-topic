def get_representation_model():
    import anthropic
    from bertopic.representation import Anthropic

    MODEL_NAME = "claude-haiku-4-5-20251001"

    client = anthropic.Anthropic()
    return Anthropic(client, model=MODEL_NAME,
                     nr_docs=5, delay_in_seconds=0.5)