from transformers.modeling_outputs import BaseModelOutputWithPooling


def embedding_to_normalized_list(embeddings: BaseModelOutputWithPooling) -> list:
    img_embedding = embeddings.pooler_output

    if img_embedding is None:
        raise ValueError("Image embedding cannot be None!")

    img_embedding = img_embedding / img_embedding.norm(dim=1, keepdim=True)

    return img_embedding.squeeze().cpu().tolist()
