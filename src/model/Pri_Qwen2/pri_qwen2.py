import torch
import torch.nn as nn
import re
from torch import Tensor
from transformers import Qwen2ForCausalLM, Qwen2Config
from typing import Optional, Union


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


class PriQwen2Config(Qwen2Config):
    def __init__(
        self,
        projector_type="mlp2x_gelu",
        retriever_hidden_size=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.retriever_hidden_size, config.hidden_size, dtype=torch.bfloat16)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.bfloat16))
            self.projector = nn.Sequential(*modules)

    def forward(self, context_embedding):
        return self.projector(context_embedding)


class PriQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if (
            hasattr(config, "retriever_hidden_size")
            and config.retriever_hidden_size > 0
        ):
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size
        self.post_init()

    def set_placeholder_token_id(self, token_id):
        self.placeholder_token_id = token_id

    def get_embed_tokens(self, input_ids):
        embeds = self.model.embed_tokens(input_ids)
        return embeds

    def prepare_inputs_embeds(self, input_ids, retrieval_embeds):
        inputs_embeds = self.model.embed_tokens(input_ids)
        retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)

        ## sanity check
        num_placeholder_tokens = torch.sum(input_ids == self.placeholder_token_id).item()
        num_retrieval_embeds = retrieval_embeds.shape[0]
        assert num_placeholder_tokens == num_retrieval_embeds, (
            num_placeholder_tokens,
            num_retrieval_embeds,
        )

        # print("retrieval_embeds2: ",retrieval_embeds.grad)
        retrieval_embeds = self.projector(retrieval_embeds).to(self.device)
        # print("retrieval_embeds3: ",retrieval_embeds.grad)

        mask = (input_ids == self.placeholder_token_id).unsqueeze(-1).expand_as(inputs_embeds)

        expanded_retrieval = torch.zeros_like(inputs_embeds).to(self.device)
        expanded_retrieval[mask] = retrieval_embeds.reshape(-1) 
        inputs_embeds = inputs_embeds * (~mask) + expanded_retrieval
        # print("inputs_embeds: ",inputs_embeds.grad)

        # print("equal: ",torch.equal(inputs_embeds,inputs_embeds2))
        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        retrieval_embeds=None,  ## [-1,retrieval_hidden_size]
        attention_mask=None,
        **kwargs,
    ):
        ## when inputs_embeds is passed, it means the model is doing generation
        ## and only the first round of generation would pass inputs_embeds
        ## https://github.com/huggingface/transformers/blob/79132d4cfe42eca5812e8c45ea1b075f04f907b6/src/transformers/models/llama/modeling_llama.py#L1250
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        at_the_beginning_of_generation = False
        if inputs_embeds is not None:
            assert not self.training
            assert retrieval_embeds is None
            at_the_beginning_of_generation = True

        if not at_the_beginning_of_generation:
            ## a single forward
            if retrieval_embeds is not None:
                inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
                input_ids = None
                if attention_mask is not None:
                    assert inputs_embeds.shape[1] == attention_mask.shape[1], (
                        inputs_embeds.shape,
                        attention_mask.shape,
                    )

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        retrieval_embeds=None,
        **kwargs,
    ):
        """
        Override the generate method to support retrieval-augmented generation.
        
        This method extends the standard Qwen2 generation by allowing the use of 
        retrieval embeddings that are projected and injected at placeholder positions.
        
        Args:
            input_ids: Token IDs for input sequence. Will be replaced with inputs_embeds 
                       if retrieval_embeds is provided.
            retrieval_embeds: External embeddings from a retriever model to incorporate 
                             during generation. Shape: [num_placeholders, retriever_hidden_size]
            **kwargs: Additional arguments passed to the underlying generate method
            
        Returns:
            torch.LongTensor: Generated token sequences
            
        Raises:
            NotImplementedError: If inputs_embeds is directly provided in kwargs
        """
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")

        inputs_embeds = None
        if retrieval_embeds is not None:
            # Convert input tokens to embeddings, injecting retrieval embeddings at placeholder positions
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None
            if attention_mask is not None:
                # Ensure attention mask matches the embedding sequence length
                assert inputs_embeds.shape[1] == attention_mask.shape[1], (
                    inputs_embeds.shape,
                    attention_mask.shape,
                )
            return super().generate(
                attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
            )

        else:
            # Fall back to standard generation without retrieval augmentation
            return super().generate(
                attention_mask=attention_mask, input_ids=input_ids, **kwargs
            )

    def get_embed_dim(self):
        """
        Get the dimension of the model's hidden representation.
        
        Returns:
            int: Size of the embedding dimension from model config
        """
        return self.config.hidden_size

    def get_embed_length(self):
        """
        Get the number of embeddings produced per input.
        
        Returns:
            int: Always 1 since this model produces a single embedding per document
        """
        return 1

    def get_embedding(self, input_ids, attention_mask):
        """
        Generate dense embedding representation from input tokens.
        
        This method performs a forward pass through the model and extracts the
        final hidden state of the last token as the embedding.
        
        Args:
            input_ids: Token IDs of the input sequence. Shape: [batch_size, seq_len]
            attention_mask: Mask identifying valid input tokens. Shape: [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Pooled embeddings with shape [batch_size, hidden_size]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        return embeddings

    def get_doc_embedding(self, input_ids, attention_mask):
        """
        Generate document embeddings using the model.
        
        A convenience method that calls get_embedding, provided for API consistency
        with other retriever models.
        
        Args:
            input_ids: Token IDs of the document
            attention_mask: Attention mask for the document
            
        Returns:
            torch.Tensor: Document embeddings with shape [batch_size, hidden_size]
        """
        return self.get_embedding(input_ids, attention_mask)

    def get_query_embedding(self, input_ids, attention_mask):
        """
        Generate query embeddings using the model.
        
        A convenience method that calls get_embedding, provided for API consistency
        with other retriever models. Allows for potential future specialization
        of query embedding behavior.
        
        Args:
            input_ids: Token IDs of the query
            attention_mask: Attention mask for the query
            
        Returns:
            torch.Tensor: Query embeddings with shape [batch_size, hidden_size]
        """
        return self.get_embedding(input_ids, attention_mask)
