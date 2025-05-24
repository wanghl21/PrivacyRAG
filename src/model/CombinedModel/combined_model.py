import torch
import torch.nn as nn
import random

class CombinedModel(nn.Module):
    def __init__(self, retriever, model):
        super().__init__()
        self.retriever = retriever
        self.model = model

    def forward(self, **kwargs):
        """
        Args:
            input_ids: input_ids for the model
            attention_mask: attention_mask for the model
            use_prirag: whether to use prirag
            retriever_input_ids: input_ids for retriever
            retriever_attention_mask: attention_mask for retriever
        """
        use_prirag = kwargs.pop("use_prirag", False)
        if use_prirag:
            retriever_input_ids = kwargs.pop("retriever_input_ids", None)
            retriever_attention_mask = kwargs.pop("retriever_attention_mask", None)
            input_query_ids = kwargs.pop("retriever_query_input_ids", None)
            query_attention_mask = kwargs.pop(
                "retriever_query_attention_mask_for_model", None
            )
            query_aware_rate = kwargs.pop("query_aware_rate", 0.0)
            query_aware_ = random.random()
            # Randomly decide whether to use query-aware retrieval based on probability
            if query_aware_ < query_aware_rate:
                retrieval_embeds = self.get_retrieval_embeds_query_aware(
                    input_query_ids=input_query_ids,
                    query_attention_mask=query_attention_mask,
                    input_ids=retriever_input_ids,
                    attention_mask=retriever_attention_mask,
                )
            else:
                retrieval_embeds = self.get_retrieval_embeds(
                    input_ids=retriever_input_ids,
                    attention_mask=retriever_attention_mask,
                )
            input_ids = kwargs.pop("input_ids", None)
            attention_mask = kwargs.pop("attention_mask", None)
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                retrieval_embeds=retrieval_embeds,
            )
        else:
            # Standard LLM inference without retrieval augmentation
            input_ids = kwargs.pop("input_ids", None)
            attention_mask = kwargs.pop("attention_mask", None)
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    def get_retrieval_embeds(
        self,
        input_ids,
        attention_mask=None,
        pool_tokenid=-1,
    ):
        """
        Generate document embeddings using the retriever component.
        
        Args:
            input_ids: Token IDs of the document to embed
            attention_mask: Attention mask to identify valid tokens
            pool_tokenid: Specific token ID to pool around, -1 means standard pooling
            
        Returns:
            Flattened document embeddings tensor
        """
        embeds = self.retriever.get_doc_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pool_tokenid=pool_tokenid,
        )
        # Reshape embeddings to ensure consistent dimensions
        embeds = embeds.view(-1, embeds.shape[-1])
        return embeds

    def get_retrieval_nnembd_embeds(self, input_ids, attention_mask=None):
        """
        Generate hybrid embeddings that combine retriever document embeddings 
        with neural network token embeddings from the language model.
        
        Args:
            input_ids: Token IDs to generate embeddings for
            attention_mask: Attention mask to identify valid tokens
            
        Returns:
            Combined embeddings tensor with dimensions [batch_size, embedding_dim*2]
        """
        # Determine which embedding method to use based on model architecture
        model_name = str(self.model.config.architectures)
        
        # Get neural network embeddings from the language model
        if "qwen" in model_name.lower():
            nnembeds = torch.mean(
                self.model.get_embed_tokens(input_ids), dim=1
            )  # torch.Size([1, 4096])
        else:
            nnembeds = torch.mean(self.model.model.embed_tokens(input_ids), dim=1)
        retrieval_embeds = self.retriever.get_doc_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        retrieval_embeds = retrieval_embeds.view(-1, retrieval_embeds.shape[-1])
        retrieval_nnembd_embeds = torch.cat([retrieval_embeds, nnembeds], dim=1)

        return retrieval_nnembd_embeds  # torch.Size([1, 4096])

    def get_retrieval_embeds_query_aware(
        self,
        input_query_ids,
        query_attention_mask,
        input_ids,
        attention_mask=None,
    ):
        embeds = self.retriever.get_query_qware_embedding(
            input_ids, attention_mask, input_query_ids, query_attention_mask
        )
        # def get_query_qware_embedding(self,doc_inputs,doc_attention,query_inputs,query_attention,num_chunks=3,sim_mode="cosine")
        embeds = embeds.view(-1, embeds.shape[-1])
        # print(embeds.shape)
        return embeds