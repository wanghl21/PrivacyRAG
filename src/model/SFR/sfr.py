import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from transformers import MistralForCausalLM, MistralModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    """
    Extract the embedding of the last token from model's hidden states.
    
    Handles both left-padded and right-padded sequences appropriately.
    For left padding, simply returns the last token representation.
    For right padding, uses attention mask to find the actual last token position.
    
    Args:
        last_hidden_states: Hidden states from the model output, shape [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask indicating valid tokens, shape [batch_size, seq_len]
        
    Returns:
        Tensor: Last token embeddings, shape [batch_size, hidden_dim]
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def specify_token_pool(last_hidden_states: Tensor,
                       attention_mask: Tensor,
                 token_id:int) -> Tensor:
    """
    Extract the embedding of a specific token position from model's hidden states.
    
    Similar to last_token_pool but allows specifying which token position to extract.
    Handles both padding types correctly.
    
    Args:
        last_hidden_states: Hidden states from the model output, shape [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask indicating valid tokens, shape [batch_size, seq_len]
        token_id: Position of the token to extract
        
    Returns:
        Tensor: Specified token embeddings, shape [batch_size, hidden_dim]
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, token_id]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), token_id]

class SFR(MistralModel):
    """
    Semantic Frame Retriever model based on Mistral architecture.
    
    This model generates embeddings for documents and queries, with
    various pooling strategies for different retrieval requirements.
    """

    def get_embed_dim(self):
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            int: Hidden dimension size from model config
        """
        return self.config.hidden_size
    
    def get_embed_length(self):
        """
        Get the number of embeddings produced per document.
        
        Returns:
            int: 1 since this model produces a single embedding per document
        """
        return 1
    
    def get_embedding(self, input_ids, attention_mask, pool_tokenid=-1):
        """
        Generate embeddings using the model with specified pooling strategy.
        
        Args:
            input_ids: Token IDs of the input sequence
            attention_mask: Mask identifying valid input tokens
            pool_tokenid: If -1, use last token pooling; otherwise, extract embedding at specified position
            
        Returns:
            Tensor: Pooled embeddings with shape [batch_size, hidden_dim]
        """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        if pool_tokenid == -1:
            embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        else:
            embeddings = specify_token_pool(outputs.last_hidden_state, attention_mask, pool_tokenid)
        return embeddings
    
    def get_doc_nn_embedding(self, input_ids):
        """
        Generate document embeddings directly from token embeddings without model forward pass.
        
        This is a more efficient approach for some use cases, computing mean
        embeddings directly from the token embedding layer.
        
        Args:
            input_ids: Token IDs of the document
            
        Returns:
            Tensor: Mean token embeddings with shape [batch_size, hidden_dim]
        """
        if len(input_ids) > 2:
            nnembeds = torch.mean(self.embed_tokens(input_ids), dim=1)  # Shape: [batch_size, hidden_dim]
        else:
            nnembeds = self.embed_tokens(input_ids)
            print("nnembeds")
        return nnembeds
    
    def get_doc_last_hidden_embedding(self, input_ids, attention_mask):
        """
        Obtain the last hidden state embeddings for a document.
        
        This method runs the document through the model and retrieves the
        last hidden states, which can be used for various downstream tasks.
        
        Args:
            input_ids: Token IDs of the document
            attention_mask: Attention mask identifying valid tokens in the document
            
        Returns:
            Tensor: Last hidden state embeddings, shape [batch_size, seq_len, hidden_dim]
        """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def get_doc_embedding(self, input_ids, attention_mask, pool_tokenid=-1):
        """
        Get the embedding for a document using the specified pooling strategy.
        
        This method is essentially a wrapper around get_embedding for clarity.
        
        Args:
            input_ids: Token IDs of the document
            attention_mask: Attention mask identifying valid tokens
            pool_tokenid: Token ID for specific token pooling, if desired
            
        Returns:
            Tensor: Document embedding, shape [batch_size, hidden_dim]
        """
        return self.get_embedding(input_ids, attention_mask, pool_tokenid)
    
    def get_query_embedding(self, input_ids, attention_mask):
        """
        Retrieve the embedding for a query using the model.
        
        Queries are typically shorter than documents, but this method ensures
        they are processed through the same embedding pipeline.
        
        Args:
            input_ids: Token IDs of the query
            attention_mask: Attention mask for the query tokens
            
        Returns:
            Tensor: Query embedding, shape [batch_size, hidden_dim]
        """
        return self.get_embedding(input_ids, attention_mask)

    def get_query_qware_embedding(self, doc_inputs, doc_attention, query_inputs, query_attention, num_chunks=3, sim_mode="cosine"):
        """
        Compute query-aware document embeddings for improved retrieval performance.
        
        This method pools document embeddings into chunks and allows the query
        embedding to attend over these chunks, producing a final pooled embedding
        that is more representative of the query's intent.
        
        Args:
            doc_inputs: Token IDs of the document
            doc_attention: Attention mask for the document tokens
            query_inputs: Token IDs of the query
            query_attention: Attention mask for the query tokens
            num_chunks: Number of chunks to divide the document into for pooling
            sim_mode: Similarity mode, currently only 'cosine' is supported
            
        Returns:
            Tensor: Query-aware pooled embedding, shape [batch_size, hidden_dim]
        """
        
        query_embeds = self.get_embedding(query_inputs, query_attention)
        doc_embeds = self.forward(input_ids=doc_inputs, attention_mask=doc_attention).last_hidden_state  # [batch_size, seq_len, hidden_dim]

        seq_len = doc_embeds.shape[1]
        # print("seq_len",seq_len)
        chunk_size = seq_len // num_chunks

        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk_emb = doc_embeds[:, start:end, :].mean(dim=1) 
            chunks.append(chunk_emb)
        doc_chunks_embs = torch.stack(chunks, dim=1)  # [batch_size, num_chunks, hidden_dim]

        if sim_mode == "cosine":
            query_embeds = F.normalize(query_embeds, p=2, dim=-1)
            doc_chunks_embs = F.normalize(doc_chunks_embs, p=2, dim=-1)
        query_embeds = query_embeds.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Compute similarity
        sim_scores = torch.sum(query_embeds * doc_chunks_embs, dim=-1)  # [batch_size, num_chunks]
        weights = F.softmax(sim_scores, dim=-1)  # [batch_size, num_chunks]

        # Weighted sum
        pooled_emb = torch.sum(weights.unsqueeze(-1) * doc_chunks_embs, dim=1)  # [batch_size, hidden_dim]
        return pooled_emb
