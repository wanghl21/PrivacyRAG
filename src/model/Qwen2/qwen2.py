import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from transformers import Qwen2Model


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
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
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, token_id]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), token_id]

class Qwen2(Qwen2Model):

    def get_embed_dim(self):
        return self.config.hidden_size
    
    def get_embed_length(self):
        return 1
    
    def get_embedding(self,input_ids,attention_mask,pool_tokenid=-1):
        outputs = self.forward(input_ids=input_ids,attention_mask=attention_mask)
        if pool_tokenid == -1:
            embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        else:
            embeddings = specify_token_pool(outputs.last_hidden_state, attention_mask, pool_tokenid)
        return embeddings
    def get_doc_nn_embedding(self,input_ids):
        if len(input_ids)>2:
            nnembeds = torch.mean(self.embed_tokens(input_ids),dim=1)# torch.Size([1, 4096])
        else:
            nnembeds = self.embed_tokens(input_ids)
            print("nnembeds")
        return nnembeds
    def get_doc_last_hidden_embedding(self,input_ids,attention_mask):
        outputs = self.forward(input_ids=input_ids,attention_mask=attention_mask)
        return outputs.last_hidden_state
    
    def get_doc_embedding(self,input_ids,attention_mask,pool_tokenid=-1):
        return self.get_embedding(input_ids,attention_mask,pool_tokenid)
    
    def get_query_embedding(self,input_ids,attention_mask):
        return self.get_embedding(input_ids,attention_mask)

    def get_query_qware_embedding(self,doc_inputs,doc_attention,query_inputs,query_attention,num_chunks=3,sim_mode="cosine"):
        
        
        query_embeds = self.get_embedding(query_inputs,query_attention)
        doc_embeds = self.forward(input_ids=doc_inputs,attention_mask=doc_attention).last_hidden_state# [batch_size, seq_len, hidden_dim]

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
        # Expand query_emb to match content_embs
        query_embeds = query_embeds.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Compute similarity
        sim_scores = torch.sum(query_embeds * doc_chunks_embs, dim=-1)  # [batch_size, num_chunks]
        weights = F.softmax(sim_scores, dim=-1)  # [batch_size, num_chunks]

        # Weighted sum
        pooled_emb = torch.sum(weights.unsqueeze(-1) * doc_chunks_embs, dim=1)  # [batch_size, hidden_dim]
        return pooled_emb
