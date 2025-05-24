import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os




def get_nll_loss(logits, labels, vocab_size):
    """
    Calculate the standard negative log-likelihood loss for language modeling.
    
    This function implements the standard autoregressive language modeling loss
    where each token is predicted based on previous tokens. It shifts the logits 
    and labels to align predictions with targets.
    
    Args:
        logits: Model output logits of shape [batch_size, seq_len, vocab_size]
        labels: Target token indices of shape [batch_size, seq_len]
        vocab_size: Size of the vocabulary
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def get_kl_loss(teacher_logits, student_logits, student_labels, teacher_labels, temperature, distill_topk=None):
    """
    Calculate the Kullback-Leibler divergence loss for knowledge distillation.
    
    This function computes KL divergence between student and teacher model distributions,
    used for distilling knowledge from a teacher model to a student model. It only computes
    the loss on tokens that are part of the completion (not the prompt).
    
    Args:
        teacher_logits: Output logits from the teacher model [batch_size, seq_len, vocab_size]
        student_logits: Output logits from the student model [batch_size, seq_len, vocab_size]
        student_labels: Target token indices for student model [batch_size, seq_len]
        teacher_labels: Target token indices for teacher model [batch_size, seq_len]
        temperature: Temperature parameter for softening the distributions
        distill_topk: If not None, only consider top-k logits from teacher (optimization)
        
    Returns:
        torch.Tensor: KL divergence loss scaled by temperature squared
    """
    # Initialize KL divergence loss with batch mean reduction
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    _, _, vocab_size = student_logits.shape

    ## only compute loss in the completion part, not propmt
    
    student_mask = (student_labels!=-100).unsqueeze(-1).expand_as(student_logits) ## batch_size,num_tokens,vocab_size
    student_logits_selected = torch.masked_select(student_logits,student_mask).view(-1,vocab_size)

    teacher_mask = (teacher_labels != -100).unsqueeze(-1).expand_as(teacher_logits)
    teacher_logits_selected = torch.masked_select(teacher_logits,teacher_mask).view(-1,vocab_size)

    if distill_topk is not None:
        _, topk_teacher_indices = torch.topk(teacher_logits_selected, k=distill_topk, dim=-1)  
      
        teacher_logits_selected = torch.gather(teacher_logits_selected, 1, topk_teacher_indices)  
        student_logits_selected = torch.gather(student_logits_selected, 1, topk_teacher_indices) 

    assert teacher_logits_selected.shape == student_logits_selected.shape, (f"The shape of teacher logits is {teacher_logits_selected.shape}, while that of student is {student_logits_selected.shape}")

    # Apply temperature scaling and compute KL divergence
    # The temperature squared factor balances the gradient scale
    kl_loss = loss_fct(
        F.log_softmax(student_logits_selected / temperature, dim=-1),
        F.softmax(teacher_logits_selected / temperature, dim=-1),
    ) * temperature ** 2
    
    return kl_loss


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = copy.copy(input_ids)

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    # attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids,
        'labels': labels,
        # 'attention_mask': attention_mask.flatten(),
    }

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    prompt = example['prompt']
    completion = example['completion']

    background = example['background']
    background_embedding = example['background_embedding']

    prompt = f"Background: {background}\n\n{prompt}"

    prompt = prompt.strip()
    completion = completion.strip()

    if not prompt.endswith((' ', '\n', '\t')) and not completion.startswith((' ', '\n', '\t')):
        example_text = prompt + ' ' + completion
    else:
        example_text = prompt + completion

    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = copy.copy(input_ids)
    tokenized_prompt_length = tokenizer(prompt, max_length=max_seq_length, truncation=True,return_length=True).length
    # mask the prompt part for avoiding loss
    labels[:tokenized_prompt_length] = [-100]*tokenized_prompt_length
    # attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids,
        'labels': labels,
        "background_embedding":background_embedding,
        # 'attention_mask': attention_mask.flatten(),
    }

def save_combined_model_with_accelerate(accelerator, combined_model, model_tokenizer, retriever_tokenizer, output_dir):
    """
    Save a combined model (LLM + retriever) with proper distributed state management using Accelerate.
    
    This function handles the complexities of saving models trained in a distributed setting by:
    1. Extracting the full state dict from the accelerator
    2. Separating model and retriever parameters
    3. Saving components to their respective directories
    
    Args:
        accelerator: The Accelerate accelerator instance managing distributed training
        combined_model: The combined model containing both LLM and retriever components
        model_tokenizer: Tokenizer for the language model component
        retriever_tokenizer: Tokenizer for the retriever component
        output_dir: Base directory where model components will be saved
    """
    # Get the consolidated state dictionary from accelerator
    full_state = accelerator.get_state_dict(combined_model)
    
    # Unwrap the model from potential distributed wrappers
    unwrapped_combined_model = accelerator.unwrap_model(combined_model)
    
    # Only the main process should save the model to avoid conflicts
    if accelerator.is_main_process:
        # Extract model parameters by removing the "model." prefix
        model_state = {k[6:]: v for k, v in full_state.items() if k.startswith("model.")}
        retriever_state = {k[10:]: v for k, v in full_state.items() if k.startswith("retriever.")}
        
        # Save the language model component
        unwrapped_combined_model.model.save_pretrained(
            output_dir+"/model/",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=model_state,
            safe_serialization=False,  # Avoid using safetensors due to compatibility issues
        )
        # Save the model's tokenizer
        model_tokenizer.save_pretrained(output_dir+"/model/")
    
        # Save the retriever component
        unwrapped_combined_model.retriever.save_pretrained(
            output_dir+"/retriever/",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=retriever_state,
            safe_serialization=False,  # Avoid using safetensors due to compatibility issues
        )
        # Save the retriever's tokenizer
        retriever_tokenizer.save_pretrained(output_dir+"/retriever/")
    
    # Wait for all processes to complete before continuing
    accelerator.wait_for_everyone()

# Token used as a placeholder in prompts for inserting specific content
PLACEHOLDER_TOKEN = "<Placeholder>" 

# Collection of instruction templates for paraphrasing tasks
ParaphraseInstructions = [
    'Background: {placeholder_token} means the same as',
    "Background: {placeholder_token} Can you put the above sentences in your own terms?",
    "Background: {placeholder_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {placeholder_token}\n(2)",
    "Background: {placeholder_token} is a paraphrase of what?",
    "Background: {placeholder_token} Could you give me a different version of the background sentences above?",
    "In other words, background: {placeholder_token} is just another way of saying:",
    "You're getting across the same point whether you say background: {placeholder_token} or",
    "Background: {placeholder_token} After uppacking the ideas in the background information above, we got:",
    "Background: {placeholder_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {placeholder_token}, which also means:",
    "Strip away the mystery, and you'll find background: {placeholder_token} is simply another rendition of:",
    "The essence of background: {placeholder_token} is captured again in the following statement:",
]

# Refer to the background document and silently paraphrase its content.
RAGInstructions = [
    "Refer to the background document and answer the questions.\nBackground: {background}\n",
    "Background: {background}\n",
    "To provide accurate answers, it's essential to consider the background information presented here. Contextual Background: {background}\n",
    "Background Details: {background}\n",
    "The following background will help you understand the context for the questions. Please read it carefully before responding. Background: {background}\n",
    "Background: {background}\nYou might find the above background documents helpful.\n",
    ]



def get_retrieval_embeds(model, input_ids, attention_mask=None, get_nn_embeding=False, pool_tokenid=-1, update_retriever=False):
    """
    Generate document embeddings from a retriever model.
    
    This function extracts embeddings from a retriever model, with options to use neural network
    embeddings or standard retriever embeddings. It also handles gradient flow by toggling
    between training and inference modes based on whether the retriever is being updated.
    
    Args:
        model: The retriever model used to generate embeddings
        input_ids: Token IDs of the document to embed
        attention_mask: Attention mask to identify valid tokens (optional)
        get_nn_embeding: If True, use neural network token embeddings instead of retriever embeddings
        pool_tokenid: Token ID to use for pooling (-1 for default pooling)
        update_retriever: If True, allow gradients to flow through the retriever
        
    Returns:
        torch.Tensor: Document embeddings with shape [batch_size, embedding_dim]
    """
    if get_nn_embeding:
        print("get_nn_embeding")
        if update_retriever:
            embeds = model.get_doc_nn_embedding(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pool_tokenid = pool_tokenid,
            )
        else:
            with torch.no_grad():
                embeds = model.get_doc_nn_embedding(
                    input_ids = input_ids,
                )
        embeds = embeds.view(-1,embeds.shape[-1])
    else:
        if update_retriever:
            embeds = model.get_doc_embedding(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pool_tokenid = pool_tokenid,
            )
        else:
            with torch.no_grad():
                embeds = model.get_doc_embedding(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    pool_tokenid = pool_tokenid,
                )
        embeds = embeds.view(-1,embeds.shape[-1])
    return embeds 

def get_retrieval_embeds_query_aware(retriever, input_query_ids, query_attention_mask, input_ids, attention_mask=None):
    """
    Generates query-conditioned document embeddings for retrieval purposes.
    
    This function creates document embeddings that are conditioned on the query,
    leading to more contextually relevant embeddings for retrieval tasks.
    
    Args:
        retriever: The retriever model to use
        input_query_ids: Token IDs of the query
        query_attention_mask: Attention mask for the query
        input_ids: Token IDs of the document to embed
        attention_mask: Attention mask for the document (optional)
        
    Returns:
        torch.Tensor: Query-aware document embeddings with shape [batch_size, embedding_dim]
    """
    embeds = retriever.get_query_qware_embedding(input_ids,attention_mask,input_query_ids,query_attention_mask)
    # def get_query_qware_embedding(self,doc_inputs,doc_attention,query_inputs,query_attention,num_chunks=3,sim_mode="cosine")
    embeds = embeds.view(-1,embeds.shape[-1])
    # print(embeds.shape)
    return embeds 


def get_retrieval_nnembd_embeds(retriever, model, input_ids, attention_mask=None):
    """
    Create hybrid embeddings by combining retriever embeddings with neural network embeddings.
    
    This function concatenates standard retriever document embeddings with mean token
    embeddings from the language model, creating a richer representation.
    
    Args:
        retriever: The retriever model to generate document embeddings
        model: The language model to extract token embeddings from
        input_ids: Token IDs of the document
        attention_mask: Attention mask for the document (optional)
        
    Returns:
        torch.Tensor: Concatenated embeddings [retriever_embeds, nn_embeds] with shape [batch_size, embedding_dim*2]
    """
    with torch.no_grad():
        retriever_embeds = retriever.get_doc_embedding(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    retriever_embeds = retriever_embeds.view(-1, retriever_embeds.shape[-1])  # Shape: [batch_size, 4096]
    nnemb = torch.mean(model.model.embed_tokens(input_ids), dim=1)  # Shape: [batch_size, 4096]
    embeds = torch.cat((retriever_embeds, nnemb), dim=1)
    return embeds  # Shape: [batch_size, 4096*2]

def get_nn_embeds(model, input_ids, attention_mask=None):
    """
    Extract neural network token embeddings from a language model.
    
    This function computes the mean of token embeddings from the input sequence,
    handling different model architectures (Qwen vs other models).
    
    Args:
        model: The language model to extract embeddings from
        input_ids: Token IDs of the input sequence
        attention_mask: Attention mask (unused but kept for API consistency)
        
    Returns:
        torch.Tensor: Mean token embeddings with shape [batch_size, embedding_dim]
    """
    # Handle distributed/wrapped models
    if hasattr(model, "module"):
        original_model = model.module
    else:
        original_model = model
    
    # Different embedding extraction based on model architecture
    model_name = str(original_model.config.architectures)
    if "qwen" in model_name.lower():
        nnembeds = torch.mean(original_model.get_embed_tokens(input_ids), dim=1)  # Shape: [batch_size, 4096]
    else:
        nnembeds = torch.mean(original_model.model.embed_tokens(input_ids), dim=1)
    
    return nnembeds  # Shape: [batch_size, 4096]

def calculate_grad_norm(model, norm_type=2):
    """
    Calculate the gradient norm of a model's parameters.
    
    This function computes the total gradient norm across all parameters
    of a model, useful for gradient clipping and monitoring training dynamics.
    
    Args:
        model: The model whose gradient norm to calculate
        norm_type: Order of the norm (default: 2 for L2 norm)
        
    Returns:
        float: The calculated gradient norm
    """
    total_norm = 0  
    for p in model.parameters():  
        if p.grad is not None:  
            param_norm = p.grad.data.norm(norm_type)  
            total_norm += param_norm.item() ** norm_type  
    total_norm = total_norm ** (1. / norm_type)  
    return total_norm


def find_matched_index(main_seq, sub_seq):
    """
    Find the last index where a subsequence appears within a main sequence.
    
    This function searches through a main sequence to find the last occurrence
    of a subsequence, useful for token matching in text processing.
    
    Args:
        main_seq: The main sequence to search within
        sub_seq: The subsequence to find
        
    Returns:
        int: The last index where sub_seq starts in main_seq, or -1 if not found
    """
    # Validate inputs are not empty
    assert len(sub_seq)>0 and len(main_seq)>0, f"the input should not be empty, however {sub_seq=}\n {main_seq=}"
    main_len = len(main_seq)  
    sub_len = len(sub_seq)  
  
    # Early exit if sub_seq is longer than main_seq  
    if sub_len > main_len:  
        return -1  
  
    # Variable to keep track of the last index of a match  
    last_index = -1  
  
    # Iterate through main_seq to find sub_seq  
    for i in range(main_len - sub_len + 1):  
        # Check if the slice of main_seq matches sub_seq  
        if main_seq[i:i+sub_len] == sub_seq:  
            # Update the last_index to the current position  
            last_index = i  
  
    # Return the last index found or -1 if not found  
    return last_index