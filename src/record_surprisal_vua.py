import pandas as pd
import transformers
import torch

from tqdm import tqdm
import string
import argparse
from pathlib import Path


######################################
BOW_SYMBOL = "Ġ"  # This works fine and tested for GPT-2, LLama3.1, Llama3.2, Pythia, and Qwen2.5.
######################################
# Helper functions
def blank_target(s: str, start: int, end: int, replacement: str) -> str:
    if start < 0:
        start = 0
    if end > len(s):
        end = len(s)
    return s[:start] + replacement + s[end:]

def subtoken_indices_for_char_span(offset_mapping, start_char, end_char):
    """
    offset_mapping: list[(start,end)] for each subtoken in the encoded sequence
    Returns the list of token positions whose spans overlap [start_char, end_char).
    """
    idxs = []
    for i, (s, e) in enumerate(offset_mapping):
        # ignore special tokens that sometimes have (0,0)
        if s == 0 and e == 0:
            continue
        # overlap test with [start_char, end_char)
        if s < end_char and e > start_char:
            idxs.append(i)
    return idxs

def find_nth(haystack: str, needle: str, n: int = 0) -> int:
    start = 0
    for _ in range(n + 1):
        pos = haystack.find(needle, start)
        if pos == -1:
            return -1
        start = pos + 1
    return pos

def subtoken_indices_for_sentence_span_in_prompt(
    prompt: str,
    sentence: str,
    offset_mapping,
    sent_start_char: int,   # in *sentence-local* coordinates
    sent_end_char: int,     # in *sentence-local* coordinates
    occurrence: int = 0     # if sentence appears multiple times in prompt
):
    sent_pos = find_nth(prompt, sentence, n=occurrence)
    if sent_pos == -1:
        raise ValueError("Sentence not found in prompt. Add markers or pass the sentence start index explicitly.")

    abs_start = sent_pos + sent_start_char
    abs_end   = sent_pos + sent_end_char

    return subtoken_indices_for_char_span(offset_mapping, abs_start, abs_end)

def build_pimentel_masks(tokenizer, model, bow_symbol="Ġ"):
    device = next(model.parameters()).device

    V = model.get_output_embeddings().out_features

    vocab = tokenizer.get_vocab()  # token_str -> id

    bow_ids = [i for tok, i in vocab.items()
               if i < V and tok and tok[0] == bow_symbol]

    punct_ids = [i for tok, i in vocab.items()
                 if i < V and tok and tok[0] in string.punctuation]

    eos_id = tokenizer.eos_token_id

    bow_mask   = torch.zeros(V, device=device, dtype=torch.float32)
    punct_mask = torch.zeros(V, device=device, dtype=torch.float32)
    eos_mask   = torch.zeros(V, device=device, dtype=torch.float32)

    if bow_ids:
        bow_mask[torch.tensor(bow_ids, device=device)] = 1.0
    if punct_ids:
        punct_mask[torch.tensor(punct_ids, device=device)] = 1.0

    if eos_id is not None and eos_id < V:
        eos_mask[eos_id] = 1.0
        punct_mask[eos_id] = 0.0

    useless_mask = torch.zeros(V, device=device, dtype=torch.float32)
    if tokenizer.vocab_size < V:
        useless_mask[tokenizer.vocab_size:] = 1.0

    mid_mask = (torch.ones(V, device=device, dtype=torch.float32)
                - bow_mask - punct_mask - useless_mask - eos_mask).clamp_(0.0, 1.0)

    return {"bow": bow_mask, "punct": punct_mask, "eos": eos_mask, "mid": mid_mask}

def compute_pimentel_fixes_from_logits(logits, masks, eps=None):
    device = logits.device
    bow_vocab = (masks["bow"].to(device=device) + masks["eos"].to(device=device)).clamp(max=1.0)
    bos_vocab = (masks["mid"].to(device=device) + masks["punct"].to(device=device) + masks["eos"].to(device=device)).clamp(max=1.0)

    probs = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)

    if eps is None:
        eps = torch.finfo(probs.dtype).tiny

    p_bow = (probs * bow_vocab).sum(-1).clamp_min(eps)  # [B, T]
    p_bos = (probs * bos_vocab).sum(-1).clamp_min(eps)  # [B, T]

    bow_fix = -torch.log(p_bow)
    bos_fix = -torch.log(p_bos)

    eow_fix = torch.empty_like(bow_fix)
    eow_fix[:, :-1] = bow_fix[:, 1:]
    eow_fix[:, -1]  = bow_fix[:, -1]

    return bow_fix, bos_fix, eow_fix

######################################
# Arg parser

def parse_args():
    ap = argparse.ArgumentParser(description="Measuring word-level surprisals of metaphoric words in VUAMC dataset.")
    
    ap.add_argument("--input", required=True, help="JSON with at least 'sentence', 'offsets' and 'vua_metaphor_labels' ")
    ap.add_argument("--output", required=True, help="Output directory to save results.")
    ap.add_argument("--model", default="", help="HF model name or path.")
    ap.add_argument("--model-revision", default=None, help="Model revision (branch, tag, commit id) if applicable.")
    ap.add_argument("--dtype", choices=["float16","bfloat16","float32"], default="float16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map-auto", action="store_true")
    ap.add_argument("--pimentel-fix", action="store_true")
    ap.add_argument("--cloze", action="store_true")
    
    return ap.parse_args()

######################################
# Main function

def main():
    args = parse_args()
    ####################################
    df = pd.read_json(args.input, lines=True)
    print(f"Loaded {len(df)} examples from {args.input}")
    ####################################
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to {output_dir}")
    ####################################
    assert 'sentence' in df.columns, "'sentence' column not found in input data."
    assert 'offsets' in df.columns, "'offsets' column not found in input data."
    assert 'vua_metaphor_labels' in df.columns, "'vua_metaphor_labels' column not found in input data."
    ####################################
    sentences = df['sentence'].tolist()
    offsets_list = df['offsets'].tolist()
    vua_metaphor_labels = df['vua_metaphor_labels'].tolist()
    ####################################
    model_id = args.model
    assert model_id, "Model name or path must be specified."
    ####################################
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    assert dtype is not None, f"Unsupported dtype: {args.dtype}"
    ####################################
    if args.device_map_auto:
        device_map = "auto"
    else:
        device_map = args.device
    assert device_map is not None, "Device map must be specified."
    ####################################
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True
    )
    if tokenizer.bos_token_id is None and tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer does not have a BOS or EOS token defined.")
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
    else:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    ####################################
    kwargs = {
        "device_map": device_map,
    }
    if args.model_revision:
        kwargs["revision"] = args.model_revision
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        **kwargs
    )
    model.eval()
    device = next(model.parameters()).device
    ####################################
    if args.pimentel_fix:
        pimentels_masks = build_pimentel_masks(tokenizer, model, bow_symbol=BOW_SYMBOL)
    
    df['subtoken_ids'] = None
    df['subtoken_strs'] = None
    df['surprisal_buggy'] = None
    if args.pimentel_fix:
        df['surprisal_fixed'] = None
    for i, sent in tqdm(enumerate(sentences), total=len(sentences), desc="Processing sentences"):
        enc = tokenizer(
            sent,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        offset_mapping = enc["offset_mapping"][0].tolist()
        B, T0 = input_ids.shape
        eos_col = torch.full((B, 1), tokenizer.eos_token_id, dtype=input_ids.dtype, device=input_ids.device)
        labels = torch.cat([input_ids, eos_col], dim=1)    # [B, T0+1] labels are ids of the original input without BOS and include the last token (eos).
        T = labels.shape[1]
        offset_mapping = offset_mapping + [(0, 0)]
        
        bos_col = torch.full((B, 1), tokenizer.bos_token_id, dtype=labels.dtype, device=labels.device)
        input_ids_bos = torch.cat([bos_col, labels[:, :-1]], dim=1)  # [B, T] input is prepended with BOS and last token is removed as causal models predict next token and we don't want to predict the token after the last token already (eos).

        with torch.no_grad():
            out = model(input_ids=input_ids_bos)

        logits = out.logits
        
        if args.pimentel_fix:
            bow_fix_tok, bos_fix_tok, eow_fix_tok = compute_pimentel_fixes_from_logits(logits, pimentels_masks)
            
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
        surprisal_tok = -tok_lp  # nats
        
        label_ids = labels[0].tolist()
        label_toks = tokenizer.convert_ids_to_tokens(label_ids)
        
        if args.pimentel_fix:
            is_bow_all = torch.tensor([tok.startswith(BOW_SYMBOL) for tok in label_toks], device=device, dtype=torch.bool)

            is_eos_all = torch.tensor([tid == tokenizer.eos_token_id for tid in label_ids], device=device, dtype=torch.bool)
            
            is_bos_all = torch.zeros(T, device=device, dtype=torch.bool)
            is_bos_all[0] = True
            
            is_eow_all = torch.zeros(T, device=device, dtype=torch.bool)
            is_eow_all[:-1] = is_bow_all[1:] | is_eos_all[1:]
            is_eow_all[-1] = True
        
        offset = offsets_list[i]
        surprisals_buggy_per_offset = []
        surprisals_fixed_per_offset = []
        subtokens_ids_list = []
        subtokens_str_list = []
        for offs in offset:
            subtokens_span = subtoken_indices_for_char_span(offset_mapping, offs[0], offs[1])
            subtokens_span = torch.tensor(subtokens_span, device=device, dtype=torch.long)

            surp_sub = surprisal_tok[0, subtokens_span]
            surprisals_buggy_per_offset.append(surp_sub.detach().cpu().numpy().sum().item())
            if args.pimentel_fix:
                bow_sub  = bow_fix_tok[0, subtokens_span]
                bos_sub  = bos_fix_tok[0, subtokens_span]
                eow_sub  = eow_fix_tok[0, subtokens_span]

                is_bow_sub = is_bow_all[subtokens_span].float()
                is_bos_sub = is_bos_all[subtokens_span].float()
                is_eow_sub = is_eow_all[subtokens_span].float()

                surp_fixed_sub = (
                    surp_sub
                    - bos_sub * is_bos_sub
                    - bow_sub * is_bow_sub
                    + eow_sub * is_eow_sub
                )
                surprisals_fixed_per_offset.append(surp_fixed_sub.detach().cpu().numpy().sum().item())

            subtokens_ids = [labels[0][idx].item() for idx in subtokens_span]
            subtokens_str = tokenizer.decode(subtokens_ids)
            
            subtokens_ids_list.append(subtokens_ids)
            subtokens_str_list.append(subtokens_str)
        
        df.at[i, 'subtoken_ids'] = subtokens_ids_list
        df.at[i, 'subtoken_strs'] = subtokens_str_list
        df.at[i, 'surprisal_buggy'] = surprisals_buggy_per_offset
        if args.pimentel_fix:
            df.at[i, 'surprisal_fixed'] = surprisals_fixed_per_offset
             
    #####################################
    
    if args.cloze:
        df['subtoken_ids_cloze'] = None
        df['subtoken_strs_cloze'] = None
        df['surprisal_buggy_cloze'] = None
        if args.pimentel_fix:
            df['surprisal_fixed_cloze'] = None
        for i, sent in tqdm(enumerate(sentences), total=len(sentences), desc="Processing sentences"):
            subtokens_ids_list = [None] * len(offsets_list[i])
            subtokens_str_list = [None] * len(offsets_list[i])
            surprisals_buggy_list = [None] * len(offsets_list[i])
            surprisals_fixed_list = [None] * len(offsets_list[i])
            for j, met_label in enumerate(vua_metaphor_labels[i]):
                if met_label == True:
                    offs = offsets_list[i][j]
                    blanked_sent = blank_target(sent, offs[0], offs[1], "______")
                    full_sent = sent
                    prompt = f'You are a helpful assistant that can predict a word to replace a blank space in a sentence.\nHere is a sentence with a blank space: " {blanked_sent} ".\nThe sentence after replacing the blank space with the predicted word is: " {full_sent} ".'
                    enc = tokenizer(
                        prompt,
                        return_offsets_mapping=True,
                        return_tensors="pt",
                        add_special_tokens=False
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    input_ids = enc["input_ids"]
                    offset_mapping = enc["offset_mapping"][0].tolist()
                    B, T0 = input_ids.shape

                    eos_col = torch.full((B, 1), tokenizer.eos_token_id, dtype=input_ids.dtype, device=input_ids.device)
                    labels = torch.cat([input_ids, eos_col], dim=1)    # [B, T0+1]
                    T = labels.shape[1]
                    
                    offset_mapping = offset_mapping + [(0, 0)]         # length T
                    
                    bos_col = torch.full((B, 1), tokenizer.bos_token_id, dtype=labels.dtype, device=labels.device)
                    input_ids_bos = torch.cat([bos_col, labels[:, :-1]], dim=1)  # [B, T] input is prepended with BOS and last token is removed as causal models predict next token and we don't want to predict the token after the last token already (eos).
                    
                    with torch.no_grad():
                        out = model(input_ids=input_ids_bos)
                    
                    logits = out.logits
                    
                    if args.pimentel_fix:
                        bow_fix_tok, bos_fix_tok, eow_fix_tok = compute_pimentel_fixes_from_logits(logits, pimentels_masks)
                    
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    tok_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
                    surprisal_tok = -tok_lp  # nats
                    
                    label_ids = labels[0].tolist()
                    label_toks = tokenizer.convert_ids_to_tokens(label_ids)
                    
                    if args.pimentel_fix:
                        is_bow_all = torch.tensor([tok.startswith(BOW_SYMBOL) for tok in label_toks], device=device, dtype=torch.bool)

                        is_eos_all = torch.tensor([tid == tokenizer.eos_token_id for tid in label_ids], device=device, dtype=torch.bool)
                        
                        is_bos_all = torch.zeros(T, device=device, dtype=torch.bool)
                        is_bos_all[0] = True
                        
                        is_eow_all = torch.zeros(T, device=device, dtype=torch.bool)
                        is_eow_all[:-1] = is_bow_all[1:] | is_eos_all[1:]
                        is_eow_all[-1] = True
                    
                    subtokens_span = subtoken_indices_for_sentence_span_in_prompt(
                        prompt,
                        sent,
                        offset_mapping,
                        offs[0],
                        offs[1],
                    )
                    subtokens_span = torch.tensor(subtokens_span, device=device, dtype=torch.long)

                    surp_sub = surprisal_tok[0, subtokens_span]
                    surprisals_buggy_list[j] = surp_sub.detach().cpu().numpy().sum().item()
                    if args.pimentel_fix:
                        bow_sub  = bow_fix_tok[0, subtokens_span]
                        bos_sub  = bos_fix_tok[0, subtokens_span]
                        eow_sub  = eow_fix_tok[0, subtokens_span]

                        is_bow_sub = is_bow_all[subtokens_span].float()
                        is_bos_sub = is_bos_all[subtokens_span].float()
                        is_eow_sub = is_eow_all[subtokens_span].float()

                        surp_fixed_sub = (
                            surp_sub
                            - bos_sub * is_bos_sub
                            - bow_sub * is_bow_sub
                            + eow_sub * is_eow_sub
                        )
                        surprisals_fixed_list[j] = surp_fixed_sub.detach().cpu().numpy().sum().item()
                    
                    subtokens_ids = [labels[0][idx].item() for idx in subtokens_span]
                    subtokens_str = tokenizer.decode(subtokens_ids)
                    
                    subtokens_ids_list[j] = subtokens_ids
                    subtokens_str_list[j] = subtokens_str
                    
            df.at[i, 'subtoken_ids_cloze'] = subtokens_ids_list
            df.at[i, 'subtoken_strs_cloze'] = subtokens_str_list
            df.at[i, 'surprisal_buggy_cloze'] = surprisals_buggy_list
            if args.pimentel_fix:
                df.at[i, 'surprisal_fixed_cloze'] = surprisals_fixed_list
                
    
    #####################################
    #####################################
    save_name = f"vua-metanov_surprisal_{model_id.replace('/', '_')}{'_cloze' if args.cloze else ''}{'_pimentel' if args.pimentel_fix else ''}{'_' + args.model_revision if args.model_revision else ''}.parquet"
    df.to_parquet(output_dir / save_name, index=False)
    print(f"\nSaved results to {output_dir / save_name}")

##################################

if __name__ == "__main__":
    main()