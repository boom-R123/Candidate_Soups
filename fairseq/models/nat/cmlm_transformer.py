# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nat import NATransformerModel
from fairseq.utils import new_arange
import numpy as np


def judge_finished(c_list, idx_list):
    for i in range(len(c_list)):
        if c_list[i][idx_list[i]] == 2 or c_list[i][idx_list[i]] == 1:
            return False
    return True

def judge_same(c_list, idx_list):
    for i in range(len(c_list)-1):
        if c_list[i][idx_list[i]] != c_list[i+1][idx_list[i+1]]:
            return False
    return True

def judge_small(idx_list, Len):
    for i in range(len(idx_list)):
        if idx_list[i] < Len - 1:
            return True
    return False

def add_hash(c_list, idx_list1, hash_list):
    for i in range(len(c_list)):
        if c_list[i][idx_list1[i]] not in hash_list[i].keys():
            hash_list[i][c_list[i][idx_list1[i]]] = idx_list1[i]
    return hash_list

def find_same(c_list, idx_list1, hash_list):
    for i in range(len(c_list)):
        flag = True
        for j in range(len(hash_list)):
            if i != j:
                if c_list[i][idx_list1[i]] not in hash_list[j].keys():
                    flag = False
                    break
        if flag:
            for j in range(len(hash_list)):
                if i != j:
                   idx_list1[j] = hash_list[j][c_list[i][idx_list1[i]]]
            break
    return idx_list1, flag

def delete_repeat(c_list, s_list):
    for i in range(len(c_list)):
        c_new = np.ones((c_list[i].shape[0]))
        s_new = np.ones((s_list[i].shape[0]))
        c_new[0] = c_list[i][0]
        s_new[0] = s_list[i][0]
        j, k = 1, 1
        while j < c_list[i].shape[0] and c_list[i][j] != 2:
            if c_list[i][j] == c_new[k-1]:
                s_new[k] = max(s_list[i][j], s_list[i][j-1])
                j += 1
            else:
                c_new[k] = c_list[i][j]
                s_new[k] = s_list[i][j]
                j += 1
                k += 1
        c_new[k] = 2
        c_list[i] = c_new
        s_list[i] = s_new
    return c_list, s_list

def combine(c, s):
    c_clone = c.clone()
    c = c.cpu().numpy()
    s = s.cpu().numpy()
    res = torch.ones((c.shape[1])).cpu().numpy()
    num = c.shape[0]
    Len = c.shape[1]
    c_list = [c[i] for i in range(num)]
    s_list = [s[i] for i in range(num)]
    idx_list = [1 for i in range(num)]
    l = 1
    res[0] = 0
    c_list, s_list = delete_repeat(c_list, s_list)
    segment = 0
    while judge_finished(c_list, idx_list) and l < Len:
        for i in range(num):
            while c_list[i][idx_list[i]] == res[l - 1]:
                idx_list[i] += 1
        if judge_same(c_list, idx_list):
            res[l] = c_list[0][idx_list[0]]
            l += 1
            idx_list = [i+1 for i in idx_list]
        else:
            segment += 1
            hash_list = [{c_list[i][idx_list[i]]: idx_list[i]} for i in range(num)]
            idx_list1 = [i+1 for i in idx_list]
            while judge_small(idx_list1, Len):
                hash_list = add_hash(c_list, idx_list1, hash_list)
                idx_list1, flag = find_same(c_list, idx_list1, hash_list)
                if flag:
                    break
                idx_list1 = [i + 1 if i < Len-1 else i for i in idx_list1]
            sum_list = [s_list[i][idx_list[i]-1:idx_list1[i]+1].mean() for i in range(num)] 
            max_idx = sum_list.index(max(sum_list))
            res[l:l + idx_list1[max_idx] - idx_list[max_idx]] = c_list[max_idx][idx_list[max_idx]:idx_list1[max_idx]]
            l = l + idx_list1[max_idx] - idx_list[max_idx]
            idx_list = idx_list1
    if l < Len:
        res[l] = 2
    res = torch.from_numpy(res).cuda().type_as(c_clone)
    return res.unsqueeze(0).repeat(num, 1), segment

def combine_all(c_all, s_all, beam_size):
    res, segment = combine(c_all[0:beam_size], s_all[0:beam_size])
    segmets = [segment]
    s = beam_size
    while s < c_all.shape[0]:
        res_tmp, segment = combine(c_all[s:s + beam_size], s_all[s:s + beam_size])
        res = torch.cat((res, res_tmp), 0)
        segmets.append(segment)
        s += beam_size
    return res, segmets

def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("cmlm_transformer")
class CMLMNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, train_ratio=None, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        word_ins_mask = prev_output_tokens.eq(self.unk)

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, reranker=None, encoder_input=None, batch=1, Use_CDS=False, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        ).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())
                  
        if (step + 1) == max_step:

            num = int(output_tokens.shape[0] / batch)
            if num > 1 and Use_CDS:       
                if reranker is not None:
                    encoder_input[0] = encoder_input[0][:, 1:]   #AT的输入的起始位置没有 0 token
                    encoder_input[1] = encoder_input[1] - 1
                    reranker_encoder_out = reranker.encoder(*encoder_input)
                    output_tokens[:, 0] = self.eos 
                    length_beam_order = (
                        utils.new_arange(
                            output_tokens, num, reranker_encoder_out["encoder_out"][0].size(1)
                        )
                        .t()
                        .reshape(-1)
                    )
                    reranker_encoder_out = reranker.encoder.reorder_encoder_out(
                        reranker_encoder_out, length_beam_order
                    )
                    reranking_scores = reranker.get_normalized_probs(
                        reranker.decoder(output_tokens[:, :-1], reranker_encoder_out),
                        True,
                        None,
                    )
                    reranking_scores = reranking_scores.gather(2, output_tokens[:, 1:, None]).squeeze(-1)
                    reranking_masks = output_tokens[:, 1:].ne(self.pad) & output_tokens[:, 1:].ne(self.eos)
                    reranking_scores = reranking_scores.masked_fill_(~reranking_masks, 0)
                    output_tokens[:, 0] = 0  
                    output_scores[:, 1:] = reranking_scores
                res, segments = combine_all(output_tokens, output_scores, num)
                output_tokens = res
            
            
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )
    
    
    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
                length_tgt[:, None]
                + utils.new_arange(length_tgt, 1, beam_size)
                - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length + 6)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length + 6
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

@register_model_architecture("cmlm_transformer", "cmlm_transformer")
def cmlm_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("cmlm_transformer", "cmlm_transformer_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)
