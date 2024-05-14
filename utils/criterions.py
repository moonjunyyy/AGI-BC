import torch
import torch.nn.functional as F

def get_criterion(mode):
    try:
        return {
            "focal": focal_loss,
            "counting": counting_loss,
            "hierarchical": hierarchical_loss,
            "no_one_left_behind": no_one_left_behind,
            "mean_pooling": cross_entropy_loss,
            "audio_only" : cross_entropy_loss,
            "text_only" : cross_entropy_loss,
            "flatten" : cross_entropy_loss,
        }[mode]
    except:
        raise NotImplementedError

def focal_loss(batch, output):
    gamma = 2
    softmax = F.softmax(output['logit'], dim=-1)
    loss = softmax[torch.arange(len(batch['label'])), batch['label']]
    loss = - (1 - loss) ** gamma * (loss + 1e-6).log()
    return loss, output['logit']

def counting_loss(batch, output):
    unique, count = batch['label'].unique(return_counts=True)
    count = count[(unique == batch['label'].unsqueeze(1)).nonzero()[:,1]]
    loss = F.cross_entropy(output['logit'], batch["label"], reduction='none')        
    loss = loss / count * len(batch['label'])
    return loss.mean(), output['logit']

def hierarchical_loss(batch, output):
    logit = output['logit']
    BC_logit = output['logit_BC']
    
    bc_loss = F.cross_entropy(BC_logit, (batch['label'] > 0).long(), reduction='none')
    logit = logit[batch['label'] > 0]
    loss = F.cross_entropy(logit, batch["label"][batch['label'] > 0] - 1, reduction='none')
    loss = loss.mean() + bc_loss.mean()

    softmax = F.softmax(output["logit"], dim=-1)
    BC_softmax = F.softmax(output["logit_BC"], dim=-1)
    logit = torch.cat((
        BC_softmax[:,:1], BC_softmax[:,1:] * softmax
    ),dim=1)
    return loss, logit

def no_one_left_behind(batch, output):
    device = batch['label'].device
    num_class = output['logit'].shape[-1]

    unique, count = batch['label'].unique(return_counts=True)
    logit = output['logit']
    logit = logit - logit.max(dim=-1, keepdim=True)[0]
    logit = logit.exp()
    
    cnt = torch.zeros(num_class, device=device)
    for c in range(num_class):
        if c in unique:
            cnt[batch['label'] == c] = count[unique == c].item()
        else:
            cnt[batch['label'] == c] = 0
    logit = logit * cnt.unsqueeze(-1)
    p = logit[torch.arange(len(batch['label'])), batch['label']]
    logit = p / logit.sum(dim=-1)
    logit = logit.squeeze()

    loss = 0
    for c, u in enumerate(unique):
        loss = loss - torch.log(logit[batch["label"] == u] + 1e-6).sum() / count[c]
    loss = loss / u
    return loss, output['logit']

def cross_entropy_loss(batch, output):
    loss_BC = F.cross_entropy(output['logit'], batch["label"], reduction='none')
    if 'sentiment' in output.keys():
        loss_SP = F.binary_cross_entropy(torch.sigmoid(output['sentiment']), batch["sentiment"])
        loss = 0.9 * loss_BC + 0.1 * loss_SP
        return loss.mean(), output['logit']
    else:
        return loss_BC.mean(), output['logit']