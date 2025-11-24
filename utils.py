import torch


def sample_batched(model, num_samples, batch_size, n_sample_steps, clip_samples):
    samples = []
    for i in range(0, num_samples, batch_size):
        corrected_batch_size = min(batch_size, num_samples - i)
        samples.append(model.sample(corrected_batch_size, n_sample_steps, clip_samples))
    return torch.cat(samples, dim=0)
