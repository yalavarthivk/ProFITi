import torch
import pdb


def obs_rep(tx, cx, mx, x, tq, cq, mq, C):
    """
    convert the data into a format suitable for the conditional module
    tx: Tensor - Time points of the observations.
    cx: Tensor - Channel ID for the observations.
    mx: Tensor - Mask for the observations.
    x: Tensor - Observed values.
    tq: Tensor - Time points for the query.
    cq: Tensor - Channel ID for the query.
    mq: Tensor - Mask for the query.
    C: int - Number of channels.

    Returns:
    K: Tensor - Processed observations in the form of a batch matrix.
    M: Tensor - Mask for the observations.
    Mq: Tensor - Mask for the query.
    unique_Ts: Tensor - Unique time points both observed and query for each batch.
    """

    batch_size = tx.shape[0]

    if (
        tx.shape[1] != cx.shape[1]
        or tx.shape[1] != mx.shape[1]
        or tx.shape[1] != x.shape[1]
    ):
        raise ValueError(
            "Time, channel, mask, and value tensors must have the same second dimension."
        )
    if tq.shape[1] != cq.shape[1] or tq.shape[1] != mq.shape[1]:
        raise ValueError(
            "Query time, channel, and mask tensors must have the same second dimension."
        )

    # --- mask valid entries ---

    mask_total = torch.cat(
        [mx, mq], dim=1
    ).bool()  # combined mask for both observations and queries
    time_total = torch.cat([tx, tq], dim=1)  # combined time points
    channel_total = torch.cat(
        [cx, cq], dim=1
    )  # combined channel IDs for both observations and queries
    value_total = torch.cat([x, torch.zeros_like(tq)], dim=1)  # combined values
    obs_mask = torch.cat(
        [mx, torch.zeros_like(mq)], dim=1
    )  # mask for observations dimension extended for queries as well
    qry_mask = torch.cat([torch.zeros_like(mx), mq], dim=1)  # mask for queries
    T, D, V, Mobs, Mq = (
        time_total[mask_total],
        channel_total[mask_total],
        value_total[mask_total],
        obs_mask[mask_total],
        qry_mask[mask_total],
    )  # Extract valid entries based on the combined mask

    batch_valid = torch.arange(batch_size, device=tx.device).repeat_interleave(
        mask_total.sum(1)
    )  # Batch indices for valid entries

    # --- offset trick to separate batches ---
    t_max = T.max().item() + 1  # Maximum time value + 1 to avoid overlap
    t_off = T + batch_valid * t_max  # Offset time points to separate batches
    unique_t_off, inverse = torch.unique(
        t_off, sorted=True, return_inverse=True
    )  # Unique time points with offsets

    unique_b = unique_t_off // t_max  # which batch each unique belongs to
    unique_t = unique_t_off % t_max  # actual time values

    # number of unique times per batch
    n_per_batch = torch.bincount(unique_b.long(), minlength=batch_size)
    n_max = int(n_per_batch.max())

    # --- compute row_idx for each entry (for K) ---
    cumsum = torch.zeros(batch_size + 1, device=tx.device, dtype=torch.long)
    cumsum[1:] = torch.cumsum(n_per_batch, 0)
    row_idx = inverse - cumsum[batch_valid]  # per-entry row index

    # --- build K with accumulation ---
    k = torch.zeros((batch_size, n_max, C), dtype=x.dtype, device=x.device)
    k.index_put_((batch_valid, row_idx, D), V, accumulate=True)

    # --- build mask with accumulation ---
    mobs = torch.zeros((batch_size, n_max, C), dtype=x.dtype, device=x.device)
    mobs.index_put_((batch_valid, row_idx, D), Mobs.to(torch.float32), accumulate=True)

    mq = torch.zeros((batch_size, n_max, C), dtype=x.dtype, device=x.device)
    mq.index_put_((batch_valid, row_idx, D), Mq.to(torch.float32), accumulate=True)

    # --- compute row_idx_u for uniques (for unique_Ts) ---
    # basically 0..N_per_batch[b]-1 for each batch
    row_idx_u = (
        torch.arange(unique_t.shape[0], device=tx.device) - cumsum[unique_b.long()]
    )

    # --- build unique_Ts ---
    unique_ts = torch.zeros((batch_size, n_max), dtype=tx.dtype, device=tx.device)
    unique_ts[unique_b.long(), row_idx_u] = unique_t

    return (
        unique_ts,
        k,
        mobs,
        mq,
    )
