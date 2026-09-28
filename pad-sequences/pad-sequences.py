import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    
    if max_len is None:
        L = max((len(seq) for seq in seqs), default=0)
    else:
        L = max_len

    #creating array full with pad_values
    arr=np.full((N,L),pad_value,dtype=int)

    #copying to output array
    for i,seq in enumerate(seqs):
        length=min(len(seq),L)
        arr[i, :length]=seq[:length]
    return arr    
    pass